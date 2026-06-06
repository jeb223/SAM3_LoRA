# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .model_misc import MLP


def _mark_trainable_adapter(module: nn.Module) -> nn.Module:
    """Mark lightweight residual adapters so LoRA setup can keep them trainable."""
    module._sam3_trainable_adapter = True
    return module


def _pick_gn_groups(num_channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model):
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        return super().forward(hs)


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim, mask_dim):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries, pixel_embed):
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum(
                    "bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
        else:
            # Assumed to have aux masks
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum(
                    "lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )

        return mask_preds


class MultiScaleFusionAdapter(nn.Module):
    """Residual fusion block on top of the original top-down FPN path."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_batch_dim(curr_fpn, prev_fpn_upsampled):
        """
        Preserve the original PixelDecoder broadcasting behavior.

        The old implementation used `curr_fpn + prev_fpn_upsampled`, which allows
        singleton batch dimensions to broadcast (e.g. [1, C, H, W] + [B, C, H, W]).
        Our concat-based fusion needs the batch dimensions to match explicitly.
        """
        if curr_fpn.shape[0] == prev_fpn_upsampled.shape[0]:
            return curr_fpn, prev_fpn_upsampled

        if curr_fpn.shape[0] == 1:
            curr_fpn = curr_fpn.expand(
                prev_fpn_upsampled.shape[0], -1, -1, -1
            )
            return curr_fpn, prev_fpn_upsampled

        if prev_fpn_upsampled.shape[0] == 1:
            prev_fpn_upsampled = prev_fpn_upsampled.expand(
                curr_fpn.shape[0], -1, -1, -1
            )
            return curr_fpn, prev_fpn_upsampled

        raise RuntimeError(
            "MultiScaleFusionAdapter got incompatible batch dimensions: "
            f"{tuple(curr_fpn.shape)} vs {tuple(prev_fpn_upsampled.shape)}"
        )

    def forward(self, curr_fpn, prev_fpn_upsampled):
        curr_fpn, prev_fpn_upsampled = self._match_batch_dim(
            curr_fpn, prev_fpn_upsampled
        )
        residual = self.fusion(torch.cat([curr_fpn, prev_fpn_upsampled], dim=1))
        return curr_fpn + prev_fpn_upsampled + self.residual_scale * residual


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        act: bool = True,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_pick_gn_groups(out_ch), out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DepthEncoderLite(nn.Module):
    """Lightweight encoder for D_norm, D_edge, and D_valid depth guidance."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 256,
        base_dim: int = 64,
        num_levels: int = 4,
        interpolation_mode: str = "bilinear",
    ):
        super().__init__()
        self.num_levels = num_levels
        self.interpolation_mode = interpolation_mode

        blocks = []
        projs = []
        in_ch = input_channels
        for level in range(num_levels):
            out_ch = min(base_dim * (2**level), hidden_dim)
            blocks.append(
                ConvGNAct(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=True,
                )
            )
            projs.append(
                ConvGNAct(
                    out_ch,
                    hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    act=True,
                )
            )
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.projs = nn.ModuleList(projs)

    def _resize_to(self, x: torch.Tensor, size_hw):
        if x.shape[-2:] == size_hw:
            return x
        if self.interpolation_mode in ("bilinear", "bicubic", "trilinear"):
            return F.interpolate(
                x,
                size=size_hw,
                mode=self.interpolation_mode,
                align_corners=False,
            )
        return F.interpolate(x, size=size_hw, mode=self.interpolation_mode)

    def forward(self, depth_batch: torch.Tensor, target_sizes: List[torch.Size]):
        if depth_batch is None:
            return None
        if depth_batch.ndim == 3:
            depth_batch = depth_batch.unsqueeze(0)

        x = depth_batch.float()
        encoded = []
        for block in self.blocks:
            x = block(x)
            encoded.append(x)

        depth_feats = []
        for feat, proj, size_hw in zip(encoded, self.projs, target_sizes):
            depth_feats.append(self._resize_to(proj(feat), size_hw))
        return depth_feats


class SRFLiteFusion(nn.Module):
    """Shallow-guided residual multi-scale fusion."""

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int = 4,
        bottleneck_dim: Optional[int] = None,
        interpolation_mode: str = "bilinear",
        alpha_init: float = 0.0,
        use_depth_guidance: bool = False,
        depth_alpha_init: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.interpolation_mode = interpolation_mode
        self.use_depth_guidance = bool(use_depth_guidance)

        if bottleneck_dim is None:
            bottleneck_dim = max(hidden_dim // 4, 32)
        bottleneck_dim = min(bottleneck_dim, hidden_dim)
        self.bottleneck_dim = bottleneck_dim

        self.align_layers = nn.ModuleList(
            [
                ConvGNAct(
                    hidden_dim,
                    bottleneck_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    act=True,
                )
                for _ in range(num_levels)
            ]
        )

        if self.use_depth_guidance:
            self.depth_align_layers = nn.ModuleList(
                [
                    ConvGNAct(
                        hidden_dim,
                        bottleneck_dim,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        act=True,
                    )
                    for _ in range(num_levels)
                ]
            )
            self.depth_gate = nn.Sequential(
                nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(bottleneck_dim, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            self.depth_alpha = nn.Parameter(torch.tensor(float(depth_alpha_init)))
        else:
            self.depth_align_layers = None
            self.depth_gate = None
            self.depth_alpha = None

        gate_mid = max(bottleneck_dim // 2, 8)
        self.attn_gate = nn.Sequential(
            nn.Conv2d(bottleneck_dim, gate_mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(gate_mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            ConvGNAct(
                bottleneck_dim * num_levels,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                act=True,
            ),
            ConvGNAct(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                act=False,
            ),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.out_norm = nn.GroupNorm(_pick_gn_groups(hidden_dim), hidden_dim)

    @staticmethod
    def _match_batch_dim(feats: List[torch.Tensor]) -> List[torch.Tensor]:
        target_bs = max(feat.shape[0] for feat in feats)
        aligned = []
        for feat in feats:
            if feat.shape[0] == target_bs:
                aligned.append(feat)
            elif feat.shape[0] == 1:
                aligned.append(feat.expand(target_bs, -1, -1, -1))
            else:
                raise RuntimeError(
                    "SRFLiteFusion got incompatible batch dimensions: "
                    f"{[tuple(x.shape) for x in feats]}"
                )
        return aligned

    @staticmethod
    def _match_batch_dim_to(
        feats: List[torch.Tensor],
        target_bs: int,
    ) -> List[torch.Tensor]:
        aligned = []
        for feat in feats:
            if feat.shape[0] == target_bs:
                aligned.append(feat)
            elif feat.shape[0] == 1:
                aligned.append(feat.expand(target_bs, -1, -1, -1))
            else:
                raise RuntimeError(
                    "SRFLiteFusion depth features cannot be broadcast to RGB batch: "
                    f"target_bs={target_bs}, depth_shapes={[tuple(x.shape) for x in feats]}"
                )
        return aligned

    def _resize_to(self, x: torch.Tensor, size_hw):
        if x.shape[-2:] == size_hw:
            return x
        if self.interpolation_mode in ("bilinear", "bicubic", "trilinear"):
            return F.interpolate(
                x,
                size=size_hw,
                mode=self.interpolation_mode,
                align_corners=False,
            )
        return F.interpolate(x, size=size_hw, mode=self.interpolation_mode)

    def forward(
        self,
        feats: List[torch.Tensor],
        depth_feats: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert len(feats) == self.num_levels, (
            f"Expected {self.num_levels} feature levels, got {len(feats)}"
        )

        feats = self._match_batch_dim(feats)
        target_hw = feats[0].shape[-2:]
        aligned = []
        for feat, proj in zip(feats, self.align_layers):
            x = proj(feat)
            x = self._resize_to(x, target_hw)
            aligned.append(x)

        if self.use_depth_guidance and depth_feats is not None:
            assert len(depth_feats) == self.num_levels, (
                f"Expected {self.num_levels} depth feature levels, got {len(depth_feats)}"
            )
            depth_feats = self._match_batch_dim(depth_feats)
            depth_feats = self._match_batch_dim_to(
                depth_feats,
                target_bs=aligned[0].shape[0],
            )
            depth_aligned = []
            for feat, proj in zip(depth_feats, self.depth_align_layers):
                x = proj(feat.to(dtype=aligned[0].dtype))
                x = self._resize_to(x, target_hw)
                depth_aligned.append(x)

            depth_gate = self.depth_gate(
                torch.cat([aligned[0], depth_aligned[0]], dim=1)
            )
            aligned = [
                rgb + self.depth_alpha * depth_gate * depth
                for rgb, depth in zip(aligned, depth_aligned)
            ]

        gate = self.attn_gate(aligned[0])
        gated_feats = [aligned[0]]
        for x in aligned[1:]:
            gated_feats.append(x * gate)

        fused = self.fuse(torch.cat(gated_feats, dim=1))
        fused = self.out_norm(fused)
        return self.alpha * fused


class DepthContextFusionAdapter(nn.Module):
    """Gate depth context into pixel features with a trainable residual scale."""

    def __init__(self, hidden_dim: int, alpha_init: float = 0.0):
        super().__init__()
        self.depth_proj = ConvGNAct(
            hidden_dim,
            hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act=True,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @staticmethod
    def _as_batched(x: torch.Tensor):
        if x.ndim == 3:
            return x.unsqueeze(0), True
        return x, False

    @staticmethod
    def _match_batch_dim(depth_context: torch.Tensor, pixel_embed: torch.Tensor):
        if depth_context.shape[0] == pixel_embed.shape[0]:
            return depth_context
        if depth_context.shape[0] == 1:
            return depth_context.expand(pixel_embed.shape[0], -1, -1, -1)
        raise RuntimeError(
            "Depth context batch dimension does not match pixel features: "
            f"{tuple(depth_context.shape)} vs {tuple(pixel_embed.shape)}"
        )

    def forward(
        self,
        pixel_embed: torch.Tensor,
        depth_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if depth_context is None:
            return pixel_embed

        pixel_batched, squeezed = self._as_batched(pixel_embed)
        depth_batched, _ = self._as_batched(depth_context)
        depth_batched = self._match_batch_dim(depth_batched, pixel_batched)
        if depth_batched.shape[-2:] != pixel_batched.shape[-2:]:
            depth_batched = F.interpolate(
                depth_batched,
                size=pixel_batched.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        depth_feat = self.depth_proj(depth_batched.to(dtype=pixel_batched.dtype))
        gate = self.gate(torch.cat([pixel_batched, depth_feat], dim=1))
        fused = pixel_batched + self.alpha * gate * depth_feat
        return fused.squeeze(0) if squeezed else fused


class BoundaryRefinementAdapter(nn.Module):
    """Lightweight boundary stream that refines pixel features and predicts edges."""

    def __init__(
        self,
        hidden_dim,
        use_depth_guidance: bool = False,
        depth_alpha_init: float = 0.0,
    ):
        super().__init__()
        self.use_depth_guidance = bool(use_depth_guidance)
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.depth_fusion = (
            DepthContextFusionAdapter(hidden_dim, alpha_init=depth_alpha_init)
            if self.use_depth_guidance
            else None
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, pixel_embed, depth_context: Optional[torch.Tensor] = None):
        refined_input = (
            self.depth_fusion(pixel_embed, depth_context)
            if self.depth_fusion is not None
            else pixel_embed
        )
        boundary_features = self.refine(refined_input)
        enhanced_pixel_embed = pixel_embed + self.residual_scale * boundary_features
        boundary_logits = self.boundary_head(boundary_features)
        return enhanced_pixel_embed, boundary_logits


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
        use_depth_boundary=False,
        depth_boundary_alpha_init=0.0,
    ):
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(
                hidden_dim, 1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.boundary_adapter = _mark_trainable_adapter(
            BoundaryRefinementAdapter(
                self.pixel_decoder.out_dim,
                use_depth_guidance=use_depth_boundary,
                depth_alpha_init=depth_boundary_alpha_init,
            )
        )

        self.act_ckpt = act_ckpt

        # used to update the output dictionary
        self.instance_keys = ["pred_masks"]

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _embed_pixels(
        self,
        backbone_feats: List[torch.Tensor],
        image_ids,
        encoder_hidden_states,
        depth_batch: Optional[torch.Tensor] = None,
    ):
        feature_device = backbone_feats[0].device  # features could be on CPU
        model_device = self.device
        image_ids_ = image_ids.to(feature_device)
        if self.use_encoder_inputs:
            depth_for_decoder = None
            if backbone_feats[0].shape[0] > 1:
                # For bs > 1, we construct the per query backbone features
                backbone_visual_feats = []
                for feat in backbone_feats:
                    # Copy the img features per query (pixel decoder won't share img feats)
                    backbone_visual_feats.append(feat[image_ids_, ...].to(model_device))
                if depth_batch is not None:
                    depth_for_decoder = depth_batch[
                        image_ids.to(depth_batch.device), ...
                    ].to(model_device)
            else:
                # Bs=1, we rely on broadcasting for query-based processing
                backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]
                if depth_batch is not None:
                    depth_for_decoder = depth_batch.to(model_device)
            # Extract visual embeddings
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
            spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
            encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
                -1, *backbone_feats[-1].shape[1:]
            )

            backbone_visual_feats[-1] = encoder_visual_embed
            if self.act_ckpt:
                pixel_embed, depth_context = checkpoint.checkpoint(
                    self.pixel_decoder,
                    backbone_visual_feats,
                    depth_for_decoder,
                    use_reentrant=False,
                )
            else:
                pixel_embed, depth_context = self.pixel_decoder(
                    backbone_visual_feats,
                    depth_batch=depth_for_decoder,
                )
        else:
            backbone_feats = [x.to(model_device) for x in backbone_feats]
            depth_for_decoder = (
                depth_batch.to(model_device) if depth_batch is not None else None
            )
            pixel_embed, depth_context = self.pixel_decoder(
                backbone_feats,
                depth_batch=depth_for_decoder,
            )
            if pixel_embed.shape[0] == 1:
                # For batch_size=1 training, we can avoid the indexing to save memory
                pixel_embed = pixel_embed.squeeze(0)
                if depth_context is not None:
                    depth_context = depth_context.squeeze(0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
                if depth_context is not None:
                    depth_context = depth_context[image_ids, ...]
        return pixel_embed, depth_context

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        depth_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed, depth_context = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            depth_batch=depth_batch,
        )
        pixel_embed, pred_boundaries = self.boundary_adapter(
            pixel_embed,
            depth_context=depth_context,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred, "pred_boundaries": pred_boundaries}


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
        use_srf_lite=False,
        srf_num_levels=4,
        srf_bottleneck_dim=None,
        srf_interpolation_mode="bilinear",
        srf_alpha_init=0.0,
        use_depth_guidance=False,
        depth_input_channels=3,
        depth_encoder_dim=64,
        depth_alpha_init=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        self.use_depth_guidance = bool(use_depth_guidance)
        conv_layers = []
        norms = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.fusion_adapters = nn.ModuleList(
            [
                _mark_trainable_adapter(MultiScaleFusionAdapter(self.hidden_dim))
                for _ in range(num_upsampling_stages)
            ]
        )
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels
        self.use_srf_lite = bool(use_srf_lite)
        max_available_levels = num_upsampling_stages + 1
        self.srf_num_levels = min(int(srf_num_levels), max_available_levels)
        if self.use_depth_guidance and self.use_srf_lite and self.srf_num_levels >= 2:
            self.depth_encoder = _mark_trainable_adapter(
                DepthEncoderLite(
                    input_channels=depth_input_channels,
                    hidden_dim=hidden_dim,
                    base_dim=depth_encoder_dim,
                    num_levels=self.srf_num_levels,
                    interpolation_mode=srf_interpolation_mode,
                )
            )
        else:
            self.depth_encoder = None
        if self.use_srf_lite and self.srf_num_levels >= 2:
            self.srf_lite = _mark_trainable_adapter(
                SRFLiteFusion(
                    hidden_dim=hidden_dim,
                    num_levels=self.srf_num_levels,
                    bottleneck_dim=srf_bottleneck_dim,
                    interpolation_mode=srf_interpolation_mode,
                    alpha_init=srf_alpha_init,
                    use_depth_guidance=self.use_depth_guidance,
                    depth_alpha_init=depth_alpha_init,
                )
            )
        else:
            self.srf_lite = None
        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=compile_mode, dynamic=True, fullgraph=True
            )
            # Needed to make checkpointing happy. But we don't know if the module is checkpointed, so we disable it by default.
            torch._dynamo.config.optimize_ddp = False

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        depth_batch: Optional[torch.Tensor] = None,
    ):
        # Assumes backbone features are already projected (C == hidden dim)

        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        pyramid_feats = [prev_fpn]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            prev_fpn_up = F.interpolate(
                prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode
            )
            prev_fpn = self.fusion_adapters[layer_idx](curr_fpn, prev_fpn_up)
            if self.shared_conv:
                # only one conv layer
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))
            pyramid_feats.append(prev_fpn)

        depth_context = None
        if self.srf_lite is not None and len(pyramid_feats) >= self.srf_num_levels:
            selected_feats = list(reversed(pyramid_feats[-self.srf_num_levels:]))
            depth_feats = None
            if self.depth_encoder is not None and depth_batch is not None:
                target_sizes = [feat.shape[-2:] for feat in selected_feats]
                depth_feats = self.depth_encoder(depth_batch, target_sizes)
                if depth_feats:
                    depth_context = depth_feats[0]
            prev_fpn = prev_fpn + self.srf_lite(
                selected_feats,
                depth_feats=depth_feats,
            )
            prev_fpn = F.relu(prev_fpn)

        return prev_fpn, depth_context


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation"""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
        use_depth_boundary=False,
        depth_boundary_alpha_init=0.0,
        use_depth_semantic_fusion=False,
        depth_semantic_alpha_init=0.0,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
            use_depth_boundary=use_depth_boundary,
            depth_boundary_alpha_init=depth_boundary_alpha_init,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, "Specifying a dot product scorer without a presence head is likely a mistake"

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.use_depth_semantic_fusion = bool(use_depth_semantic_fusion)
        self.semantic_depth_fusion = (
            _mark_trainable_adapter(
                DepthContextFusionAdapter(
                    self.pixel_decoder.out_dim,
                    alpha_init=depth_semantic_alpha_init,
                )
            )
            if self.use_depth_semantic_fusion
            else None
        )
        self.instance_seg_head = nn.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1
        )

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        depth_batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    pooled_enc.view(1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed, depth_context = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            depth_batch=depth_batch,
        )
        pixel_embed, pred_boundaries = self.boundary_adapter(
            pixel_embed,
            depth_context=depth_context,
        )

        instance_embeds = self.instance_seg_head(pixel_embed)

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        semantic_embed = (
            self.semantic_depth_fusion(pixel_embed, depth_context)
            if self.semantic_depth_fusion is not None
            else pixel_embed
        )

        return {
            "pred_masks": mask_pred,
            "pred_boundaries": pred_boundaries,
            "semantic_seg": self.semantic_seg_head(semantic_embed),
            "presence_logit": presence_logit,
        }
