#!/usr/bin/env python3
"""
SAM3 + LoRA Inference Script

Based on official SAM3 batched inference patterns.
Supports text prompts and visual prompts with LoRA fine-tuned weights.

Usage:
    # Text prompt inference
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" \
        --output output.png

    # Multiple prompts
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" "defect" "damage" \
        --output output.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms
from sam3.perflib.nms import nms_masks

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights


class SAM3LoRAInference:
    """SAM3 model with LoRA for inference."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        use_base_model: bool = False,
        gamma_preprocess: float = 1.0,
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        device: str = "cuda"
    ):
        """
        Initialize SAM3 with LoRA.

        Args:
            config_path: Path to training config YAML
            weights_path: Path to LoRA weights (optional, auto-detected from config)
            resolution: Input image resolution (default: 1008)
            detection_threshold: Confidence threshold for detections (default: 0.5)
            nms_iou_threshold: IoU threshold for NMS (default: 0.5)
            gamma_preprocess: Gamma correction factor applied before inference.
                Values < 1 brighten dark images, values > 1 darken them.
            use_clahe: Whether to apply CLAHE before inference.
            device: Device to run on (default: "cuda")
        """
        self.use_base_model = use_base_model
        if not use_base_model and config_path is None:
            raise ValueError("--config is required unless --use-base-model is set")

        # Load config when available. Base SAM3 inference intentionally ignores
        # trained adapters such as LoRA and SRF-Lite.
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # Auto-detect weights if not provided
        if not use_base_model and weights_path is None:
            output_dir = self.config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
            weights_path = os.path.join(output_dir, 'best_lora_weights.pt')
            print(f"鈩癸笍  Auto-detected weights: {weights_path}")

        if not use_base_model and not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        self.weights_path = weights_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.gamma_preprocess = gamma_preprocess
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        model_name = "base SAM3" if use_base_model else "SAM3 + LoRA"
        print(f"馃敡 Initializing {model_name}...")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")
        if abs(gamma_preprocess - 1.0) > 1e-6:
            print(f"   Gamma preprocess: {gamma_preprocess}")
        if use_clahe:
            print(
                "   CLAHE: "
                f"clip_limit={clahe_clip_limit}, tile_grid={clahe_tile_grid_size}"
            )

        # Build base model
        print("\n馃摝 Building SAM3 model...")
        model_cfg = {} if use_base_model else self.config.get("model", {})
        srf_cfg = model_cfg.get("srf_lite", {})
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True,
            use_srf_lite=bool(srf_cfg.get("enabled", False)),
            srf_num_levels=int(srf_cfg.get("num_levels", 4)),
            srf_bottleneck_dim=srf_cfg.get("bottleneck_dim", None),
            srf_interpolation_mode=str(srf_cfg.get("interpolation_mode", "bilinear")),
            srf_alpha_init=float(srf_cfg.get("alpha_init", 0.0)),
        )

        if use_base_model:
            print("Using original SAM3 model without LoRA weights.")
        else:
            # Apply LoRA configuration
            print("馃敆 Applying LoRA configuration...")
            lora_cfg = self.config["lora"]
            lora_config = LoRAConfig(
                rank=lora_cfg["rank"],
                alpha=lora_cfg["alpha"],
                dropout=0.0,  # No dropout during inference
                target_modules=lora_cfg["target_modules"],
                apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
                apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
                apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
                apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
                apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
                apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
            )
            self.model = apply_lora_to_model(self.model, lora_config)

            # Load LoRA weights
            print(f"馃捑 Loading LoRA weights from {weights_path}...")
            load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (official SAM3 pattern)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Setup postprocessor
        # Note: Using simpler manual postprocessing instead of PostProcessImage
        # because PostProcessImage may have additional filtering logic
        self.use_manual_postprocess = True

        print(f"鉁?{model_name} ready for inference!\n")

    def _apply_gamma_preprocess(self, pil_image: PILImage.Image) -> PILImage.Image:
        if abs(self.gamma_preprocess - 1.0) < 1e-6:
            return pil_image

        image_np = np.asarray(pil_image).astype(np.float32) / 255.0
        image_np = np.power(np.clip(image_np, 0.0, 1.0), self.gamma_preprocess)
        image_np = np.clip(image_np, 0.0, 1.0)
        return PILImage.fromarray((image_np * 255.0).astype(np.uint8))

    def _apply_clahe_preprocess(self, pil_image: PILImage.Image) -> PILImage.Image:
        if not self.use_clahe:
            return pil_image

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "CLAHE preprocessing requires OpenCV. Install opencv-python or disable --clahe."
            ) from exc

        rgb_np = np.asarray(pil_image)
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_grid_size, self.clahe_tile_grid_size),
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        rgb_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return PILImage.fromarray(rgb_np)

    def _preprocess_image_for_inference(
        self, pil_image: PILImage.Image
    ) -> PILImage.Image:
        processed = pil_image
        processed = self._apply_gamma_preprocess(processed)
        processed = self._apply_clahe_preprocess(processed)
        return processed

    def create_datapoint(self, pil_image: PILImage.Image, text_prompts: List[str]) -> Datapoint:
        """
        Create a SAM3 datapoint from image and text prompts.

        Args:
            pil_image: PIL Image
            text_prompts: List of text queries

        Returns:
            Datapoint with image and queries
        """
        w, h = pil_image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        # Create queries for each text prompt
        queries = []
        for idx, text_query in enumerate(text_prompts):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        text_prompts: List[str]
    ) -> dict:
        """
        Run inference on an image with text prompts.

        Args:
            image_path: Path to input image
            text_prompts: List of text queries (e.g., ["crack", "defect"])

        Returns:
            Dictionary mapping prompt index to predictions:
            {
                0: {'boxes': [...], 'scores': [...], 'masks': [...]},
                1: {'boxes': [...], 'scores': [...], 'masks': [...]}
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        original_pil_image = PILImage.open(image_path).convert("RGB")
        pil_image = self._preprocess_image_for_inference(original_pil_image)
        print(f"馃摲 Loaded image: {image_path}")
        print(f"   Size: {original_pil_image.size}")
        print(f"   Prompts: {text_prompts}")
        if pil_image is not original_pil_image:
            print("   Applied low-light preprocessing before inference")

        print("\n馃敭 Running inference...")

        results = {}

        # Process each prompt separately (SAM3 expects one query per forward pass)
        for query_idx, prompt in enumerate(text_prompts):
            # Create datapoint with single prompt
            datapoint = self.create_datapoint(pil_image, [prompt])

            # Apply transforms
            datapoint = self.transform(datapoint)

            # Collate into batch
            batch = collate_fn_api([datapoint], dict_key="input")["input"]

            # Move to device
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(batch)

            # Manual post-processing
            last_output = outputs[-1]
            pred_logits = last_output['pred_logits']  # [batch, num_queries, num_classes]
            pred_boxes = last_output['pred_boxes']    # [batch, num_queries, 4]
            pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

            # Get probabilities
            out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

            # Get scores for this query
            scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

            # Filter by threshold
            keep = scores > self.detection_threshold
            num_keep = keep.sum().item()

            if num_keep > 0:
                # Get boxes and masks selected by score.
                boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
                kept_scores = scores[keep]
                kept_masks = pred_masks[0, keep] if pred_masks is not None else None

                if kept_masks is not None and len(kept_scores) > 0:
                    nms_keep_mask = nms_masks(
                        pred_probs=kept_scores,
                        pred_masks=(kept_masks.sigmoid() > 0.5).float(),
                        prob_threshold=0.0,
                        iou_threshold=self.nms_iou_threshold,
                    )
                    boxes_cxcywh = boxes_cxcywh[nms_keep_mask]
                    kept_scores = kept_scores[nms_keep_mask]
                    kept_masks = kept_masks[nms_keep_mask]

                cx, cy, w, h = boxes_cxcywh.unbind(-1)

                # Convert to xyxy and scale to original image size
                orig_w, orig_h = original_pil_image.size
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                if kept_masks is None and len(kept_scores) > 0:
                    keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
                    boxes_xyxy = boxes_xyxy[keep_nms]
                    kept_scores = kept_scores[keep_nms]
                num_keep = len(kept_scores)

                # Get masks and resize to original size
                if kept_masks is not None:
                    # Resize sigmoid probabilities, then threshold at original size.
                    import torch.nn.functional as F
                    masks_resized = F.interpolate(
                        kept_masks.sigmoid().unsqueeze(0).float(),
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0) > 0.5

                    masks_np = masks_resized.cpu().numpy()
                else:
                    masks_np = None

                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': boxes_xyxy.cpu().numpy(),
                    'scores': kept_scores.cpu().numpy(),
                    'masks': masks_np,
                    'num_detections': num_keep
                }
                print(f"   '{prompt}': {num_keep} detections after NMS (max score: {kept_scores.max().item():.3f})")
            else:
                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': None,
                    'scores': None,
                    'masks': None,
                    'num_detections': 0
                }
                print(f"   '{prompt}': 0 detections")

        # Store original image for visualization
        results['_image'] = original_pil_image
        if pil_image is not original_pil_image:
            results['_model_image'] = pil_image

        return results

    def visualize(
        self,
        results: dict,
        output_path: str,
        show_boxes: bool = True,
        show_masks: bool = True
    ):
        """
        Visualize predictions on image.

        Args:
            results: Results from predict()
            output_path: Where to save visualization
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show segmentation masks
        """
        pil_image = results['_image']

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(pil_image)

        # Colors for different prompts
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

        total_detections = 0
        result_indices = sorted([k for k in results.keys() if isinstance(k, int)])

        # Draw results for each prompt
        for idx in result_indices:
            result = results[idx]
            prompt = result['prompt']
            color = colors[idx % len(colors)]

            if result['num_detections'] == 0:
                continue

            total_detections += result['num_detections']

            boxes = result['boxes']
            scores = result['scores']
            masks = result['masks']

            for i in range(result['num_detections']):
                # Draw mask
                if show_masks and masks is not None:
                    mask = masks[i]
                    colored_mask = np.zeros((*mask.shape, 4))
                    # Use different colors for different prompts
                    if color == 'red':
                        colored_mask[mask] = [1, 0, 0, 0.4]
                    elif color == 'blue':
                        colored_mask[mask] = [0, 0, 1, 0.4]
                    elif color == 'green':
                        colored_mask[mask] = [0, 1, 0, 0.4]
                    else:
                        colored_mask[mask] = [1, 1, 0, 0.4]
                    ax.imshow(colored_mask)

                # Draw box
                if show_boxes and boxes is not None:
                    box = boxes[i]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    img_w, img_h = pil_image.size
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add label
                    score = scores[i] if scores is not None else 0
                    label = f"{prompt}: {score:.2f}"
                    ax.text(
                        x1, y1 - 5,
                        label,
                        bbox=dict(facecolor=color, alpha=0.5),
                        fontsize=10,
                        color='white'
                    )

        ax.axis('off')

        # Add title with all prompts
        prompts_str = ", ".join([f'"{results[k]["prompt"]}"' for k in result_indices])
        plt.suptitle(f'Text Prompts: {prompts_str}', fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"\n鉁?Saved visualization to {output_path}")
        print(f"   Total detections: {total_detections}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Inference")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML. Required unless --use-base-model is set."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights (auto-detected if not provided)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Run original SAM3 without applying or loading LoRA weights"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs='+',
        default=["object"],
        help='Text prompt(s) to guide segmentation (e.g., "crack" or "crack" "defect")'
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output visualization path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)"
    )
    parser.add_argument(
        "--boundingbox",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Show bounding boxes: True or False (default: False)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5, lower = fewer overlapping boxes)"
    )
    parser.add_argument(
        "--gamma-preprocess",
        type=float,
        default=1.0,
        help="Apply gamma correction before inference. Values < 1 brighten dark images."
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE to the luminance channel before inference."
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (default: 2.0)"
    )
    parser.add_argument(
        "--clahe-tile-grid-size",
        type=int,
        default=8,
        help="CLAHE tile grid size (default: 8)"
    )

    args = parser.parse_args()

    # Initialize model
    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        use_base_model=args.use_base_model,
        gamma_preprocess=args.gamma_preprocess,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid_size=args.clahe_tile_grid_size,
    )

    # Run inference
    results = inferencer.predict(args.image, args.prompt)

    # Visualize
    inferencer.visualize(
        results,
        args.output,
        show_boxes=args.boundingbox,
        show_masks=not args.no_masks
    )

    # Print summary
    print("\n" + "="*60)
    print("馃搳 Summary:")
    for idx in sorted([k for k in results.keys() if isinstance(k, int)]):
        result = results[idx]
        print(f"   Prompt '{result['prompt']}': {result['num_detections']} detections")
        if result['num_detections'] > 0 and result['scores'] is not None:
            print(f"      Max confidence: {result['scores'].max():.3f}")
    print("="*60)


if __name__ == "__main__":
    main()


