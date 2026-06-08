#!/usr/bin/env python3
"""Evaluate TRef-SAM3 on referring-expression segmentation data.

Metrics:
  - mIoU: mean IoU over referring queries
  - cIoU: cumulative intersection / cumulative union
  - Pr@0.5 / Pr@0.7 / Pr@0.9

The evaluator uses the unified TRef format:
  annotations.json + referring_annotations.json
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from sam3.model.model_misc import SAM3Output
from sam3.model.tref_matching import TextCandidateMatchingHead, text_attribute_feature_dim
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from tref_sam3_dataset import ReferringSegmentDataset
from validate_sam3_lora import attach_tref_matching_scores, move_to_device


def build_model(config, weights_path, device):
    model_cfg = config.get("model", {})
    srf_cfg = model_cfg.get("srf_lite", {})
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
        use_srf_lite=bool(srf_cfg.get("enabled", False)),
        srf_num_levels=int(srf_cfg.get("num_levels", 4)),
        srf_bottleneck_dim=srf_cfg.get("bottleneck_dim", None),
        srf_interpolation_mode=str(srf_cfg.get("interpolation_mode", "bilinear")),
        srf_alpha_init=float(srf_cfg.get("alpha_init", 0.0)),
    )

    lora_cfg = config["lora"]
    model = apply_lora_to_model(
        model,
        LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        ),
    )

    tref_cfg = config.get("tref_matching", config["training"].get("tref_matching", {}))
    if bool(tref_cfg.get("enabled", False)):
        query_dim = int(getattr(model, "hidden_dim", 256))
        text_dim = text_attribute_feature_dim(int(tref_cfg.get("text_hash_dim", 32)))
        model.tref_matching_head = TextCandidateMatchingHead(
            query_dim=query_dim,
            text_feature_dim=text_dim,
            hidden_dim=int(tref_cfg.get("hidden_dim", 256)),
            dropout=float(tref_cfg.get("dropout", 0.1)),
        )

    load_lora_weights(model, weights_path)
    model.to(device)
    model.eval()
    return model


def build_dataset(config, data_dir, split):
    ref_cfg = config["training"].get("referring", {})
    return ReferringSegmentDataset(
        data_dir=data_dir,
        split=split,
        annotations_file=ref_cfg.get("annotations_file", "annotations.json"),
        refs_file=ref_cfg.get("refs_file", "referring_annotations.json"),
        max_queries_per_image=int(ref_cfg.get("eval_max_queries_per_image", 0)),
        include_multi_target=bool(ref_cfg.get("include_multi_target", True)),
        fallback_to_category_queries=False,
        training=False,
        resolution=int(ref_cfg.get("resolution", 1008)),
    )


def union_target_masks(targets, query_index, pred_hw):
    num_boxes = targets.num_boxes.detach().cpu().tolist()
    start = sum(int(n) for n in num_boxes[:query_index])
    count = int(num_boxes[query_index])
    if count <= 0 or targets.segments is None:
        return None, count

    target = targets.segments[start : start + count].any(dim=0).float()
    if tuple(target.shape[-2:]) != tuple(pred_hw):
        target = torch.nn.functional.interpolate(
            target[None, None],
            size=pred_hw,
            mode="nearest",
        ).squeeze(0).squeeze(0)
    return target.bool(), count


def compute_query_iou(pred_mask, target_mask):
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()
    inter = torch.logical_and(pred_mask, target_mask).sum().item()
    union = torch.logical_or(pred_mask, target_mask).sum().item()
    iou = inter / union if union > 0 else 0.0
    return iou, inter, union


def evaluate(
    config_path,
    weights_path,
    data_dir,
    split,
    output_json,
    batch_size=None,
    score_threshold=0.35,
    mask_threshold=0.5,
    num_samples=None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, weights_path, device)
    tref_cfg = config.get("tref_matching", config["training"].get("tref_matching", {}))
    dataset = build_dataset(config, data_dir, split)
    if batch_size is None:
        batch_size = int(config["training"].get("batch_size", 1))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_api(batch, dict_key="input", with_seg_masks=True),
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=True,
    )

    ious = []
    unique_ious = []
    multi_ious = []
    total_inter = 0.0
    total_union = 0.0
    total_queries = 0
    skipped_queries = 0

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(loader, desc="TRef validation")):
            if num_samples is not None and total_queries >= num_samples:
                break
            input_batch = move_to_device(batch_dict["input"], device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs_list = model(input_batch)
                    attach_tref_matching_scores(model, outputs_list, input_batch, tref_cfg)
            else:
                outputs_list = model(input_batch)
                attach_tref_matching_scores(model, outputs_list, input_batch, tref_cfg)

            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]
            targets = input_batch.find_targets[0]
            pred_logits = final_outputs["pred_logits"]
            pred_masks = final_outputs["pred_masks"]
            bsz = pred_logits.shape[0]

            for query_idx in range(bsz):
                if num_samples is not None and total_queries >= num_samples:
                    break
                target_mask, num_targets = union_target_masks(
                    targets,
                    query_index=query_idx,
                    pred_hw=pred_masks.shape[-2:],
                )
                if target_mask is None:
                    skipped_queries += 1
                    continue

                scores = pred_logits[query_idx].sigmoid().squeeze(-1)
                masks = pred_masks[query_idx].sigmoid() > mask_threshold
                if int(num_targets) > 1:
                    keep = scores > score_threshold
                    if not keep.any():
                        keep[scores.argmax()] = True
                    pred_mask = masks[keep].any(dim=0)
                else:
                    pred_mask = masks[scores.argmax()]

                iou, inter, union = compute_query_iou(pred_mask, target_mask)
                ious.append(iou)
                if int(num_targets) > 1:
                    multi_ious.append(iou)
                else:
                    unique_ious.append(iou)
                total_inter += inter
                total_union += union
                total_queries += 1

    def mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    metrics = {
        "config_path": str(config_path),
        "weights_path": str(weights_path),
        "data_dir": str(data_dir),
        "split": str(split),
        "num_queries": int(total_queries),
        "num_skipped_queries": int(skipped_queries),
        "mIoU": mean(ious),
        "cIoU": float(total_inter / total_union) if total_union > 0 else 0.0,
        "Pr@0.5": mean([1.0 if x >= 0.5 else 0.0 for x in ious]),
        "Pr@0.7": mean([1.0 if x >= 0.7 else 0.0 for x in ious]),
        "Pr@0.9": mean([1.0 if x >= 0.9 else 0.0 for x in ious]),
        "unique_mIoU": mean(unique_ious),
        "multi_target_mIoU": mean(multi_ious),
        "score_threshold": float(score_threshold),
        "mask_threshold": float(mask_threshold),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRef-SAM3 referring segmentation.")
    parser.add_argument("--config", default="configs/tref_sam3_config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output_json", default="outputs/tref_sam3/tref_metrics.json")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=0.35)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ref_cfg = cfg["training"].get("referring", {})
    data_dir = args.data_dir or ref_cfg.get("val_data_dir", cfg["training"]["data_dir"])
    split = args.split or ref_cfg.get("val_split", "valid")

    evaluate(
        config_path=args.config,
        weights_path=args.weights,
        data_dir=data_dir,
        split=split,
        output_json=args.output_json,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
