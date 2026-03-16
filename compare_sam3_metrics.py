#!/usr/bin/env python3
"""
Compare base SAM3 vs SAM3+LoRA on the same validation split.
"""

import argparse
import json
from datetime import datetime

from validate_sam3_lora import validate


METRIC_SPECS = [
    ("mAP", "mAP"),
    ("mAP50", "mAP@50"),
    ("mAP75", "mAP@75"),
    ("cgF1", "cgF1"),
    ("cgF1_50", "cgF1@50"),
    ("cgF1_75", "cgF1@75"),
    ("semantic_mIoU", "Semantic mIoU"),
    ("semantic_macro_F1", "Semantic pixel F1 (macro)"),
    ("semantic_micro_F1", "Semantic pixel F1 (micro)"),
    ("semantic_pixel_accuracy", "Pixel accuracy"),
]


def _fmt(value):
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _delta(lora_value, base_value):
    if lora_value is None or base_value is None:
        return "n/a"
    diff = lora_value - base_value
    return f"{diff:+.4f}"


def _print_summary(base_metrics, lora_metrics):
    print("\n" + "=" * 96)
    print("BASE SAM3 VS SAM3 + LORA")
    print("=" * 96)
    header = f"{'Metric':<30}{'Base SAM3':>14}{'SAM3+LoRA':>14}{'Delta':>14}"
    print(header)
    print("-" * len(header))
    for key, label in METRIC_SPECS:
        print(
            f"{label:<30}"
            f"{_fmt(base_metrics.get(key)):>14}"
            f"{_fmt(lora_metrics.get(key)):>14}"
            f"{_delta(lora_metrics.get(key), base_metrics.get(key)):>14}"
        )
    print("-" * len(header))
    print(
        f"{'Predictions':<30}"
        f"{base_metrics.get('num_predictions', 0):>14}"
        f"{lora_metrics.get('num_predictions', 0):>14}"
        f"{lora_metrics.get('num_predictions', 0) - base_metrics.get('num_predictions', 0):>14}"
    )
    print(
        f"{'Query items':<30}"
        f"{base_metrics.get('num_query_items', 0):>14}"
        f"{lora_metrics.get('num_query_items', 0):>14}"
        f"{lora_metrics.get('num_query_items', 0) - base_metrics.get('num_query_items', 0):>14}"
    )
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base SAM3 and SAM3+LoRA metrics on the same validation split."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to LoRA config file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to LoRA weights file.",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Validation directory containing _annotations.coco.json.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit validation to N samples.",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for prediction filtering.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable crack-style overlap merging.",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.15,
        help="IoU threshold used for merging overlaps.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the comparison as JSON.",
    )
    args = parser.parse_args()

    print("\nRunning base SAM3 validation...")
    base_metrics = validate(
        config_path=None,
        weights_path=None,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        use_base_model=True,
    )

    print("\nRunning SAM3+LoRA validation...")
    lora_metrics = validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        use_base_model=False,
    )

    _print_summary(base_metrics, lora_metrics)

    if args.output_json:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "base_model": base_metrics,
            "lora_model": lora_metrics,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved comparison JSON to {args.output_json}")


if __name__ == "__main__":
    main()
