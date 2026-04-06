#!/usr/bin/env python3
"""
Convert a Cityscapes-style dataset into the flat COCO split layout expected by
the SAM3 LoRA scripts in this repository.

Output layout:
  output_root/
    train/
      _annotations.coco.json
      <image files>
    valid/
      _annotations.coco.json
      <image files>

By default this converter skips Cityscapes ignore-like labels and any labels
ending with "group", producing a more usable instance-style COCO dataset for
training with this repo.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path


DEFAULT_IGNORE_LABELS = {
    "unlabeled",
    "ego vehicle",
    "rectification border",
    "out of roi",
    "static",
    "dynamic",
    "ground",
    "parking",
    "rail track",
    "guard rail",
    "bridge",
    "tunnel",
    "caravan",
    "trailer",
    "license plate",
}


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported file mode: {mode}")


def polygon_area(flat_polygon):
    xs = flat_polygon[0::2]
    ys = flat_polygon[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5


def flatten_polygon(points):
    flat = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None
        flat.extend([float(pt[0]), float(pt[1])])
    return flat


def should_skip_label(label: str, ignore_labels: set[str]) -> bool:
    if not label:
        return True
    if label in ignore_labels:
        return True
    if label.endswith("group"):
        return True
    return False


def find_image_path(source_root: Path, split: str, city: str, image_name: str) -> Path:
    candidates = [
        source_root / split / city / image_name,
        source_root / "leftImg8bit" / split / city / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find image {image_name} for split={split}, city={city}. "
        f"Tried: {candidates}"
    )


def collect_labels(gt_root: Path, splits, ignore_labels: set[str]) -> list[str]:
    labels = set()
    for split in splits:
        split_dir = gt_root / split
        for json_path in split_dir.rglob("*_gtFine_polygons.json"):
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for obj in data.get("objects", []):
                label = obj.get("label", "")
                if should_skip_label(label, ignore_labels):
                    continue
                polygon = flatten_polygon(obj.get("polygon", []))
                if polygon is None or len(polygon) < 6:
                    continue
                if polygon_area(polygon) <= 0:
                    continue
                labels.add(label)
    return sorted(labels)


def build_split(
    source_root: Path,
    gt_root: Path,
    split: str,
    output_split_dir: Path,
    category_to_id: dict[str, int],
    file_mode: str,
):
    images = []
    annotations = []
    image_id = 1
    ann_id = 1
    num_skipped_labels = 0

    for json_path in sorted((gt_root / split).rglob("*_gtFine_polygons.json")):
        city = json_path.parent.name
        image_name = json_path.name.replace("_gtFine_polygons.json", "_leftImg8bit.png")
        image_src = find_image_path(source_root, split, city, image_name)
        image_dst = output_split_dir / image_name
        link_or_copy_file(image_src, image_dst, file_mode)

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images.append(
            {
                "id": image_id,
                "file_name": image_name,
                "width": int(data["imgWidth"]),
                "height": int(data["imgHeight"]),
            }
        )

        for obj in data.get("objects", []):
            label = obj.get("label", "")
            if label not in category_to_id:
                num_skipped_labels += 1
                continue

            polygon = flatten_polygon(obj.get("polygon", []))
            if polygon is None or len(polygon) < 6:
                continue

            xs = polygon[0::2]
            ys = polygon[1::2]
            x_min = max(0.0, min(xs))
            y_min = max(0.0, min(ys))
            x_max = min(float(data["imgWidth"]), max(xs))
            y_max = min(float(data["imgHeight"]), max(ys))
            width = max(0.0, x_max - x_min)
            height = max(0.0, y_max - y_min)
            area = polygon_area(polygon)

            if width <= 0 or height <= 0 or area <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_to_id[label],
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "segmentation": [polygon],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    coco = {
        "info": {
            "description": "Cityscapes converted to flat COCO layout for SAM3 LoRA",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cat_id, "name": label, "supercategory": "cityscapes"}
            for label, cat_id in sorted(category_to_id.items(), key=lambda x: x[1])
        ],
    }

    with (output_split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return len(images), len(annotations), num_skipped_labels


def main():
    parser = argparse.ArgumentParser(
        description="Convert Cityscapes polygons to flat COCO train/valid layout."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Cityscapes root containing gtFine/ and image folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root to create train/ and valid/ under.",
    )
    parser.add_argument(
        "--file-mode",
        type=str,
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to materialize image files in the output layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination if it already exists.",
    )
    parser.add_argument(
        "--keep-all-labels",
        action="store_true",
        help="Keep all labels, including ignore-like classes and *group labels.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    gt_root = source_root / "gtFine"
    if not gt_root.exists():
        raise FileNotFoundError(f"Missing Cityscapes gtFine directory: {gt_root}")

    ignore_labels = set() if args.keep_all_labels else set(DEFAULT_IGNORE_LABELS)
    labels = collect_labels(gt_root, ["train", "val"], ignore_labels)
    if not labels:
        raise ValueError("No usable labels found in Cityscapes polygons.")
    category_to_id = {label: idx + 1 for idx, label in enumerate(labels)}

    output_root = Path(args.output_root)
    ensure_clean_dir(output_root, overwrite=args.overwrite)
    train_dir = output_root / "train"
    valid_dir = output_root / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_annotations, train_skipped = build_split(
        source_root=source_root,
        gt_root=gt_root,
        split="train",
        output_split_dir=train_dir,
        category_to_id=category_to_id,
        file_mode=args.file_mode,
    )
    valid_images, valid_annotations, valid_skipped = build_split(
        source_root=source_root,
        gt_root=gt_root,
        split="val",
        output_split_dir=valid_dir,
        category_to_id=category_to_id,
        file_mode=args.file_mode,
    )

    print("Conversion complete.")
    print(f"Source:        {source_root}")
    print(f"Destination:   {output_root}")
    print(f"File mode:     {args.file_mode}")
    print(f"Categories:    {len(category_to_id)}")
    print(f"Train:         {train_images} images, {train_annotations} annotations")
    print(f"Valid:         {valid_images} images, {valid_annotations} annotations")
    print(f"Skipped train labels not in category map: {train_skipped}")
    print(f"Skipped valid labels not in category map: {valid_skipped}")
    print(f"Train json:    {train_dir / '_annotations.coco.json'}")
    print(f"Valid json:    {valid_dir / '_annotations.coco.json'}")


if __name__ == "__main__":
    main()
