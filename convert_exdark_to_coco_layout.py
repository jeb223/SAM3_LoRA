#!/usr/bin/env python3
"""
Convert the ExDark dataset into the flat COCO split layout expected by this
repository.

Output layout:
  output_root/
    train/
      _annotations.coco.json
      <image files>
    valid/
      _annotations.coco.json
      <image files>

Notes:
- ExDark annotations are bounding-box only. The generated COCO files do not
  contain segmentation masks.
- No official train/valid split file was found in the local dataset copy, so
  this script creates a deterministic split grouped by the source class folder.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image


DEFAULT_CATEGORY_ORDER = [
    "Bicycle",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cup",
    "Dog",
    "Motorbike",
    "People",
    "Table",
]


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


def parse_annotation_file(txt_path: Path):
    records = []
    with txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Invalid annotation line in {txt_path}: {line}")

            label = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            records.append(
                {
                    "label": label,
                    "bbox": [x, y, w, h],
                }
            )
    return records


def collect_entries(image_root: Path, anno_root: Path):
    entries = []
    labels = set()
    collisions = {}

    for class_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        anno_class_dir = anno_root / class_dir.name
        if not anno_class_dir.exists():
            raise FileNotFoundError(f"Missing annotation directory: {anno_class_dir}")

        for image_path in sorted(p for p in class_dir.iterdir() if p.is_file()):
            anno_path = anno_class_dir / f"{image_path.name}.txt"
            if not anno_path.exists():
                raise FileNotFoundError(f"Missing annotation file for {image_path}: {anno_path}")

            key = image_path.name
            collisions[key] = collisions.get(key, 0) + 1

            annotations = parse_annotation_file(anno_path)
            for ann in annotations:
                labels.add(ann["label"])

            entries.append(
                {
                    "source_group": class_dir.name,
                    "image_path": image_path,
                    "anno_path": anno_path,
                    "image_name": image_path.name,
                    "annotations": annotations,
                }
            )

    duplicate_names = [name for name, count in collisions.items() if count > 1]
    if duplicate_names:
        sample = ", ".join(sorted(duplicate_names)[:10])
        raise ValueError(
            "Found duplicate image basenames that would collide in the flat output "
            f"layout. Sample duplicates: {sample}"
        )

    return entries, labels


def build_category_to_id(labels: set[str]) -> dict[str, int]:
    ordered = [label for label in DEFAULT_CATEGORY_ORDER if label in labels]
    extras = sorted(label for label in labels if label not in DEFAULT_CATEGORY_ORDER)
    ordered.extend(extras)
    return {label: idx + 1 for idx, label in enumerate(ordered)}


def split_entries(entries, valid_ratio: float, seed: int):
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["source_group"], []).append(entry)

    rng = random.Random(seed)
    train_entries = []
    valid_entries = []

    for group_name in sorted(grouped):
        group_entries = list(grouped[group_name])
        rng.shuffle(group_entries)

        if valid_ratio <= 0:
            valid_count = 0
        else:
            valid_count = int(round(len(group_entries) * valid_ratio))
            if len(group_entries) > 1:
                valid_count = max(1, min(valid_count, len(group_entries) - 1))
            else:
                valid_count = min(valid_count, 1)

        valid_entries.extend(group_entries[:valid_count])
        train_entries.extend(group_entries[valid_count:])

    train_entries.sort(key=lambda item: item["image_name"])
    valid_entries.sort(key=lambda item: item["image_name"])
    return train_entries, valid_entries


def build_split(entries, output_dir: Path, category_to_id: dict[str, int], file_mode: str):
    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    for entry in entries:
        image_path = entry["image_path"]
        dst_path = output_dir / entry["image_name"]
        link_or_copy_file(image_path, dst_path, file_mode)

        with Image.open(image_path) as img:
            width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": entry["image_name"],
                "width": int(width),
                "height": int(height),
            }
        )

        for ann in entry["annotations"]:
            label = ann["label"]
            x, y, w, h = ann["bbox"]

            x = max(0.0, min(x, float(width)))
            y = max(0.0, min(y, float(height)))
            w = max(0.0, min(w, float(width) - x))
            h = max(0.0, min(h, float(height) - y))
            area = w * h

            if w <= 0 or h <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_to_id[label],
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    coco = {
        "info": {
            "description": "ExDark converted to flat COCO layout for SAM3 LoRA",
            "version": "1.0",
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cat_id, "name": label, "supercategory": "exdark"}
            for label, cat_id in sorted(category_to_id.items(), key=lambda item: item[1])
        ],
    }

    with (output_dir / "_annotations.coco.json").open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return len(images), len(annotations)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ExDark to flat COCO train/valid layout."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="ExDark root containing ExDark/ and ExDark_Annno/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root to create train/ and valid/ under.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio created from the local dataset copy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
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
    args = parser.parse_args()

    if not (0.0 <= args.valid_ratio < 1.0):
        raise ValueError("--valid-ratio must be in [0, 1).")

    source_root = Path(args.source_root)
    image_root = source_root / "ExDark"
    anno_root = source_root / "ExDark_Annno"
    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    valid_dir = output_root / "valid"

    if not image_root.exists():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    if not anno_root.exists():
        raise FileNotFoundError(f"Missing annotation root: {anno_root}")

    ensure_clean_dir(train_dir, args.overwrite)
    ensure_clean_dir(valid_dir, args.overwrite)

    entries, labels = collect_entries(image_root, anno_root)
    category_to_id = build_category_to_id(labels)
    train_entries, valid_entries = split_entries(entries, args.valid_ratio, args.seed)

    train_images, train_annotations = build_split(
        train_entries, train_dir, category_to_id, args.file_mode
    )
    valid_images, valid_annotations = build_split(
        valid_entries, valid_dir, category_to_id, args.file_mode
    )

    print("ExDark conversion complete.")
    print(f"Categories ({len(category_to_id)}): {list(category_to_id.keys())}")
    print(
        f"Train: {train_images} images, {train_annotations} annotations -> {train_dir}"
    )
    print(
        f"Valid: {valid_images} images, {valid_annotations} annotations -> {valid_dir}"
    )
    print("Note: generated COCO annotations contain bbox only; no segmentation masks.")


if __name__ == "__main__":
    main()
