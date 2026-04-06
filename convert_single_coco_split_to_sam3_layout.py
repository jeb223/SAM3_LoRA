#!/usr/bin/env python3
"""
Convert a single COCO split directory into the train/valid layout expected by
the SAM3 LoRA training scripts in this repository.

Input layout:
  source_dir/
    instances_val2017.json
    000000000001.jpg
    000000000002.jpg
    ...

Output layout:
  output_root/
    train/
      _annotations.coco.json
      000000000001.jpg
      ...
    valid/                    # Optional when --valid-ratio > 0
      _annotations.coco.json
      000000000123.jpg
      ...

Images are hardlinked by default to avoid data duplication when source and
destination are on the same filesystem. Fallback to copying is supported.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ANN_CANDIDATES = (
    "_annotations.coco.json",
    "instances_train2017.json",
    "instances_val2017.json",
    "annotations.json",
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def auto_detect_annotation_file(source_dir: Path) -> Path:
    for candidate in DEFAULT_ANN_CANDIDATES:
        ann_path = source_dir / candidate
        if ann_path.exists():
            return ann_path

    json_candidates = sorted(source_dir.glob("*.json"))
    if len(json_candidates) == 1:
        return json_candidates[0]

    candidate_str = ", ".join(DEFAULT_ANN_CANDIDATES)
    raise FileNotFoundError(
        f"Could not auto-detect a COCO annotation file in {source_dir}. "
        f"Tried: {candidate_str}"
    )


def subset_coco(coco: Dict, image_ids: Sequence[int]) -> Dict:
    image_id_set = set(int(x) for x in image_ids)

    subset = {
        "images": [img for img in coco.get("images", []) if int(img["id"]) in image_id_set],
        "annotations": [
            ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in image_id_set
        ],
        "categories": list(coco.get("categories", [])),
    }

    for key in ("info", "licenses"):
        if key in coco:
            subset[key] = coco[key]

    return subset


def validate_source_images(coco: Dict, source_dir: Path) -> List[Tuple[int, str]]:
    missing: List[Tuple[int, str]] = []
    for img in coco.get("images", []):
        file_name = img.get("file_name")
        if not file_name:
            missing.append((int(img.get("id", -1)), "<missing file_name>"))
            continue
        src_path = source_dir / file_name
        if not src_path.exists():
            missing.append((int(img.get("id", -1)), str(file_name)))
    return missing


def split_image_ids(images: Iterable[Dict], valid_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    image_ids = [int(img["id"]) for img in images]
    if not image_ids:
        return [], []

    if valid_ratio <= 0:
        return image_ids, []

    if len(image_ids) < 2:
        raise ValueError("Need at least 2 images to create both train and valid splits.")

    num_valid = int(round(len(image_ids) * valid_ratio))
    num_valid = max(1, min(num_valid, len(image_ids) - 1))

    rng = random.Random(seed)
    shuffled_ids = image_ids[:]
    rng.shuffle(shuffled_ids)

    valid_ids = sorted(shuffled_ids[:num_valid])
    train_ids = sorted(shuffled_ids[num_valid:])
    return train_ids, valid_ids


def materialize_split(
    source_dir: Path,
    coco: Dict,
    image_ids: Sequence[int],
    dst_split_dir: Path,
    file_mode: str,
) -> Tuple[int, int]:
    subset = subset_coco(coco, image_ids)
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    for img in subset["images"]:
        src_image = source_dir / img["file_name"]
        dst_image = dst_split_dir / img["file_name"]
        link_or_copy_file(src_image, dst_image, file_mode)

    save_json(dst_split_dir / "_annotations.coco.json", subset)
    return len(subset["images"]), len(subset["annotations"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a single COCO split directory to SAM3 train/valid layout."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Source directory containing one COCO json plus the matching images.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output dataset root to create.",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default=None,
        help="Optional explicit annotation file path. Defaults to auto-detect inside source-dir.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of images to place into valid/. Use 0 to create train only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/valid split.",
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

    if not 0 <= args.valid_ratio < 1:
        raise ValueError("--valid-ratio must be in [0, 1).")

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    ann_path = Path(args.ann_file) if args.ann_file else auto_detect_annotation_file(source_dir)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file does not exist: {ann_path}")

    coco = load_json(ann_path)
    images = list(coco.get("images", []))
    if not images:
        raise ValueError(f"No images found in COCO annotation file: {ann_path}")

    missing_images = validate_source_images(coco, source_dir)
    if missing_images:
        preview = ", ".join(f"{img_id}:{name}" for img_id, name in missing_images[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_images)} image files referenced by {ann_path}. "
            f"First missing entries: {preview}"
        )

    train_ids, valid_ids = split_image_ids(images, args.valid_ratio, args.seed)

    output_root = Path(args.output_root)
    ensure_clean_dir(output_root, overwrite=args.overwrite)

    train_images, train_annotations = materialize_split(
        source_dir=source_dir,
        coco=coco,
        image_ids=train_ids,
        dst_split_dir=output_root / "train",
        file_mode=args.file_mode,
    )

    if valid_ids:
        valid_images, valid_annotations = materialize_split(
            source_dir=source_dir,
            coco=coco,
            image_ids=valid_ids,
            dst_split_dir=output_root / "valid",
            file_mode=args.file_mode,
        )
    else:
        valid_images, valid_annotations = 0, 0

    print("Conversion complete.")
    print(f"Source dir:   {source_dir}")
    print(f"Annotation:   {ann_path}")
    print(f"Output root:  {output_root}")
    print(f"File mode:    {args.file_mode}")
    print(f"Valid ratio:  {args.valid_ratio}")
    print(f"Train:        {train_images} images, {train_annotations} annotations")
    if valid_ids:
        print(f"Valid:        {valid_images} images, {valid_annotations} annotations")
        print(f"Valid json:   {output_root / 'valid' / '_annotations.coco.json'}")
    else:
        print("Valid:        skipped")
    print(f"Train json:   {output_root / 'train' / '_annotations.coco.json'}")


if __name__ == "__main__":
    main()
