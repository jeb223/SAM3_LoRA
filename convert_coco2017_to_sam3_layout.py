#!/usr/bin/env python3
"""
Convert a COCO2017-style dataset layout into the train/valid layout expected by
the SAM3 LoRA scripts in this repository.

Input layout:
  dataset_root/
    annotations/
      instances_train2017.json
      instances_val2017.json
    train2017/
      *.jpg
    val2017/
      *.jpg

Output layout:
  output_root/
    train/
      _annotations.coco.json
      *.jpg
    valid/
      _annotations.coco.json
      *.jpg

Images are hardlinked by default to avoid data duplication when source and
destination are on the same filesystem. Fallback to copying is supported.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
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


def iter_split_images(coco: Dict) -> Iterable[Dict]:
    return coco.get("images", [])


def link_or_copy_file(src: Path, dst: Path, mode: str) -> None:
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


def convert_split(
    src_image_dir: Path,
    src_ann_path: Path,
    dst_split_dir: Path,
    file_mode: str,
) -> Tuple[int, int]:
    coco = load_json(src_ann_path)
    images = list(iter_split_images(coco))
    annotations = list(coco.get("annotations", []))

    dst_split_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        file_name = img["file_name"]
        src_image = src_image_dir / file_name
        dst_image = dst_split_dir / file_name
        if not src_image.exists():
            raise FileNotFoundError(f"Missing source image referenced in COCO json: {src_image}")
        link_or_copy_file(src_image, dst_image, file_mode)

    save_json(dst_split_dir / "_annotations.coco.json", coco)
    return len(images), len(annotations)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO2017 layout to SAM3 train/valid layout."
    )
    parser.add_argument(
        "--src-root",
        type=str,
        required=True,
        help="Source dataset root with annotations/, train2017/, val2017/.",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Output dataset root to create.",
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

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    src_train_dir = src_root / "train2017"
    src_val_dir = src_root / "val2017"
    src_train_ann = src_root / "annotations" / "instances_train2017.json"
    src_val_ann = src_root / "annotations" / "instances_val2017.json"

    for required in [src_train_dir, src_val_dir, src_train_ann, src_val_ann]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required source path: {required}")

    ensure_clean_dir(dst_root, overwrite=args.overwrite)
    dst_train_dir = dst_root / "train"
    dst_valid_dir = dst_root / "valid"

    train_images, train_annotations = convert_split(
        src_image_dir=src_train_dir,
        src_ann_path=src_train_ann,
        dst_split_dir=dst_train_dir,
        file_mode=args.file_mode,
    )
    valid_images, valid_annotations = convert_split(
        src_image_dir=src_val_dir,
        src_ann_path=src_val_ann,
        dst_split_dir=dst_valid_dir,
        file_mode=args.file_mode,
    )

    print("Conversion complete.")
    print(f"Source:      {src_root}")
    print(f"Destination: {dst_root}")
    print(f"File mode:   {args.file_mode}")
    print(f"Train:       {train_images} images, {train_annotations} annotations")
    print(f"Valid:       {valid_images} images, {valid_annotations} annotations")
    print(f"Train json:  {dst_train_dir / '_annotations.coco.json'}")
    print(f"Valid json:  {dst_valid_dir / '_annotations.coco.json'}")


if __name__ == "__main__":
    main()
