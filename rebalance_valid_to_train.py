#!/usr/bin/env python3
"""
Move a ratio of validation images into training split for COCO-format data.

Expected layout:
  data/
    train/
      _annotations.coco.json
      *.jpg|*.png|...
    valid/
      _annotations.coco.json
      *.jpg|*.png|...
"""

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak_{ts}")
    shutil.copy2(path, backup)
    return backup


def pick_images(images: List[Dict], ratio: float, seed: int) -> List[Dict]:
    count = int(len(images) * ratio)
    rng = random.Random(seed)
    items = images[:]
    rng.shuffle(items)
    return items[:count]


def remap_image_ids(
    moved_images: List[Dict], target_images: List[Dict]
) -> Tuple[List[Dict], Dict[int, int]]:
    used_ids = {int(img["id"]) for img in target_images}
    next_id = (max(used_ids) + 1) if used_ids else 0
    id_map: Dict[int, int] = {}
    out: List[Dict] = []

    for img in moved_images:
        old_id = int(img["id"])
        new_id = old_id
        if new_id in used_ids:
            new_id = next_id
            next_id += 1
        used_ids.add(new_id)

        new_img = dict(img)
        new_img["id"] = new_id
        out.append(new_img)
        id_map[old_id] = new_id

    return out, id_map


def remap_annotations(
    moved_annotations: List[Dict], image_id_map: Dict[int, int], target_annotations: List[Dict]
) -> List[Dict]:
    used_ids = {int(ann["id"]) for ann in target_annotations if "id" in ann}
    next_id = (max(used_ids) + 1) if used_ids else 0
    out: List[Dict] = []

    for ann in moved_annotations:
        new_ann = dict(ann)
        new_ann["image_id"] = image_id_map[int(ann["image_id"])]
        new_ann["id"] = next_id
        next_id += 1
        out.append(new_ann)

    return out


def move_image_files(
    data_root: Path,
    source_split: str,
    target_split: str,
    images: List[Dict],
) -> None:
    for img in images:
        file_name = img["file_name"]
        src = data_root / source_split / file_name
        dst = data_root / target_split / file_name

        if not src.exists():
            raise FileNotFoundError(f"Source image missing: {src}")
        if dst.exists():
            raise FileExistsError(f"Target image already exists: {dst}")

        shutil.move(str(src), str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move a fraction of valid split into train split for COCO data."
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--source-split", type=str, default="valid")
    parser.add_argument("--target-split", type=str, default="train")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0.0 < args.ratio < 1.0):
        raise ValueError("--ratio must be between 0 and 1")

    data_root = Path(args.data_root)
    source_dir = data_root / args.source_split
    target_dir = data_root / args.target_split
    source_json = source_dir / "_annotations.coco.json"
    target_json = target_dir / "_annotations.coco.json"

    if not source_json.exists():
        raise FileNotFoundError(f"Missing source annotations: {source_json}")
    if not target_json.exists():
        raise FileNotFoundError(f"Missing target annotations: {target_json}")

    source_coco = load_coco(source_json)
    target_coco = load_coco(target_json)

    source_images = list(source_coco.get("images", []))
    source_annotations = list(source_coco.get("annotations", []))
    target_images = list(target_coco.get("images", []))
    target_annotations = list(target_coco.get("annotations", []))

    moved_images = pick_images(source_images, args.ratio, args.seed)
    moved_image_ids = {int(img["id"]) for img in moved_images}
    moved_annotations = [
        ann for ann in source_annotations if int(ann["image_id"]) in moved_image_ids
    ]

    kept_images = [img for img in source_images if int(img["id"]) not in moved_image_ids]
    kept_annotations = [
        ann for ann in source_annotations if int(ann["image_id"]) not in moved_image_ids
    ]

    remapped_images, image_id_map = remap_image_ids(moved_images, target_images)
    remapped_annotations = remap_annotations(
        moved_annotations, image_id_map, target_annotations
    )

    print("=== Rebalance Summary ===")
    print(f"Source split ({args.source_split}): {len(source_images)} images, {len(source_annotations)} anns")
    print(f"Target split ({args.target_split}): {len(target_images)} images, {len(target_annotations)} anns")
    print(f"Move ratio: {args.ratio}")
    print(f"Selected to move: {len(moved_images)} images, {len(moved_annotations)} anns")
    print(f"After move - {args.source_split}: {len(kept_images)} images, {len(kept_annotations)} anns")
    print(
        f"After move - {args.target_split}: "
        f"{len(target_images) + len(remapped_images)} images, "
        f"{len(target_annotations) + len(remapped_annotations)} anns"
    )

    if args.dry_run:
        print("Dry-run mode: no files were changed.")
        return

    backup_source = backup_file(source_json)
    backup_target = backup_file(target_json)
    print(f"Backups created:\n  {backup_source}\n  {backup_target}")

    move_image_files(data_root, args.source_split, args.target_split, moved_images)

    source_coco["images"] = kept_images
    source_coco["annotations"] = kept_annotations
    target_coco["images"] = target_images + remapped_images
    target_coco["annotations"] = target_annotations + remapped_annotations

    save_coco(source_json, source_coco)
    save_coco(target_json, target_coco)

    print("Rebalance complete.")


if __name__ == "__main__":
    main()

