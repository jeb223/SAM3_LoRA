#!/usr/bin/env python3
"""Convert RefCOCO-style REFER annotations to the TRef-SAM3 format.

Inputs usually look like:
  - instances.json: COCO instance annotations
  - refs(unc).p / refs(umd).p: a pickle list produced by the REFER API

Output:
  output_root/
    annotations.json
    referring_annotations.json
    stats.json
"""

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path


def norm_path(path):
    return str(Path(path).resolve()).replace("\\", "/")


def clean_expr(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def image_path_for_coco_image(image_root, file_name, image_prefix=""):
    file_name = Path(str(file_name))
    if file_name.is_absolute():
        return norm_path(file_name)
    root = Path(image_root)
    if image_prefix:
        return norm_path(root / image_prefix / file_name)
    return norm_path(root / file_name)


def convert_refcoco(
    instances_json,
    refs_pickle,
    image_root,
    output_root,
    image_prefix="",
    split="all",
    source_name="refcoco",
):
    coco = load_json(instances_json)
    with open(refs_pickle, "rb") as f:
        raw_refs = pickle.load(f)

    ann_by_id = {int(ann["id"]): ann for ann in coco.get("annotations", [])}
    selected_refs = []
    selected_image_ids = set()
    selected_ann_ids = set()

    for ref in raw_refs:
        ref_split = str(ref.get("split", "train"))
        if split != "all" and ref_split != split:
            continue
        ann_id = int(ref["ann_id"])
        if ann_id not in ann_by_id:
            continue
        image_id = int(ref["image_id"])
        ann = ann_by_id[ann_id]
        selected_image_ids.add(image_id)
        selected_ann_ids.add(ann_id)
        category_id = int(ann.get("category_id", -1))

        for sent in ref.get("sentences", []):
            expression = clean_expr(sent.get("raw") or sent.get("sent") or "")
            if not expression:
                continue
            selected_refs.append(
                {
                    "image_id": image_id,
                    "expression": expression,
                    "target_ann_ids": [ann_id],
                    "category_ids": [category_id],
                    "is_unique": True,
                    "num_targets": 1,
                    "split": ref_split,
                    "source": source_name,
                    "phrase_type": "refcoco_expression",
                    "attributes": {},
                }
            )

    selected_images = []
    for image in coco.get("images", []):
        if int(image["id"]) not in selected_image_ids:
            continue
        item = dict(image)
        item["file_name"] = image_path_for_coco_image(
            image_root=image_root,
            file_name=item["file_name"],
            image_prefix=image_prefix,
        )
        selected_images.append(item)

    # Keep all instance annotations from selected images so non-target objects remain
    # available as hard negatives during text-candidate matching.
    selected_annotations = [
        ann for ann in coco.get("annotations", [])
        if int(ann.get("image_id", -1)) in selected_image_ids
    ]

    out_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": selected_images,
        "annotations": selected_annotations,
    }

    refs = []
    seen = set()
    for item in selected_refs:
        key = (item["image_id"], item["expression"], tuple(item["target_ann_ids"]), item["split"])
        if key in seen:
            continue
        seen.add(key)
        item = dict(item)
        item["id"] = len(refs) + 1
        refs.append(item)

    output_root = Path(output_root)
    write_json(output_root / "annotations.json", out_coco)
    write_json(output_root / "referring_annotations.json", refs)
    split_counts = Counter(ref["split"] for ref in refs)
    stats = {
        "num_images": len(selected_images),
        "num_annotations": len(selected_annotations),
        "num_referenced_annotations": len(selected_ann_ids),
        "num_referring_annotations": len(refs),
        "split_counts": dict(split_counts),
        "examples": refs[:20],
    }
    write_json(output_root / "stats.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert RefCOCO REFER annotations to TRef format.")
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--refs-pickle", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--image-prefix", default="")
    parser.add_argument("--split", default="all")
    parser.add_argument("--source-name", default="refcoco")
    args = parser.parse_args()

    stats = convert_refcoco(
        instances_json=args.instances_json,
        refs_pickle=args.refs_pickle,
        image_root=args.image_root,
        output_root=args.output_root,
        image_prefix=args.image_prefix,
        split=args.split,
        source_name=args.source_name,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
