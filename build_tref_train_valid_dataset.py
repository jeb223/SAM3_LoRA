import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from build_text_refseg_dataset import (
    bbox_gap,
    build_generated_refs,
    make_abs_coco,
    norm_path,
    polygon_bbox,
    write_dataset,
    write_json,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    return inter / union if union > 0 else 0.0


def build_label_index(museg_root):
    museg_root = Path(museg_root)
    index = {}
    for mine_id in range(1, 7):
        label_dir = museg_root / f"{mine_id:02d}-Mine" / "Label"
        if not label_dir.exists():
            continue
        for path in label_dir.glob("*_polygons.json"):
            stem = path.name[: -len("_polygons.json")]
            index[stem] = path
    return index


def labels_from_polygon_file(path, cat_id_by_name):
    data = load_json(path)
    labels = []
    unknown = Counter()
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        if not label or label == "__background__":
            continue
        if label not in cat_id_by_name:
            unknown[label] += 1
            continue
        pts = shape.get("points") or []
        if len(pts) < 3:
            continue
        flat = []
        for p in pts:
            if len(p) >= 2:
                flat.extend([float(p[0]), float(p[1])])
        if len(flat) < 6:
            continue
        labels.append({"category_id": cat_id_by_name[label], "bbox": polygon_bbox(flat)})
    return labels, unknown


def verify_split(coco, split_root, museg_label_index):
    split_root = Path(split_root)
    images = {int(img["id"]): img for img in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    cat_id_by_name = {c["name"].strip().lower(): int(c["id"]) for c in coco.get("categories", [])}
    report = {
        "num_images": len(images),
        "num_annotations": len(coco.get("annotations", [])),
        "missing_image_files": [],
        "dimension_mismatches": [],
        "missing_source_polygon_files": [],
        "category_count_mismatches": [],
        "low_bbox_iou_matches": [],
        "bbox_out_of_bounds": [],
        "segmentation_out_of_bounds": [],
        "unknown_source_labels": {},
    }

    unknown_total = Counter()
    for image_id, img in images.items():
        image_path = split_root / img["file_name"]
        if not image_path.exists():
            report["missing_image_files"].append(img["file_name"])
            continue
        try:
            with Image.open(image_path) as im:
                actual_w, actual_h = im.size
        except Exception:
            report["missing_image_files"].append(img["file_name"])
            continue
        width, height = int(img["width"]), int(img["height"])
        if (actual_w, actual_h) != (width, height):
            report["dimension_mismatches"].append(
                {"file_name": img["file_name"], "json": [width, height], "actual": [actual_w, actual_h]}
            )

        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            if x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1 or w <= 0 or h <= 0:
                report["bbox_out_of_bounds"].append({"image_id": image_id, "ann_id": ann["id"], "bbox": ann["bbox"]})
            for poly in ann.get("segmentation", []) or []:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                xs, ys = poly[0::2], poly[1::2]
                if min(xs) < -1 or min(ys) < -1 or max(xs) > width + 1 or max(ys) > height + 1:
                    report["segmentation_out_of_bounds"].append(
                        {
                            "image_id": image_id,
                            "ann_id": ann["id"],
                            "x_range": [min(xs), max(xs)],
                            "y_range": [min(ys), max(ys)],
                        }
                    )

        stem = Path(img["file_name"]).stem
        label_path = museg_label_index.get(stem)
        if label_path is None:
            report["missing_source_polygon_files"].append(img["file_name"])
            continue
        source_labels, unknown = labels_from_polygon_file(label_path, cat_id_by_name)
        unknown_total.update(unknown)

        coco_counts = Counter(int(a["category_id"]) for a in anns_by_image.get(image_id, []))
        source_counts = Counter(int(x["category_id"]) for x in source_labels)
        if coco_counts != source_counts:
            report["category_count_mismatches"].append(
                {
                    "file_name": img["file_name"],
                    "coco_counts": dict(coco_counts),
                    "source_counts": dict(source_counts),
                }
            )
            continue

        for cat_id in sorted(source_counts):
            coco_boxes = [a["bbox"] for a in anns_by_image.get(image_id, []) if int(a["category_id"]) == cat_id]
            source_boxes = [x["bbox"] for x in source_labels if int(x["category_id"]) == cat_id]
            used = set()
            for sb in source_boxes:
                best_iou, best_j = -1.0, None
                for j, cb in enumerate(coco_boxes):
                    if j in used:
                        continue
                    iou = bbox_iou(sb, cb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j is not None:
                    used.add(best_j)
                if best_iou < 0.995:
                    report["low_bbox_iou_matches"].append(
                        {
                            "file_name": img["file_name"],
                            "category_id": cat_id,
                            "source_bbox": sb,
                            "best_iou": round(float(best_iou), 6),
                        }
                    )

    report["unknown_source_labels"] = dict(unknown_total)
    for key in [
        "missing_image_files",
        "dimension_mismatches",
        "missing_source_polygon_files",
        "category_count_mismatches",
        "low_bbox_iou_matches",
        "bbox_out_of_bounds",
        "segmentation_out_of_bounds",
    ]:
        report[f"num_{key}"] = len(report[key])
        report[key] = report[key][:50]
    return report


def remap_and_combine(train_coco, valid_coco):
    combined = {
        "info": {"description": "TRef MineRefSeg train/valid split dataset"},
        "licenses": [],
        "categories": train_coco["categories"],
        "images": [],
        "annotations": [],
    }
    split_by_image = {}
    next_image_id = 1
    next_ann_id = 1
    for split, coco in [("train", train_coco), ("valid", valid_coco)]:
        image_map = {}
        ann_map = {}
        for img in coco["images"]:
            old_id = int(img["id"])
            new_img = dict(img)
            new_img["id"] = next_image_id
            new_img["split"] = split
            image_map[old_id] = next_image_id
            split_by_image[next_image_id] = split
            combined["images"].append(new_img)
            next_image_id += 1
        for ann in coco["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_map[int(ann["image_id"])]
            new_ann["split"] = split
            ann_map[int(ann["id"])] = next_ann_id
            combined["annotations"].append(new_ann)
            next_ann_id += 1
    return combined, split_by_image


def validate_ref_targets(coco, refs):
    ann_ids = {int(a["id"]) for a in coco["annotations"]}
    image_ids = {int(i["id"]) for i in coco["images"]}
    missing_ann_refs = []
    missing_image_refs = []
    for ref in refs:
        if int(ref["image_id"]) not in image_ids:
            missing_image_refs.append(ref["id"])
        missing = [aid for aid in ref["target_ann_ids"] if int(aid) not in ann_ids]
        if missing:
            missing_ann_refs.append({"ref_id": ref["id"], "missing_ann_ids": missing})
    return {
        "num_missing_image_refs": len(missing_image_refs),
        "num_missing_ann_refs": len(missing_ann_refs),
        "missing_image_refs": missing_image_refs[:50],
        "missing_ann_refs": missing_ann_refs[:50],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="data/train")
    parser.add_argument("--valid-root", default="data/valid")
    parser.add_argument("--museg-root", default=r"E:/museg/museg")
    parser.add_argument("--output-root", default="data/tref_mine_refseg/sam3_train_valid_phrases")
    args = parser.parse_args()

    train_root = Path(args.train_root)
    valid_root = Path(args.valid_root)
    output_root = Path(args.output_root)

    train_raw = load_json(train_root / "_annotations.coco.json")
    valid_raw = load_json(valid_root / "_annotations.coco.json")
    train_coco = make_abs_coco(train_raw, train_root)
    valid_coco = make_abs_coco(valid_raw, valid_root)

    label_index = build_label_index(args.museg_root)
    verification = {
        "source_label_files_found": len(label_index),
        "train": verify_split(train_raw, train_root, label_index),
        "valid": verify_split(valid_raw, valid_root, label_index),
    }

    train_refs = build_generated_refs(train_coco, split="train", source_name="sam3_train_generated")
    valid_refs = build_generated_refs(valid_coco, split="valid", source_name="sam3_valid_generated")
    write_dataset(output_root / "train", train_coco, train_refs, {"verification": verification["train"]})
    write_dataset(output_root / "valid", valid_coco, valid_refs, {"verification": verification["valid"]})

    combined_coco, split_by_image = remap_and_combine(train_coco, valid_coco)
    combined_refs = build_generated_refs(
        combined_coco,
        split="train",
        source_name="sam3_train_valid_generated",
        split_by_image=split_by_image,
    )
    combined_validation = validate_ref_targets(combined_coco, combined_refs)
    write_dataset(
        output_root / "combined",
        combined_coco,
        combined_refs,
        {"verification": verification, "ref_target_validation": combined_validation},
    )
    write_json(output_root / "verification_report.json", verification)

    print("Wrote:", norm_path(output_root))
    print("train refs:", len(train_refs), "valid refs:", len(valid_refs), "combined refs:", len(combined_refs))
    print("source label files:", len(label_index))
    print("train mismatches:", verification["train"]["num_category_count_mismatches"], "valid mismatches:", verification["valid"]["num_category_count_mismatches"])
    print("train low bbox matches:", verification["train"]["num_low_bbox_iou_matches"], "valid low bbox matches:", verification["valid"]["num_low_bbox_iou_matches"])


if __name__ == "__main__":
    main()
