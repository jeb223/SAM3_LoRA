import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import build_text_refseg_dataset as gen


EXTERNAL_CATEGORY_NAMES = {
    "Coal miner": "coal miner",
    "coal miner": "coal miner",
    "large_coal": "large coal",
    "Mine_Safety_Helmet": "mine safety helmet",
    "towline": "towline",
    "walking": "walking miner",
    "sitting": "sitting miner",
    "standing": "standing miner",
    "operation": "operating miner",
    "stoop": "stooping miner",
    "lean against": "leaning miner",
    "tumble": "fallen miner",
    "climb over": "climbing miner",
    "Shearer": "shearer",
    "hydraulic_support_guard_plate_90": "hydraulic support guard plate 90",
    "hydraulic_support_guard_plate_60_90": "hydraulic support guard plate 60 90",
    "hydraulic_support_guard_plate_00_30": "hydraulic support guard plate 0 30",
    "hydraulic_support_guard_plate_90_120": "hydraulic support guard plate 90 120",
    "hydraulic_support_guard_plate_30_60": "hydraulic support guard plate 30 60",
    "hydraulic_support_guard_plate_00": "hydraulic support guard plate 0",
    "hydraulic_support_guard_plate_abnormal": "abnormal hydraulic support guard plate",
    "hydraulic_support_guard_plate_90_abnormal": "abnormal hydraulic support guard plate 90",
}


EXTERNAL_ALIASES = {
    "coal miner": ["coal miner", "miner", "person"],
    "large coal": ["large coal", "coal block", "coal"],
    "mine safety helmet": ["mine safety helmet", "safety helmet", "helmet"],
    "towline": ["towline", "cable", "line"],
    "walking miner": ["walking miner", "miner", "person"],
    "sitting miner": ["sitting miner", "miner", "person"],
    "standing miner": ["standing miner", "miner", "person"],
    "operating miner": ["operating miner", "miner", "person"],
    "stooping miner": ["stooping miner", "miner", "person"],
    "leaning miner": ["leaning miner", "miner", "person"],
    "fallen miner": ["fallen miner", "miner", "person"],
    "climbing miner": ["climbing miner", "miner", "person"],
    "shearer": ["shearer", "mining machine", "machine"],
    "hydraulic support guard plate 90": ["hydraulic support guard plate", "guard plate"],
    "hydraulic support guard plate 60 90": ["hydraulic support guard plate", "guard plate"],
    "hydraulic support guard plate 0 30": ["hydraulic support guard plate", "guard plate"],
    "hydraulic support guard plate 90 120": ["hydraulic support guard plate", "guard plate"],
    "hydraulic support guard plate 30 60": ["hydraulic support guard plate", "guard plate"],
    "hydraulic support guard plate 0": ["hydraulic support guard plate", "guard plate"],
    "abnormal hydraulic support guard plate": ["abnormal guard plate", "guard plate"],
    "abnormal hydraulic support guard plate 90": ["abnormal guard plate", "guard plate"],
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def source_dirs(external_root):
    root = Path(external_root)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def normalize_category_name(name):
    return EXTERNAL_CATEGORY_NAMES.get(name, name.replace("_", " ").strip().lower())


def pick_images(coco, n, seed, dataset_name, split):
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)
    candidates = [img for img in coco.get("images", []) if anns_by_image.get(int(img["id"]))]
    rng = random.Random(f"{seed}:{dataset_name}:{split}")
    candidates = sorted(candidates, key=lambda x: str(x.get("file_name", "")))
    if len(candidates) <= n:
        return candidates
    return sorted(rng.sample(candidates, n), key=lambda x: str(x.get("file_name", "")))


def register_category(category_map, categories, source_dataset, original_cat):
    original_name = str(original_cat["name"])
    readable_name = normalize_category_name(original_name)
    key = (source_dataset, int(original_cat["id"]), readable_name)
    if key in category_map:
        return category_map[key]
    new_id = len(categories) + 1
    category_map[key] = new_id
    categories.append(
        {
            "id": new_id,
            "name": readable_name,
            "source_dataset": source_dataset,
            "source_category_id": int(original_cat["id"]),
            "source_category_name": original_name,
        }
    )
    return new_id


def append_selected_split(
    combined,
    source_coco,
    source_root,
    selected_images,
    source_dataset,
    split,
    category_map,
):
    original_cats = {int(c["id"]): c for c in source_coco.get("categories", [])}
    anns_by_image = defaultdict(list)
    for ann in source_coco.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    next_image_id = max([0] + [int(x["id"]) for x in combined["images"]]) + 1
    next_ann_id = max([0] + [int(x["id"]) for x in combined["annotations"]]) + 1
    image_id_map = {}

    for img in selected_images:
        old_image_id = int(img["id"])
        new_image_id = next_image_id
        next_image_id += 1
        image_id_map[old_image_id] = new_image_id

        file_name = Path(str(img["file_name"]))
        image_path = file_name if file_name.is_absolute() else Path(source_root) / file_name
        new_img = dict(img)
        new_img["id"] = new_image_id
        new_img["file_name"] = gen.norm_path(image_path)
        new_img["split"] = split
        new_img["source_dataset"] = source_dataset
        new_img["source_image_id"] = old_image_id
        combined["images"].append(new_img)

        for ann in anns_by_image.get(old_image_id, []):
            source_cat_id = int(ann["category_id"])
            new_cat_id = register_category(
                category_map,
                combined["categories"],
                source_dataset,
                original_cats[source_cat_id],
            )
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = new_image_id
            new_ann["category_id"] = new_cat_id
            new_ann["source_dataset"] = source_dataset
            new_ann["source_split"] = split
            new_ann["source_image_id"] = old_image_id
            new_ann["source_ann_id"] = int(ann["id"])
            new_ann["source_category_id"] = source_cat_id
            combined["annotations"].append(new_ann)
            next_ann_id += 1


def validate_coco(coco):
    report = {
        "missing_image_files": [],
        "dimension_mismatches": [],
        "bbox_out_of_bounds": [],
        "segmentation_out_of_bounds": [],
    }
    images = {int(x["id"]): x for x in coco.get("images", [])}
    for img in coco.get("images", []):
        path = Path(img["file_name"])
        if not path.exists():
            report["missing_image_files"].append(img["file_name"])
            continue
        try:
            with Image.open(path) as im:
                actual_w, actual_h = im.size
        except Exception:
            report["missing_image_files"].append(img["file_name"])
            continue
        if (int(img["width"]), int(img["height"])) != (actual_w, actual_h):
            report["dimension_mismatches"].append(
                {"file_name": img["file_name"], "json": [img["width"], img["height"]], "actual": [actual_w, actual_h]}
            )

    for ann in coco.get("annotations", []):
        img = images[int(ann["image_id"])]
        width, height = int(img["width"]), int(img["height"])
        x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        if x < -1 or y < -1 or w <= 0 or h <= 0 or x + w > width + 1 or y + h > height + 1:
            report["bbox_out_of_bounds"].append({"ann_id": ann["id"], "image_id": ann["image_id"], "bbox": ann.get("bbox")})
        for poly in ann.get("segmentation", []) or []:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            xs, ys = poly[0::2], poly[1::2]
            if min(xs) < -1 or min(ys) < -1 or max(xs) > width + 1 or max(ys) > height + 1:
                report["segmentation_out_of_bounds"].append(
                    {
                        "ann_id": ann["id"],
                        "image_id": ann["image_id"],
                        "x_range": [min(xs), max(xs)],
                        "y_range": [min(ys), max(ys)],
                    }
                )

    for key in list(report.keys()):
        report[f"num_{key}"] = len(report[key])
        report[key] = report[key][:50]
    return report


def split_coco(coco, split):
    image_ids = {int(img["id"]) for img in coco["images"] if img.get("split") == split}
    return {
        "info": dict(coco.get("info", {})),
        "licenses": list(coco.get("licenses", [])),
        "categories": list(coco.get("categories", [])),
        "images": [img for img in coco["images"] if int(img["id"]) in image_ids],
        "annotations": [ann for ann in coco["annotations"] if int(ann["image_id"]) in image_ids],
    }


def validate_refs(coco, refs):
    ann_ids = {int(x["id"]) for x in coco.get("annotations", [])}
    image_ids = {int(x["id"]) for x in coco.get("images", [])}
    return {
        "missing_image_refs": [r["id"] for r in refs if int(r["image_id"]) not in image_ids][:50],
        "missing_ann_refs": [
            {"ref_id": r["id"], "missing": [x for x in r["target_ann_ids"] if int(x) not in ann_ids]}
            for r in refs
            if any(int(x) not in ann_ids for x in r["target_ann_ids"])
        ][:50],
    }


def write_split_dataset(output_dir, coco, split, source_name, verification_extra=None):
    refs = gen.build_generated_refs(coco, split=split, source_name=source_name)
    ref_validation = validate_refs(coco, refs)
    gen.write_dataset(output_dir, coco, refs, {"verification": verification_extra or {}, "ref_validation": ref_validation})
    return refs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", default="data/external")
    parser.add_argument("--output-root", default="data/tref_mine_refseg/external_400_100_phrases")
    parser.add_argument("--train-per-type", type=int, default=400)
    parser.add_argument("--valid-per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    gen.NOUN_ALIASES.update(EXTERNAL_ALIASES)
    gen.NO_COLOR_CATEGORIES.update({"coal miner", "walking miner", "sitting miner", "standing miner", "operating miner", "stooping miner", "leaning miner", "fallen miner", "climbing miner"})

    combined = {
        "info": {
            "description": "External underground/mining text-guided referring segmentation dataset",
            "train_per_type": args.train_per_type,
            "valid_per_type": args.valid_per_type,
            "seed": args.seed,
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }
    category_map = {}
    sampling_report = {}

    for dataset_dir in source_dirs(args.external_root):
        sampling_report[dataset_dir.name] = {}
        for split, n in [("train", args.train_per_type), ("valid", args.valid_per_type)]:
            split_root = dataset_dir / split
            ann_path = split_root / "_annotations.coco.json"
            source_coco = load_json(ann_path)
            selected = pick_images(source_coco, n, args.seed, dataset_dir.name, split)
            append_selected_split(
                combined=combined,
                source_coco=source_coco,
                source_root=split_root,
                selected_images=selected,
                source_dataset=dataset_dir.name,
                split=split,
                category_map=category_map,
            )
            sampling_report[dataset_dir.name][split] = {
                "available_images_with_annotations": len([img for img in source_coco.get("images", [])]),
                "selected_images": len(selected),
                "selected_file_names": [img["file_name"] for img in selected[:20]],
            }

    output_root = Path(args.output_root)
    train_coco = split_coco(combined, "train")
    valid_coco = split_coco(combined, "valid")

    train_verify = validate_coco(train_coco)
    valid_verify = validate_coco(valid_coco)
    combined_verify = validate_coco(combined)

    train_refs = write_split_dataset(
        output_root / "train",
        train_coco,
        "train",
        f"external_{args.train_per_type}_train_generated",
        train_verify,
    )
    valid_refs = write_split_dataset(
        output_root / "valid",
        valid_coco,
        "valid",
        f"external_{args.valid_per_type}_valid_generated",
        valid_verify,
    )
    split_by_image = {int(img["id"]): img.get("split", "train") for img in combined["images"]}
    combined_refs = gen.build_generated_refs(
        combined,
        split="train",
        source_name=f"external_{args.train_per_type}_{args.valid_per_type}_generated",
        split_by_image=split_by_image,
    )
    gen.write_dataset(
        output_root / "combined",
        combined,
        combined_refs,
        {"verification": combined_verify, "sampling_report": sampling_report, "ref_validation": validate_refs(combined, combined_refs)},
    )
    gen.write_json(
        output_root / "sampling_report.json",
        {
            "sampling": sampling_report,
            "train_verification": train_verify,
            "valid_verification": valid_verify,
            "combined_verification": combined_verify,
        },
    )

    print("Wrote:", gen.norm_path(output_root))
    print("train images:", len(train_coco["images"]), "train anns:", len(train_coco["annotations"]), "train refs:", len(train_refs))
    print("valid images:", len(valid_coco["images"]), "valid anns:", len(valid_coco["annotations"]), "valid refs:", len(valid_refs))
    print("combined categories:", len(combined["categories"]))
    print("verification train:", {k: v for k, v in train_verify.items() if k.startswith("num_")})
    print("verification valid:", {k: v for k, v in valid_verify.items() if k.startswith("num_")})


if __name__ == "__main__":
    main()
