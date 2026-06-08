import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CATEGORY_ALIASES = {
    "person": "person",
    "cable": "cable",
    "tube": "tube",
    "indicator": "indicator",
    "electrical equipment": "electrical equipment",
    "electronic equipment": "electronic equipment",
    "mining equipment": "mining equipment",
    "rail area": "rail",
    "support equipment": "support equipment",
    "door": "door",
    "tools and materials": "tools",
    "rescue equipment": "rescue equipment",
    "container": "container",
    "metal fixture": "fixture",
    "anchoring equipment": "anchoring equipment",
}

NO_COLOR_CATEGORIES = {
    "person",
    "rail area",
    "tools and materials",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def norm_path(path):
    return str(Path(path).resolve()).replace("\\", "/")


def make_abs_coco(coco, image_root):
    out = {
        "info": dict(coco.get("info", {})),
        "licenses": list(coco.get("licenses", [])),
        "categories": list(coco.get("categories", [])),
        "images": [],
        "annotations": list(coco.get("annotations", [])),
    }
    image_root = Path(image_root)
    for image in coco.get("images", []):
        item = dict(image)
        file_name = Path(str(item["file_name"]))
        image_path = file_name if file_name.is_absolute() else image_root / file_name
        item["file_name"] = norm_path(image_path)
        item["split"] = "valid"
        out["images"].append(item)
    return out


def polygon_mask(segmentation, width, height):
    if not isinstance(segmentation, list) or not segmentation:
        return None
    mask = Image.new("1", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(mask)
    ok = False
    for poly in segmentation:
        if not isinstance(poly, list):
            continue
        if poly and isinstance(poly[0], (list, tuple)):
            flat = []
            for point in poly:
                if len(point) >= 2:
                    flat.extend([point[0], point[1]])
            poly = flat
        if len(poly) < 6:
            continue
        draw.polygon(list(zip(poly[0::2], poly[1::2])), outline=1, fill=1)
        ok = True
    if not ok:
        return None
    return np.array(mask, dtype=bool)


def sample_ann_pixels(image_np, ann, width, height, max_pixels=5000):
    mask = polygon_mask(ann.get("segmentation"), width, height)
    if mask is not None:
        pixels = image_np[mask]
    else:
        x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        x0 = max(0, int(math.floor(x)))
        y0 = max(0, int(math.floor(y)))
        x1 = min(int(width), int(math.ceil(x + w)))
        y1 = min(int(height), int(math.ceil(y + h)))
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 3), dtype=np.uint8)
        pixels = image_np[y0:y1, x0:x1].reshape(-1, 3)

    if len(pixels) > max_pixels:
        step = max(1, len(pixels) // max_pixels)
        pixels = pixels[::step][:max_pixels]
    return pixels


def rgb_to_hsv_np(rgb):
    arr = rgb.astype(np.float32) / 255.0
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    mx = arr.max(axis=1)
    mn = arr.min(axis=1)
    diff = mx - mn

    hue = np.zeros_like(mx)
    nonzero = diff > 1e-6
    red = nonzero & (mx == r)
    green = nonzero & (mx == g)
    blue = nonzero & (mx == b)
    hue[red] = ((g[red] - b[red]) / diff[red]) % 6.0
    hue[green] = ((b[green] - r[green]) / diff[green]) + 2.0
    hue[blue] = ((r[blue] - g[blue]) / diff[blue]) + 4.0
    hue = hue * 60.0

    sat = diff / np.maximum(mx, 1e-6)
    val = mx
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return hue, sat, val, luma


def strict_color_name(pixels, category_name):
    if category_name in NO_COLOR_CATEGORIES or len(pixels) < 30:
        return "", 0.0

    hue, sat, _, luma = rgb_to_hsv_np(pixels)

    # Neutral colors are only used when the object is overwhelmingly neutral.
    neutral = sat < 0.16
    black = neutral & (luma < 0.18)
    white = neutral & (luma > 0.80)
    gray = neutral & (luma >= 0.30) & (luma <= 0.72)
    neutral_labels = {
        "black": float(black.mean()),
        "white": float(white.mean()),
        "gray": float(gray.mean()),
    }
    neutral_color, neutral_share = max(neutral_labels.items(), key=lambda x: x[1])
    if neutral_share >= 0.85:
        return neutral_color, neutral_share

    chromatic = (sat >= 0.32) & (luma >= 0.16) & (luma <= 0.92)
    if chromatic.mean() < 0.45:
        return "", 0.0

    h = hue[chromatic]
    color_masks = {
        "red": (h < 15) | (h >= 345),
        "orange": (h >= 15) & (h < 35),
        "yellow": (h >= 35) & (h < 70),
        "green": (h >= 80) & (h < 165),
        "cyan": (h >= 165) & (h < 205),
        "blue": (h >= 205) & (h < 255),
        "purple": (h >= 255) & (h < 305),
    }
    shares = {name: float(mask.mean()) for name, mask in color_masks.items()}
    color, color_share = max(shares.items(), key=lambda x: x[1])
    all_pixel_share = color_share * float(chromatic.mean())
    if color_share >= 0.75 and all_pixel_share >= 0.75:
        return color, all_pixel_share
    return "", 0.0


def position_from_bbox(bbox, width, height):
    x, y, w, h = [float(v) for v in bbox]
    cx = (x + w / 2.0) / max(float(width), 1.0)
    cy = (y + h / 2.0) / max(float(height), 1.0)

    hpos = "left" if cx < 0.35 else "right" if cx > 0.65 else ""
    vpos = "upper" if cy < 0.35 else "lower" if cy > 0.65 else ""
    if hpos and vpos:
        return f"{vpos}-{hpos}"
    if hpos:
        return hpos
    if vpos:
        return vpos
    return "center"


def size_labels_for_image(anns):
    by_cat = defaultdict(list)
    for ann in anns:
        by_cat[int(ann["category_id"])].append(ann)

    labels = {}
    for _, items in by_cat.items():
        areas = [
            float(a.get("area") or (float(a["bbox"][2]) * float(a["bbox"][3])))
            for a in items
        ]
        if not areas:
            continue
        sorted_areas = sorted(areas)
        median_area = sorted_areas[len(sorted_areas) // 2]
        for ann, area in zip(items, areas):
            image_area = max(1.0, float(ann.get("_image_area", 1.0)))
            rel = area / image_area
            label = ""
            if len(items) == 1:
                if rel < 0.015:
                    label = "small"
                elif rel > 0.16:
                    label = "large"
            else:
                rank = sum(1 for a in sorted_areas if a <= area) / len(sorted_areas)
                if rank <= 0.25 and area <= median_area * 0.75:
                    label = "small"
                elif rank >= 0.80 and area >= median_area * 1.35:
                    label = "large"
            labels[int(ann["id"])] = label
    return labels


def expression_for_ann(category_name, features):
    noun = CATEGORY_ALIASES.get(category_name, category_name)
    parts = []
    position = features.get("position", "")
    size = features.get("size", "")
    color = features.get("color", "")

    if position:
        parts.append(position)
    if size:
        parts.append(size)
    if color:
        parts.append(color)
    parts.extend(noun.split())

    # Keep phrases compact. Color is retained when present because it passed a strict check.
    while len(parts) > 5 and size and size in parts:
        parts.remove(size)
        size = ""
    while len(parts) > 5 and position and position in parts:
        parts.remove(position)
        position = ""
    return " ".join(parts)


def build_refs(coco):
    categories = {int(c["id"]): str(c["name"]).strip().lower() for c in coco["categories"]}
    images = {int(img["id"]): img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    grouped = {}
    color_counter = Counter()
    phrase_len_counter = Counter()

    for image_id, anns in anns_by_image.items():
        image = images[image_id]
        width, height = int(image["width"]), int(image["height"])
        for ann in anns:
            ann["_image_area"] = width * height
        size_labels = size_labels_for_image(anns)

        try:
            image_np = np.array(Image.open(image["file_name"]).convert("RGB"))
        except Exception:
            image_np = None

        for ann in anns:
            ann_id = int(ann["id"])
            cat_id = int(ann["category_id"])
            category_name = categories[cat_id]
            color = ""
            color_confidence = 0.0
            if image_np is not None:
                pixels = sample_ann_pixels(image_np, ann, width, height)
                color, color_confidence = strict_color_name(pixels, category_name)

            features = {
                "class": category_name,
                "position": position_from_bbox(ann["bbox"], width, height),
                "size": size_labels.get(ann_id, ""),
                "color": color,
                "color_confidence": round(float(color_confidence), 4),
            }
            expression = expression_for_ann(category_name, features)
            phrase_len_counter[len(expression.split())] += 1
            if color:
                color_counter[color] += 1

            key = (image_id, expression)
            if key not in grouped:
                grouped[key] = {
                    "image_id": image_id,
                    "expression": expression,
                    "target_ann_ids": [],
                    "category_ids": set(),
                    "split": "valid",
                    "source": "sam3_valid_strict_generated",
                    "instruction_type": "simple",
                    "phrase_type": "position_size_color",
                    "attributes": dict(features),
                }
            grouped[key]["target_ann_ids"].append(ann_id)
            grouped[key]["category_ids"].add(cat_id)

    refs = []
    for idx, item in enumerate(sorted(grouped.values(), key=lambda x: (x["image_id"], x["expression"]))):
        target_ids = sorted(set(int(x) for x in item["target_ann_ids"]))
        category_ids = sorted(int(x) for x in item["category_ids"])
        refs.append(
            {
                "id": idx + 1,
                "image_id": int(item["image_id"]),
                "expression": item["expression"],
                "target_ann_ids": target_ids,
                "category_ids": category_ids,
                "is_unique": len(target_ids) == 1,
                "num_targets": len(target_ids),
                "split": item["split"],
                "source": item["source"],
                "instruction_type": item["instruction_type"],
                "phrase_type": item["phrase_type"],
                "attributes": item["attributes"],
            }
        )

    stats_extra = {
        "color_phrase_counts": dict(color_counter),
        "phrase_length_histogram": dict(sorted(phrase_len_counter.items())),
    }
    return refs, stats_extra


def validate_refs(coco, refs):
    ann_ids = {int(a["id"]) for a in coco.get("annotations", [])}
    image_ids = {int(i["id"]) for i in coco.get("images", [])}
    missing_images = []
    missing_anns = []
    long_phrases = []
    for ref in refs:
        if int(ref["image_id"]) not in image_ids:
            missing_images.append(ref["id"])
        missing = [x for x in ref["target_ann_ids"] if int(x) not in ann_ids]
        if missing:
            missing_anns.append({"ref_id": ref["id"], "missing": missing})
        if len(ref["expression"].split()) > 5:
            long_phrases.append({"ref_id": ref["id"], "expression": ref["expression"]})
    return {
        "num_missing_image_refs": len(missing_images),
        "num_missing_ann_refs": len(missing_anns),
        "num_longer_than_5_words": len(long_phrases),
        "missing_image_refs": missing_images[:50],
        "missing_ann_refs": missing_anns[:50],
        "longer_than_5_words": long_phrases[:50],
    }


def dataset_stats(coco, refs, extra):
    target_hist = Counter(int(r["num_targets"]) for r in refs)
    phrase_types = Counter(r.get("phrase_type", "") for r in refs)
    unique_count = sum(1 for r in refs if r["is_unique"])
    return {
        "num_images": len(coco.get("images", [])),
        "num_annotations": len(coco.get("annotations", [])),
        "num_categories": len(coco.get("categories", [])),
        "num_referring_annotations": len(refs),
        "num_unique_refs": unique_count,
        "num_multi_target_refs": len(refs) - unique_count,
        "target_count_histogram": dict(sorted(target_hist.items())),
        "phrase_type_counts": dict(phrase_types),
        "examples": refs[:30],
        "extra": extra,
    }


def main():
    parser = argparse.ArgumentParser(description="Build strict short referring expressions for data/valid.")
    parser.add_argument("--input-root", default="data/valid")
    parser.add_argument("--input-json", default="data/valid/_annotations.coco.json")
    parser.add_argument("--output-root", default="data/valid_tref_strict")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    raw_coco = load_json(args.input_json)
    coco = make_abs_coco(raw_coco, input_root)
    refs, extra = build_refs(coco)
    validation = validate_refs(coco, refs)
    extra["validation"] = validation

    write_json(output_root / "annotations.json", coco)
    write_json(output_root / "referring_annotations.json", refs)
    write_json(output_root / "stats.json", dataset_stats(coco, refs, extra))

    print(f"Wrote: {norm_path(output_root)}")
    print(f"Images: {len(coco.get('images', []))}")
    print(f"Annotations: {len(coco.get('annotations', []))}")
    print(f"Referring annotations: {len(refs)}")
    print(f"Unique refs: {sum(1 for r in refs if r['is_unique'])}")
    print(f"Multi-target refs: {sum(1 for r in refs if not r['is_unique'])}")
    print(f"Colors used: {extra['color_phrase_counts']}")
    print(f"Validation: {validation}")


if __name__ == "__main__":
    main()
