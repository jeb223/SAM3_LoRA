import argparse
import colorsys
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


EQUIPMENT_ALIASES = {
    "electrical equipment",
    "electronic equipment",
    "mining equipment",
    "support equipment",
    "rescue equipment",
}

NOUN_ALIASES = {
    "cable": ["cable", "wire", "line"],
    "tube": ["tube", "pipe"],
    "indicator": ["indicator", "label"],
    "electrical equipment": ["electrical equipment", "equipment"],
    "electronic equipment": ["electronic equipment", "equipment"],
    "mining equipment": ["mining equipment", "equipment"],
    "support equipment": ["support equipment", "support unit", "equipment"],
    "rescue equipment": ["rescue equipment", "equipment"],
    "metal fixture": ["metal fixture", "metal support"],
}

NO_COLOR_CATEGORIES = {"person", "rail area"}


def norm_path(path):
    return str(Path(path).resolve()).replace("\\", "/")


def clean_expr(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def polygon_area(flat_points):
    xs = flat_points[0::2]
    ys = flat_points[1::2]
    if len(xs) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        acc += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(acc) * 0.5


def polygon_bbox(flat_points):
    xs = flat_points[0::2]
    ys = flat_points[1::2]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_abs_coco(coco, image_root):
    out = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": [],
        "annotations": coco.get("annotations", []),
        "categories": coco.get("categories", []),
    }
    image_root = Path(image_root)
    for img in coco.get("images", []):
        new_img = dict(img)
        file_name = Path(str(new_img["file_name"]))
        if file_name.is_absolute():
            img_path = file_name
        else:
            img_path = image_root / file_name
        new_img["file_name"] = norm_path(img_path)
        out["images"].append(new_img)
    return out


def read_stems(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_refmuseg_full(ref_root, museg_root):
    ref_root = Path(ref_root)
    museg_root = Path(museg_root)
    base = load_json(ref_root / "annotations.json")

    coco = make_abs_coco(base, ref_root)
    categories = {int(c["id"]): c["name"] for c in coco["categories"]}
    cat_id_by_name = {name.lower(): cid for cid, name in categories.items()}

    max_image_id = max(int(img["id"]) for img in coco["images"]) if coco["images"] else 0
    max_ann_id = max(int(ann["id"]) for ann in coco["annotations"]) if coco["annotations"] else 0
    val_txt = museg_root / "val.txt"
    label_dir = museg_root / "Label"
    image_dir = museg_root / "Image"
    out_image_dir = ref_root / "images"
    unknown_labels = Counter()

    next_image_id = max_image_id + 1
    next_ann_id = max_ann_id + 1

    for stem in read_stems(val_txt):
        image_path = out_image_dir / f"{stem}.jpg"
        if not image_path.exists():
            image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}_polygons.json"
        if not label_path.exists() or not image_path.exists():
            continue

        with Image.open(image_path) as im:
            width, height = im.size

        image_id = next_image_id
        next_image_id += 1
        coco["images"].append(
            {
                "id": image_id,
                "file_name": norm_path(image_path),
                "width": width,
                "height": height,
            }
        )

        label_data = load_json(label_path)
        for shape in label_data.get("shapes", []):
            label = str(shape.get("label", "")).strip().lower()
            if not label or label == "__background__":
                continue
            if label not in cat_id_by_name:
                unknown_labels[label] += 1
                continue
            pts = shape.get("points") or []
            if len(pts) < 3:
                continue
            flat = []
            for point in pts:
                if len(point) >= 2:
                    flat.extend([float(point[0]), float(point[1])])
            if len(flat) < 6:
                continue
            coco["annotations"].append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": int(cat_id_by_name[label]),
                    "segmentation": [flat],
                    "area": float(polygon_area(flat)),
                    "bbox": polygon_bbox(flat),
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    return coco, dict(unknown_labels)


def mask_from_polygons(segmentation, width, height):
    if not isinstance(segmentation, list) or not segmentation:
        return None
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    ok = False
    for poly in segmentation:
        if isinstance(poly, dict):
            continue
        if poly and isinstance(poly[0], (list, tuple)):
            flat = []
            for pair in poly:
                flat.extend(pair[:2])
            poly = flat
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        pts = list(zip(poly[0::2], poly[1::2]))
        draw.polygon(pts, outline=1, fill=1)
        ok = True
    if not ok:
        return None
    return np.array(mask, dtype=bool)


def sampled_mask_pixels(img_np, ann, width, height, max_pixels=4096):
    mask = mask_from_polygons(ann.get("segmentation"), width, height)
    if mask is None:
        x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
        x0 = max(0, int(math.floor(x)))
        y0 = max(0, int(math.floor(y)))
        x1 = min(width, int(math.ceil(x + w)))
        y1 = min(height, int(math.ceil(y + h)))
        if x1 <= x0 or y1 <= y0:
            return np.empty((0, 3), dtype=np.uint8)
        pixels = img_np[y0:y1, x0:x1].reshape(-1, 3)
    else:
        pixels = img_np[mask]
    if len(pixels) > max_pixels:
        step = max(1, len(pixels) // max_pixels)
        pixels = pixels[::step][:max_pixels]
    return pixels


def dominant_color(pixels, category_name):
    if category_name in NO_COLOR_CATEGORIES or len(pixels) == 0:
        return "", 0.0
    arr = pixels.astype(np.float32) / 255.0
    mx = arr.max(axis=1)
    mn = arr.min(axis=1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    luma = 0.299 * arr[:, 0] + 0.587 * arr[:, 1] + 0.114 * arr[:, 2]
    sat_med = float(np.median(sat))
    luma_med = float(np.median(luma))
    if sat_med < 0.15:
        if luma_med < 0.25:
            return "black", 0.65
        if luma_med > 0.72:
            return "white", 0.65
        return "gray", 0.55

    hues = []
    for r, g, b in arr[:: max(1, len(arr) // 1024)]:
        h, _, _ = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        hues.append(h * 360.0)
    if not hues:
        return "", 0.0
    h = float(np.median(hues))
    if h < 15 or h >= 345:
        color = "red"
    elif h < 35:
        color = "orange"
    elif h < 70:
        color = "yellow"
    elif h < 170:
        color = "green"
    elif h < 205:
        color = "cyan"
    elif h < 255:
        color = "blue"
    elif h < 300:
        color = "purple"
    else:
        color = "red"
    return color, min(1.0, max(0.5, sat_med))


def region_from_bbox(bbox, width, height):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / max(width, 1)
    cy = (y + h / 2.0) / max(height, 1)
    hpos = "left" if cx < 0.34 else "right" if cx > 0.66 else "center"
    vpos = "upper" if cy < 0.34 else "lower" if cy > 0.66 else "middle"
    if hpos == "center" and vpos == "middle":
        return "center", hpos, vpos
    if hpos == "center":
        return f"{vpos} center", hpos, vpos
    if vpos == "middle":
        return f"middle {hpos}", hpos, vpos
    return f"{vpos} {hpos}", hpos, vpos


def shape_from_bbox(bbox):
    _, _, w, h = bbox
    if w <= 0 or h <= 0:
        return "", ""
    ratio = max(w / h, h / w)
    orient = "horizontal" if w >= h else "vertical"
    if ratio >= 7:
        return "thin", orient
    if ratio >= 3.5:
        return ("long" if w >= h else "tall"), orient
    return "", orient


def bbox_center(ann):
    x, y, w, h = ann["bbox"]
    return x + w / 2.0, y + h / 2.0


def bbox_gap(a, b):
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    dx = max(0.0, max(bx - ax1, ax - bx1))
    dy = max(0.0, max(by - ay1, ay - by1))
    return math.hypot(dx, dy)


def relation_to(a, b, cat_name):
    acx, acy = bbox_center(a)
    bcx, bcy = bbox_center(b)
    dx, dy = bcx - acx, bcy - acy
    if abs(dx) > abs(dy) * 1.15:
        direction = "to the left of" if dx > 0 else "to the right of"
    elif abs(dy) > abs(dx) * 1.15:
        direction = "above" if dy > 0 else "below"
    else:
        direction = "near"
    return direction, cat_name


def size_labels_for_image(anns):
    by_cat = defaultdict(list)
    for ann in anns:
        by_cat[int(ann["category_id"])].append(ann)
    labels = {}
    for _, items in by_cat.items():
        areas = sorted(float(a.get("area") or a["bbox"][2] * a["bbox"][3]) for a in items)
        for ann in items:
            area = float(ann.get("area") or ann["bbox"][2] * ann["bbox"][3])
            if len(items) == 1:
                img_area = max(1.0, float(ann.get("_image_area", 1.0)))
                ratio = area / img_area
                labels[ann["id"]] = "small" if ratio < 0.02 else "large" if ratio > 0.18 else ""
                continue
            rank = sum(1 for x in areas if x <= area) / len(areas)
            if rank <= 0.25:
                labels[ann["id"]] = "small"
            elif rank >= 0.80:
                labels[ann["id"]] = "large"
            else:
                labels[ann["id"]] = ""
    return labels


def ordinal_labels_for_image(anns):
    out = {}
    by_cat = defaultdict(list)
    for ann in anns:
        by_cat[int(ann["category_id"])].append(ann)
    ord_words = ["leftmost", "second from left", "third from left", "fourth from left", "fifth from left"]
    for _, items in by_cat.items():
        items = sorted(items, key=lambda a: bbox_center(a)[0])
        n = len(items)
        if n <= 1:
            continue
        for idx, ann in enumerate(items):
            if idx == 0:
                out[ann["id"]] = "leftmost"
            elif idx == n - 1:
                out[ann["id"]] = "rightmost"
            elif idx < len(ord_words):
                out[ann["id"]] = ord_words[idx]
    return out


def category_nouns(category_name):
    return NOUN_ALIASES.get(category_name, [category_name])


def add_phrase(phrases, text, phrase_type, attrs):
    text = clean_expr(text)
    if text:
        phrases.append((text, phrase_type, dict(attrs)))


def generate_phrases_for_ann(ann, features, category_name, nearest_category):
    nouns = category_nouns(category_name)
    base_noun = nouns[0]
    region = features.get("region", "")
    hpos = features.get("hpos", "")
    color = features.get("color", "")
    size = features.get("size", "")
    shape = features.get("shape", "")
    orient = features.get("orientation", "")
    ordinal = features.get("ordinal", "")
    rel_dir = features.get("relation_direction", "")
    rel_cat = nearest_category or ""
    attrs = {
        "class": category_name,
        "color": color,
        "size": size,
        "shape": shape,
        "orientation": orient,
        "position": region,
        "horizontal_position": hpos,
        "ordinal": ordinal,
        "relation": f"{rel_dir} {rel_cat}".strip() if rel_cat else "",
    }

    phrases = []
    for noun in nouns:
        add_phrase(phrases, f"the {noun}", "category", attrs)

    for noun in nouns[:2]:
        if region:
            add_phrase(phrases, f"the {noun} in the {region}", "position", attrs)
        if ordinal and region:
            add_phrase(phrases, f"the {ordinal} {noun} in the {region}", "position", attrs)
        elif ordinal:
            add_phrase(phrases, f"the {ordinal} {noun}", "position", attrs)
        if hpos in {"left", "right"}:
            add_phrase(phrases, f"the {hpos} {noun}", "position", attrs)

        adjective_sets = []
        if size:
            adjective_sets.append([size])
        if color:
            adjective_sets.append([color])
        if shape:
            adjective_sets.append([shape])
        if size and color:
            adjective_sets.append([size, color])
        if color and shape:
            adjective_sets.append([color, shape])
        if size and shape:
            adjective_sets.append([size, shape])
        if size and color and shape:
            adjective_sets.append([size, color, shape])
        for adj in adjective_sets:
            add_phrase(phrases, f"the {' '.join(adj)} {noun}", "attribute", attrs)
            if region:
                add_phrase(phrases, f"the {' '.join(adj)} {noun} in the {region}", "specific", attrs)
            if hpos in {"left", "right"}:
                add_phrase(phrases, f"the {hpos} {' '.join(adj)} {noun}", "specific", attrs)

        if rel_cat:
            add_phrase(phrases, f"the {noun} near the {rel_cat}", "relation", attrs)
            if rel_dir and rel_dir != "near":
                add_phrase(phrases, f"the {noun} {rel_dir} the {rel_cat}", "relation", attrs)
            if shape:
                add_phrase(phrases, f"the {shape} {noun} near the {rel_cat}", "relation", attrs)

    if category_name in EQUIPMENT_ALIASES and size:
        add_phrase(phrases, f"the {size} equipment", "attribute", attrs)
        if hpos in {"left", "right"}:
            add_phrase(phrases, f"the {hpos} {size} equipment", "specific", attrs)

    seen = set()
    unique = []
    for item in phrases:
        if item[0] in seen:
            continue
        seen.add(item[0])
        unique.append(item)
    return unique


def compute_image_features(coco):
    categories = {int(c["id"]): c["name"] for c in coco["categories"]}
    images = {int(img["id"]): img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    features = {}
    for image_id, anns in anns_by_image.items():
        img = images[image_id]
        width = int(img["width"])
        height = int(img["height"])
        for ann in anns:
            ann["_image_area"] = width * height
        size_labels = size_labels_for_image(anns)
        ordinal_labels = ordinal_labels_for_image(anns)

        img_np = None
        try:
            img_np = np.array(Image.open(img["file_name"]).convert("RGB"))
        except Exception:
            img_np = None

        for ann in anns:
            cat_name = categories[int(ann["category_id"])]
            region, hpos, vpos = region_from_bbox(ann["bbox"], width, height)
            shape, orient = shape_from_bbox(ann["bbox"])
            color = ""
            color_conf = 0.0
            if img_np is not None:
                pixels = sampled_mask_pixels(img_np, ann, width, height)
                color, color_conf = dominant_color(pixels, cat_name)
                if color_conf < 0.52:
                    color = ""

            nearest = None
            nearest_dist = float("inf")
            for other in anns:
                if other["id"] == ann["id"]:
                    continue
                dist = bbox_gap(ann, other)
                if int(other["category_id"]) != int(ann["category_id"]):
                    dist *= 0.75
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = other

            relation_direction = ""
            relation_category = ""
            if nearest is not None:
                relation_category = category_nouns(categories[int(nearest["category_id"])])[0]
                relation_direction, _ = relation_to(ann, nearest, relation_category)

            features[int(ann["id"])] = {
                "region": region,
                "hpos": hpos,
                "vpos": vpos,
                "size": size_labels.get(ann["id"], ""),
                "shape": shape,
                "orientation": orient,
                "color": color,
                "color_confidence": color_conf,
                "ordinal": ordinal_labels.get(ann["id"], ""),
                "relation_direction": relation_direction,
                "relation_category": relation_category,
            }
    return features


def build_generated_refs(coco, split, source_name, split_by_image=None):
    categories = {int(c["id"]): c["name"] for c in coco["categories"]}
    features = compute_image_features(coco)
    grouped = {}

    for ann in coco["annotations"]:
        ann_id = int(ann["id"])
        image_id = int(ann["image_id"])
        cat_name = categories[int(ann["category_id"])]
        feat = features.get(ann_id, {})
        phrases = generate_phrases_for_ann(ann, feat, cat_name, feat.get("relation_category", ""))
        for expr, phrase_type, attrs in phrases:
            item_split = split_by_image.get(image_id, split) if split_by_image else split
            key = (image_id, expr, phrase_type)
            if key not in grouped:
                grouped[key] = {
                    "image_id": image_id,
                    "expression": expr,
                    "target_ann_ids": [],
                    "category_ids": set(),
                    "split": item_split,
                    "source": source_name,
                    "phrase_type": phrase_type,
                    "attributes": attrs,
                }
            grouped[key]["target_ann_ids"].append(ann_id)
            grouped[key]["category_ids"].add(int(ann["category_id"]))

    return finalize_grouped_refs(grouped.values())


def build_refmuseg_existing_refs(ref_pickle_path, valid_ann_ids, source_name):
    with open(ref_pickle_path, "rb") as f:
        refs = pickle.load(f)
    grouped = {}
    for ref in refs:
        ann_id = int(ref["ann_id"])
        if ann_id not in valid_ann_ids:
            continue
        image_id = int(ref["image_id"])
        split = str(ref.get("split", "train"))
        for sent in ref.get("sentences", []):
            expr = clean_expr(sent.get("raw") or sent.get("sent") or "")
            key = (image_id, expr, split)
            if key not in grouped:
                grouped[key] = {
                    "image_id": image_id,
                    "expression": expr,
                    "target_ann_ids": [],
                    "category_ids": set(),
                    "split": split,
                    "source": source_name,
                    "phrase_type": "existing_ref",
                    "attributes": {},
                }
            grouped[key]["target_ann_ids"].append(ann_id)
            grouped[key]["category_ids"].add(int(ref["category_id"]))
    return finalize_grouped_refs(grouped.values())


def finalize_grouped_refs(items):
    out = []
    for idx, item in enumerate(sorted(items, key=lambda x: (x["image_id"], x["expression"], x["phrase_type"]))):
        target_ids = sorted(set(int(x) for x in item["target_ann_ids"]))
        category_ids = sorted(int(x) for x in item["category_ids"])
        out.append(
            {
                "id": idx + 1,
                "image_id": int(item["image_id"]),
                "expression": item["expression"],
                "target_ann_ids": target_ids,
                "category_ids": category_ids,
                "is_unique": len(target_ids) == 1,
                "num_targets": len(target_ids),
                "split": item.get("split", "train"),
                "source": item.get("source", ""),
                "phrase_type": item.get("phrase_type", ""),
                "attributes": item.get("attributes", {}),
            }
        )
    return out


def merge_refs(*ref_lists):
    grouped = {}
    for refs in ref_lists:
        for ref in refs:
            key = (ref["image_id"], ref["expression"], ref.get("split", "train"))
            if key not in grouped:
                grouped[key] = {
                    "image_id": ref["image_id"],
                    "expression": ref["expression"],
                    "target_ann_ids": [],
                    "category_ids": set(),
                    "split": ref.get("split", "train"),
                    "source": ref.get("source", ""),
                    "phrase_type": ref.get("phrase_type", ""),
                    "attributes": ref.get("attributes", {}),
                }
            grouped[key]["target_ann_ids"].extend(ref["target_ann_ids"])
            grouped[key]["category_ids"].update(ref.get("category_ids", []))
            if ref.get("source") and ref["source"] not in grouped[key]["source"]:
                grouped[key]["source"] = (grouped[key]["source"] + "+" + ref["source"]).strip("+")
            if ref.get("phrase_type") and ref["phrase_type"] not in grouped[key]["phrase_type"]:
                grouped[key]["phrase_type"] = (grouped[key]["phrase_type"] + "+" + ref["phrase_type"]).strip("+")
    return finalize_grouped_refs(grouped.values())


def dataset_stats(coco, refs, extra=None):
    split_counts = Counter(ref.get("split", "train") for ref in refs)
    phrase_counts = Counter(ref.get("phrase_type", "") for ref in refs)
    target_hist = Counter(ref["num_targets"] for ref in refs)
    return {
        "num_images": len(coco.get("images", [])),
        "num_annotations": len(coco.get("annotations", [])),
        "num_categories": len(coco.get("categories", [])),
        "num_referring_annotations": len(refs),
        "num_unique_refs": sum(1 for r in refs if r["is_unique"]),
        "num_multi_target_refs": sum(1 for r in refs if not r["is_unique"]),
        "split_counts": dict(split_counts),
        "phrase_type_counts": dict(phrase_counts),
        "target_count_histogram": dict(sorted(target_hist.items())),
        "examples": refs[:20],
        "extra": extra or {},
    }


def write_dataset(out_dir, coco, refs, extra_stats=None):
    out_dir = Path(out_dir)
    write_json(out_dir / "annotations.json", coco)
    write_json(out_dir / "referring_annotations.json", refs)
    write_json(out_dir / "stats.json", dataset_stats(coco, refs, extra_stats))


def main():
    parser = argparse.ArgumentParser(description="Build text-guided referring segmentation annotations.")
    parser.add_argument("--output-root", default="data/tref_mine_refseg")
    parser.add_argument("--sam3-train-root", default="data/train")
    parser.add_argument("--sam3-train-json", default="data/train/_annotations.coco.json")
    parser.add_argument("--refmuseg-root", default=r"E:/museg/refmuseg")
    parser.add_argument("--museg-root", default=r"E:/museg")
    args = parser.parse_args()

    output_root = Path(args.output_root)

    sam_coco_raw = load_json(args.sam3_train_json)
    sam_coco = make_abs_coco(sam_coco_raw, args.sam3_train_root)
    sam_refs = build_generated_refs(sam_coco, split="train", source_name="sam3_train_generated")
    write_dataset(output_root / "sam3_train_phrases", sam_coco, sam_refs)

    ref_coco, unknown_labels = build_refmuseg_full(args.refmuseg_root, args.museg_root)
    valid_ann_ids = {int(a["id"]) for a in ref_coco["annotations"]}
    ref_pickle_path = Path(args.refmuseg_root) / "ref_museg(unc).p"
    existing_refs = build_refmuseg_existing_refs(
        ref_pickle_path,
        valid_ann_ids=valid_ann_ids,
        source_name="refmuseg_existing",
    )
    with open(ref_pickle_path, "rb") as f:
        refmuseg_raw_refs = pickle.load(f)
    split_by_image = {int(r["image_id"]): str(r.get("split", "train")) for r in refmuseg_raw_refs}
    generated_refs = build_generated_refs(
        ref_coco,
        split="train",
        source_name="refmuseg_generated",
        split_by_image=split_by_image,
    )
    ref_refs = merge_refs(existing_refs, generated_refs)
    write_dataset(
        output_root / "refmuseg_full_phrases",
        ref_coco,
        ref_refs,
        extra_stats={"unknown_labels_skipped": unknown_labels},
    )

    print("Wrote:", norm_path(output_root / "sam3_train_phrases"))
    print("Wrote:", norm_path(output_root / "refmuseg_full_phrases"))


if __name__ == "__main__":
    main()
