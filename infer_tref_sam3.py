#!/usr/bin/env python3
"""Single-image TRef-SAM3 inference.

Example:
  python infer_tref_sam3.py \
    --config configs/tref_sam3_config.yaml \
    --weights outputs/tref_sam3/best_lora_weights.pt \
    --image data/valid/xxx.jpg \
    --expression "left blue cable" \
    --output_dir outputs/tref_single
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image as PILImage
from PIL import ImageDraw
from torchvision.transforms import v2

from sam3.model.model_misc import SAM3Output
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
)
from validate_sam3_lora import attach_tref_matching_scores, move_to_device
from validate_tref_sam3 import build_model


def build_single_datapoint(image_path, expression, resolution=1008):
    pil_original = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = pil_original.size
    pil_resized = pil_original.resize((resolution, resolution), PILImage.BILINEAR)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image_tensor = transform(pil_resized)

    image_obj = Image(
        data=image_tensor,
        objects=[],
        size=(resolution, resolution),
    )
    query = FindQueryLoaded(
        query_text=str(expression).strip().lower(),
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0,
            original_image_id=0,
            original_category_id=-1,
            original_size=(orig_h, orig_w),
            object_id=-1,
            frame_index=-1,
        ),
    )
    return Datapoint(
        find_queries=[query],
        images=[image_obj],
        raw_images=[pil_resized],
    ), pil_original


def get_final_outputs(outputs_list):
    with SAM3Output.iteration_mode(
        outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
    ) as outputs_iter:
        final_stage = list(outputs_iter)[-1]
        return final_stage[-1]


def overlay_mask(image, mask, color=(255, 40, 40), alpha=0.45):
    base = image.convert("RGBA")
    mask_img = PILImage.fromarray((mask.astype("uint8") * 255), mode="L")
    color_img = PILImage.new("RGBA", base.size, (*color, 0))
    alpha_img = mask_img.point(lambda v: int(v * alpha))
    color_img.putalpha(alpha_img)
    return PILImage.alpha_composite(base, color_img).convert("RGB")


def infer(
    config_path,
    weights_path,
    image_path,
    expression,
    output_dir,
    mask_threshold=0.5,
    score_threshold=0.35,
    multi=False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, weights_path, device)
    tref_cfg = config.get("tref_matching", config["training"].get("tref_matching", {}))
    resolution = int(config["training"].get("referring", {}).get("resolution", 1008))

    datapoint, pil_original = build_single_datapoint(
        image_path=image_path,
        expression=expression,
        resolution=resolution,
    )
    batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=True)["input"]
    batch = move_to_device(batch, device)

    model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs_list = model(batch)
                attach_tref_matching_scores(model, outputs_list, batch, tref_cfg)
        else:
            outputs_list = model(batch)
            attach_tref_matching_scores(model, outputs_list, batch, tref_cfg)

    outputs = get_final_outputs(outputs_list)
    scores = outputs["pred_logits"][0].sigmoid().squeeze(-1)
    masks_prob = outputs["pred_masks"][0].sigmoid()

    if multi:
        keep = scores > float(score_threshold)
        if not keep.any():
            keep[scores.argmax()] = True
        selected_indices = torch.nonzero(keep, as_tuple=False).flatten()
        selected_score = scores[selected_indices].max()
        mask_prob = masks_prob[selected_indices].max(dim=0).values
    else:
        selected_indices = scores.argmax().view(1)
        selected_score = scores[selected_indices[0]]
        mask_prob = masks_prob[selected_indices[0]]

    orig_w, orig_h = pil_original.size
    mask_prob_resized = F.interpolate(
        mask_prob[None, None].float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    mask_np = (mask_prob_resized.detach().cpu().numpy() > float(mask_threshold))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)

    mask_path = output_dir / f"{safe_stem}_tref_mask.png"
    overlay_path = output_dir / f"{safe_stem}_tref_overlay.png"
    result_path = output_dir / f"{safe_stem}_tref_result.json"

    PILImage.fromarray((mask_np.astype("uint8") * 255), mode="L").save(mask_path)
    vis = overlay_mask(pil_original, mask_np)
    draw = ImageDraw.Draw(vis)
    label = f"{expression} | score={float(selected_score):.4f}"
    draw.rectangle((8, 8, min(vis.width - 8, 8 + len(label) * 7), 32), fill=(0, 0, 0))
    draw.text((12, 14), label, fill=(255, 255, 255))
    vis.save(overlay_path)

    result = {
        "image": str(image_path),
        "expression": str(expression),
        "selected_indices": [int(x) for x in selected_indices.detach().cpu().tolist()],
        "score": float(selected_score.detach().cpu().item()),
        "mask_threshold": float(mask_threshold),
        "score_threshold": float(score_threshold),
        "multi": bool(multi),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Single-image TRef-SAM3 inference.")
    parser.add_argument("--config", default="configs/tref_sam3_config.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--output_dir", default="outputs/tref_single")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.35)
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Return the union of all candidates above score_threshold instead of top-1.",
    )
    args = parser.parse_args()

    infer(
        config_path=args.config,
        weights_path=args.weights,
        image_path=args.image,
        expression=args.expression,
        output_dir=args.output_dir,
        mask_threshold=args.mask_threshold,
        score_threshold=args.score_threshold,
        multi=args.multi,
    )


if __name__ == "__main__":
    main()
