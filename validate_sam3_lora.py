#!/usr/bin/env python3
"""
Validation script for SAM3 LoRA model
Loads saved weights and runs validation with detailed debugging
"""

import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, count_parameters

from torchvision.transforms import v2

# Import evaluation modules
from sam3.eval.cgf1_eval import CGF1Evaluator, COCOCustom
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from sam3.train.masks_ops import rle_encode

# Import SAM3's NMS
from sam3.perflib.nms import nms_masks

class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""
    def __init__(self, data_dir, split="train"):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split

        # Load COCO annotations
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.split_dir / img_info['file_name']
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []
        object_category_ids = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = int(ann.get("category_id", 0))
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)
            object_category_ids.append(category_id)

            # Convert from COCO [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = bbox_coco
            box_tensor = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)

            # Scale box to resolution
            box_tensor[0] *= scale_w
            box_tensor[2] *= scale_w
            box_tensor[1] *= scale_h
            box_tensor[3] *= scale_h

            # IMPORTANT: Normalize boxes to [0, 1] range (required by SAM3 loss functions)
            box_tensor /= self.resolution

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    # Check if it's RLE format (dict) or polygon format (list)
                    if isinstance(segmentation, dict):
                        # RLE format: {"counts": "...", "size": [h, w]}
                        mask_np = mask_utils.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # Polygon format: [[x1, y1, x2, y2, ...], ...]
                        # Convert polygon to RLE, then decode
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                    else:
                        print(f"Warning: Unknown segmentation format: {type(segmentation)}")
                        segment = None
                        continue

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2]-box_tensor[0])*(box_tensor[3]-box_tensor[1]),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - one per unique category
        # Each query maps to only the objects of that category
        from collections import defaultdict

        # Group object IDs by category id (query text is category name)
        cat_to_object_ids = defaultdict(list)
        cat_to_query_text = {}
        for obj, cat_id, class_name in zip(objects, object_category_ids, object_class_names):
            cat_to_object_ids[cat_id].append(obj.object_id)
            if cat_id not in cat_to_query_text:
                cat_to_query_text[cat_id] = class_name.lower()

        # Create one query per category
        queries = []
        default_category_id = int(next(iter(self.categories.keys()), 1))
        if len(cat_to_object_ids) > 0:
            for category_id, obj_ids in cat_to_object_ids.items():
                query_text = cat_to_query_text.get(category_id, "object")
                query = FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=int(category_id),
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1
                    )
                )
                queries.append(query)
        else:
            # No annotations: create a single generic query
            query = FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=default_category_id,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.15):
    """
    Merge overlapping masks that likely represent the same object (e.g., crack segments).

    This is more aggressive than NMS - it MERGES masks instead of suppressing them.
    Useful for cracks where model splits one crack into many segments.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            # Compute IoU
            intersection = (current_mask & binary_masks[j]).sum().item()
            union = (current_mask | binary_masks[j]).sum().item()
            iou = intersection / union if union > 0 else 0

            # If overlaps significantly, merge it
            if iou > iou_threshold:
                current_mask = current_mask | binary_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def apply_sam3_nms(pred_logits, pred_masks, pred_boxes, prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100):
    """
    Apply SAM3's standard NMS pipeline to filter predictions.

    Args:
        pred_logits: [N, 1] logits
        pred_masks: [N, H, W] mask logits
        pred_boxes: [N, 4] boxes in normalized format
        prob_threshold: Score threshold for filtering (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: IoU threshold for NMS (default: 0.7, SAM3 uses 0.5-0.7)
        max_detections: Maximum detections to keep (default: 100)

    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_boxes)
    """
    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits).squeeze(-1)  # [N]

    # Convert mask logits to binary masks (sigmoid + threshold)
    pred_masks_sigmoid = torch.sigmoid(pred_masks)  # [N, H, W]
    pred_masks_binary = pred_masks_sigmoid > 0.5  # [N, H, W]

    # Apply SAM3's NMS
    # nms_masks expects: pred_probs [N], pred_masks [N, H, W], prob_threshold, iou_threshold
    # Returns: keep mask [N] of booleans
    keep_mask = nms_masks(
        pred_probs=pred_probs,
        pred_masks=pred_masks_binary.float(),  # NMS expects float masks
        prob_threshold=prob_threshold,
        iou_threshold=nms_iou_threshold
    )

    # Filter predictions
    filtered_masks = pred_masks_sigmoid[keep_mask]  # Keep sigmoid masks for later
    filtered_scores = pred_probs[keep_mask]
    filtered_boxes = pred_boxes[keep_mask]

    # Top-K selection by score
    if max_detections > 0 and len(filtered_scores) > max_detections:
        top_k_scores, top_k_indices = torch.topk(filtered_scores, k=max_detections, largest=True)
        filtered_masks = filtered_masks[top_k_indices]
        filtered_scores = top_k_scores
        filtered_boxes = filtered_boxes[top_k_indices]

    return filtered_masks, filtered_scores, filtered_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288,
                                       prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100,
                                       merge_cracks=False, merge_iou_threshold=0.15,
                                       verbose=True, category_ids=None):
    """
    Convert model predictions to COCO format using SAM3's NMS pipeline.

    Args:
        predictions_list: List of predictions per query-item
        image_ids: List of COCO image IDs aligned with predictions_list
        category_ids: List of COCO category IDs aligned with predictions_list
        resolution: Resolution for box scaling (default: 288)
        prob_threshold: Score threshold (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: NMS IoU threshold (default: 0.7)
        max_detections: Max detections per image (default: 100)
        merge_cracks: If True, merge overlapping segments instead of NMS suppression (default: False)
        merge_iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
    """
    if category_ids is None:
        category_ids = [1] * len(predictions_list)
    if len(category_ids) != len(predictions_list):
        raise ValueError(
            f"category_ids length ({len(category_ids)}) must match predictions_list length ({len(predictions_list)})"
        )

    coco_predictions = []
    pred_id = 0

    if verbose:
        if merge_cracks:
            print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
            print(f"[INFO] Using CRACK MERGING mode: prob_threshold={prob_threshold}, merge_iou={merge_iou_threshold}, max_dets={max_detections}")
            print(f"[INFO] This will MERGE overlapping crack segments instead of suppressing them")
        else:
            print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
            print(f"[INFO] Using SAM3 NMS: prob_threshold={prob_threshold}, nms_iou={nms_iou_threshold}, max_dets={max_detections}")

    pred_iter = zip(image_ids, predictions_list, category_ids)
    if verbose:
        pred_iter = tqdm(pred_iter, total=len(predictions_list), desc="Converting predictions")

    for img_id, preds, category_id in pred_iter:
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        logits = preds['pred_logits']  # [N, 1]
        boxes = preds['pred_boxes']    # [N, 4]
        masks = preds['pred_masks']    # [N, H, W]

        if merge_cracks:
            # Step 1: Filter by score threshold
            pred_probs = torch.sigmoid(logits).squeeze(-1)  # [N]
            valid_mask = pred_probs > prob_threshold

            filtered_masks = masks[valid_mask]
            filtered_scores = pred_probs[valid_mask]
            filtered_boxes = boxes[valid_mask]

            if len(filtered_masks) > 0:
                # Step 2: Convert masks to binary
                pred_masks_sigmoid = torch.sigmoid(filtered_masks)
                pred_masks_binary = (pred_masks_sigmoid > 0.5)

                # Step 3: MERGE overlapping crack segments
                merged_masks, merged_scores, merged_boxes = merge_overlapping_masks(
                    pred_masks_binary.cpu(),
                    filtered_scores.cpu(),
                    filtered_boxes.cpu(),
                    iou_threshold=merge_iou_threshold
                )

                # Step 4: Top-K selection by score
                if max_detections > 0 and len(merged_scores) > max_detections:
                    top_k_scores, top_k_indices = torch.topk(merged_scores, k=max_detections, largest=True)
                    merged_masks = merged_masks[top_k_indices]
                    merged_scores = top_k_scores
                    merged_boxes = merged_boxes[top_k_indices]

                # Return merged results (already binary)
                filtered_masks = merged_masks.float()  # Already binary, just convert to float
                filtered_scores = merged_scores
                filtered_boxes = merged_boxes
            else:
                filtered_masks = torch.tensor([])
                filtered_scores = torch.tensor([])
                filtered_boxes = torch.tensor([])
        else:
            # Apply SAM3's NMS pipeline (standard suppression)
            filtered_masks, filtered_scores, filtered_boxes = apply_sam3_nms(
                pred_logits=logits,
                pred_masks=masks,
                pred_boxes=boxes,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections
            )

        if len(filtered_masks) > 0:
            # Convert filtered masks to binary for RLE encoding
            binary_masks = (filtered_masks > 0.5).cpu()
            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, filtered_scores.cpu().tolist(), filtered_boxes.cpu().tolist())):
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': int(category_id),
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from dataset.

    OPTIMIZATION: Downsample GT masks to 288×288 to match prediction resolution.
    """
    print(f"\n[INFO] Creating COCO ground truth (downsampling to {mask_resolution}×{mask_resolution})...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': dataset.coco_data.get('categories', [{'id': 1, 'name': 'object'}])
    }

    ann_id = 0
    coco_id_to_index = {int(coco_id): idx for idx, coco_id in enumerate(dataset.image_ids)}

    resolved_items = []
    if image_ids is None:
        resolved_items = [
            (int(coco_id), idx) for idx, coco_id in enumerate(dataset.image_ids)
        ]
    else:
        skipped = 0
        seen = set()
        for raw_id in image_ids:
            candidate = int(raw_id)
            if candidate in seen:
                continue
            seen.add(candidate)

            if candidate in coco_id_to_index:
                resolved_items.append((candidate, coco_id_to_index[candidate]))
            elif 0 <= candidate < len(dataset):
                # Backward compatibility: allow passing dataset indices.
                coco_image_id = int(dataset.image_ids[candidate])
                resolved_items.append((coco_image_id, candidate))
            else:
                skipped += 1
        if skipped > 0:
            print(f"[WARN] Skipped {skipped} image ids not found in dataset.")

    for coco_image_id, dataset_idx in tqdm(resolved_items, desc="Creating GT"):
        source_annotations = dataset.img_to_anns.get(coco_image_id, [])
        coco_gt['images'].append({
            'id': int(coco_image_id),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True
        })

        datapoint = dataset[dataset_idx]

        for obj in datapoint.images[0].objects:
            # Scale boxes to mask_resolution
            box = obj.bbox * mask_resolution
            x1, y1, x2, y2 = box.tolist()
            x, y, w, h = x1, y1, x2-x1, y2-y1

            source_ann_idx = int(getattr(obj, "object_id", -1))
            if 0 <= source_ann_idx < len(source_annotations):
                source_category_id = int(source_annotations[source_ann_idx].get('category_id', 1))
            else:
                source_category_id = 1

            ann = {
                'id': ann_id,
                'image_id': int(coco_image_id),
                'category_id': source_category_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")

    return coco_gt


def compute_multiclass_semantic_metrics(coco_gt_dict, coco_predictions):
    """
    Compute semantic segmentation metrics from instance-style COCO masks.

    We collapse instances into a per-pixel semantic label map:
    - GT: later annotations overwrite earlier ones if overlaps exist.
    - Pred: the category of the highest-score mask wins for each pixel.

    Returns:
        Dict with multi-class semantic mIoU and pixel-level F1 summaries.
    """
    categories = coco_gt_dict.get("categories", [])
    sorted_cat_ids = sorted(int(cat["id"]) for cat in categories)
    if not sorted_cat_ids:
        return {
            "semantic_mIoU": 0.0,
            "semantic_macro_F1": 0.0,
            "semantic_micro_F1": 0.0,
            "semantic_pixel_accuracy": 0.0,
            "semantic_num_classes": 0,
        }

    cat_id_to_index = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted_cat_ids)}
    num_labels = len(sorted_cat_ids) + 1  # background = 0

    images_by_id = {int(img["id"]): img for img in coco_gt_dict.get("images", [])}

    gt_by_image = defaultdict(list)
    for ann in coco_gt_dict.get("annotations", []):
        gt_by_image[int(ann["image_id"])].append(ann)

    pred_by_image = defaultdict(list)
    for ann in coco_predictions:
        pred_by_image[int(ann["image_id"])].append(ann)

    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)

    for image_id, image_info in images_by_id.items():
        height = int(image_info["height"])
        width = int(image_info["width"])

        gt_semantic = np.zeros((height, width), dtype=np.int32)
        for ann in gt_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            if cat_id not in cat_id_to_index:
                continue
            if "segmentation" not in ann:
                continue
            mask = mask_utils.decode(ann["segmentation"])
            if mask.ndim == 3:
                mask = mask[..., 0]
            gt_semantic[mask.astype(bool)] = cat_id_to_index[cat_id]

        pred_semantic = np.zeros((height, width), dtype=np.int32)
        pred_score_map = np.zeros((height, width), dtype=np.float32)
        for ann in pred_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            if cat_id not in cat_id_to_index:
                continue
            if "segmentation" not in ann:
                continue
            mask = mask_utils.decode(ann["segmentation"])
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = mask.astype(bool)
            if not np.any(mask):
                continue
            score = float(ann.get("score", 0.0))
            overwrite = mask & (score >= pred_score_map)
            pred_semantic[overwrite] = cat_id_to_index[cat_id]
            pred_score_map[overwrite] = score

        combined = num_labels * gt_semantic.reshape(-1) + pred_semantic.reshape(-1)
        confusion += np.bincount(
            combined,
            minlength=num_labels * num_labels,
        ).reshape(num_labels, num_labels)

    tp = np.diag(confusion).astype(np.float64)
    gt_pixels = confusion.sum(axis=1).astype(np.float64)
    pred_pixels = confusion.sum(axis=0).astype(np.float64)

    fg_tp = tp[1:]
    fg_gt = gt_pixels[1:]
    fg_pred = pred_pixels[1:]

    class_union = fg_gt + fg_pred - fg_tp
    class_iou = np.divide(
        fg_tp,
        class_union,
        out=np.zeros_like(fg_tp),
        where=class_union > 0,
    )
    class_f1 = np.divide(
        2.0 * fg_tp,
        fg_gt + fg_pred,
        out=np.zeros_like(fg_tp),
        where=(fg_gt + fg_pred) > 0,
    )

    valid_iou = class_union > 0
    valid_f1 = (fg_gt + fg_pred) > 0

    total_fg_tp = float(fg_tp.sum())
    total_fg_fp = float((fg_pred - fg_tp).sum())
    total_fg_fn = float((fg_gt - fg_tp).sum())
    semantic_micro_f1 = (
        (2.0 * total_fg_tp) / (2.0 * total_fg_tp + total_fg_fp + total_fg_fn + 1e-6)
    )

    total_pixels = float(confusion.sum())
    semantic_pixel_accuracy = float(tp.sum()) / (total_pixels + 1e-6)

    return {
        "semantic_mIoU": float(class_iou[valid_iou].mean()) if np.any(valid_iou) else 0.0,
        "semantic_macro_F1": float(class_f1[valid_f1].mean()) if np.any(valid_f1) else 0.0,
        "semantic_micro_F1": float(semantic_micro_f1),
        "semantic_pixel_accuracy": float(semantic_pixel_accuracy),
        "semantic_num_classes": len(sorted_cat_ids),
    }


def compute_multiclass_semantic_metrics_from_prob_maps(
    coco_gt_dict, semantic_prob_maps, threshold=0.5
):
    """
    Compute semantic metrics from the semantic head's dense probability maps.

    `semantic_prob_maps` is expected to be:
      image_id -> {category_id -> probability_map(H, W)}
    """
    categories = coco_gt_dict.get("categories", [])
    sorted_cat_ids = sorted(int(cat["id"]) for cat in categories)
    if not sorted_cat_ids:
        return {
            "semantic_mIoU": 0.0,
            "semantic_macro_F1": 0.0,
            "semantic_micro_F1": 0.0,
            "semantic_pixel_accuracy": 0.0,
            "semantic_num_classes": 0,
        }

    cat_id_to_index = {cat_id: idx + 1 for idx, cat_id in enumerate(sorted_cat_ids)}
    num_labels = len(sorted_cat_ids) + 1  # background = 0

    images_by_id = {int(img["id"]): img for img in coco_gt_dict.get("images", [])}

    gt_by_image = defaultdict(list)
    for ann in coco_gt_dict.get("annotations", []):
        gt_by_image[int(ann["image_id"])].append(ann)

    confusion = np.zeros((num_labels, num_labels), dtype=np.int64)

    for image_id, image_info in images_by_id.items():
        height = int(image_info["height"])
        width = int(image_info["width"])

        gt_semantic = np.zeros((height, width), dtype=np.int32)
        for ann in gt_by_image.get(image_id, []):
            cat_id = int(ann["category_id"])
            if cat_id not in cat_id_to_index or "segmentation" not in ann:
                continue
            mask = mask_utils.decode(ann["segmentation"])
            if mask.ndim == 3:
                mask = mask[..., 0]
            gt_semantic[mask.astype(bool)] = cat_id_to_index[cat_id]

        pred_semantic = np.zeros((height, width), dtype=np.int32)
        pred_score_map = np.zeros((height, width), dtype=np.float32)
        image_prob_maps = semantic_prob_maps.get(image_id, {})
        for cat_id, prob_map in image_prob_maps.items():
            cat_id = int(cat_id)
            if cat_id not in cat_id_to_index:
                continue

            prob_map = np.asarray(prob_map, dtype=np.float32)
            if prob_map.shape != (height, width):
                prob_tensor = torch.from_numpy(prob_map).float().unsqueeze(0).unsqueeze(0)
                prob_map = (
                    F.interpolate(
                        prob_tensor,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            overwrite = prob_map >= pred_score_map
            pred_semantic[overwrite] = cat_id_to_index[cat_id]
            pred_score_map[overwrite] = prob_map[overwrite]

        pred_semantic[pred_score_map < float(threshold)] = 0

        combined = num_labels * gt_semantic.reshape(-1) + pred_semantic.reshape(-1)
        confusion += np.bincount(
            combined,
            minlength=num_labels * num_labels,
        ).reshape(num_labels, num_labels)

    tp = np.diag(confusion).astype(np.float64)
    gt_pixels = confusion.sum(axis=1).astype(np.float64)
    pred_pixels = confusion.sum(axis=0).astype(np.float64)

    fg_tp = tp[1:]
    fg_gt = gt_pixels[1:]
    fg_pred = pred_pixels[1:]

    class_union = fg_gt + fg_pred - fg_tp
    class_iou = np.divide(
        fg_tp,
        class_union,
        out=np.zeros_like(fg_tp),
        where=class_union > 0,
    )
    class_f1 = np.divide(
        2.0 * fg_tp,
        fg_gt + fg_pred,
        out=np.zeros_like(fg_tp),
        where=(fg_gt + fg_pred) > 0,
    )

    valid_iou = class_union > 0
    valid_f1 = (fg_gt + fg_pred) > 0

    total_fg_tp = float(fg_tp.sum())
    total_fg_fp = float((fg_pred - fg_tp).sum())
    total_fg_fn = float((fg_gt - fg_tp).sum())
    semantic_micro_f1 = (
        (2.0 * total_fg_tp) / (2.0 * total_fg_tp + total_fg_fp + total_fg_fn + 1e-6)
    )

    total_pixels = float(confusion.sum())
    semantic_pixel_accuracy = float(tp.sum()) / (total_pixels + 1e-6)

    return {
        "semantic_mIoU": float(class_iou[valid_iou].mean()) if np.any(valid_iou) else 0.0,
        "semantic_macro_F1": float(class_f1[valid_f1].mean()) if np.any(valid_f1) else 0.0,
        "semantic_micro_F1": float(semantic_micro_f1),
        "semantic_pixel_accuracy": float(semantic_pixel_accuracy),
        "semantic_num_classes": len(sorted_cat_ids),
    }


def run_coco_eval(coco_gt, coco_dt, iou_type):
    """Run COCO evaluation for a specific IoU type and print the summary."""
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.params.useCats = True
            coco_eval.evaluate()
            coco_eval.accumulate()

    coco_eval.summarize()
    return coco_eval


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset,
                                                    model_resolution=288,
                                                    prob_threshold=0.3,
                                                    nms_iou_threshold=0.7,
                                                    max_detections=100,
                                                    merge_cracks=False,
                                                    merge_iou_threshold=0.15,
                                                    debug=False,
                                                    category_ids=None):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    Args:
        predictions_list: List of predictions per query-item
        image_ids: List of COCO image IDs aligned with predictions_list
        dataset: Dataset that provides original image sizes via dataset.images
        category_ids: List of COCO category IDs aligned with predictions_list
        model_resolution: Model output resolution (default: 288)
        prob_threshold: Confidence threshold for filtering predictions
        nms_iou_threshold: NMS IoU threshold
        max_detections: Maximum detections per image
        merge_cracks: Whether to merge overlapping predictions instead of NMS
        merge_iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    del model_resolution
    if category_ids is None:
        category_ids = [1] * len(predictions_list)
    if len(category_ids) != len(predictions_list):
        raise ValueError(
            f"category_ids length ({len(category_ids)}) must match predictions_list length ({len(predictions_list)})"
        )

    coco_predictions = []
    pred_id = 0
    coco_id_to_image = {int(coco_id): img for coco_id, img in dataset.images.items()}

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_cracks:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={merge_iou_threshold})")

    for img_id, preds, category_id in zip(image_ids, predictions_list, category_ids):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        coco_image_id = int(img_id)
        image_info = coco_id_to_image.get(coco_image_id)
        if image_info is None:
            if debug:
                print(f"[DEBUG] Skipping unknown COCO image id: {coco_image_id}")
            continue
        orig_w = int(image_info["width"])
        orig_h = int(image_info["height"])

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, H, W]

        if merge_cracks:
            pred_probs = torch.sigmoid(logits).squeeze(-1)
            valid_mask = pred_probs > prob_threshold
            filtered_masks = masks[valid_mask]
            filtered_scores = pred_probs[valid_mask]
            filtered_boxes = boxes[valid_mask]

            if len(filtered_masks) == 0:
                continue

            pred_masks_sigmoid = torch.sigmoid(filtered_masks)
            masks_upsampled = torch.nn.functional.interpolate(
                pred_masks_sigmoid.unsqueeze(1).float(),
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            pred_masks_binary = masks_upsampled > 0.5

            merged_masks, merged_scores, merged_boxes = merge_overlapping_masks(
                pred_masks_binary.cpu(),
                filtered_scores.cpu(),
                filtered_boxes.cpu(),
                iou_threshold=merge_iou_threshold
            )

            if max_detections > 0 and len(merged_scores) > max_detections:
                top_k_scores, top_k_indices = torch.topk(
                    merged_scores, k=max_detections, largest=True
                )
                merged_scores = top_k_scores
                merged_boxes = merged_boxes[top_k_indices]

            filtered_scores = merged_scores
            filtered_boxes = merged_boxes
        else:
            _, filtered_scores, filtered_boxes = apply_sam3_nms(
                pred_logits=logits,
                pred_masks=masks,
                pred_boxes=boxes,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections
            )

        if debug and img_id == image_ids[0] and len(filtered_scores) > 0:
            print(f"[DEBUG] Image {img_id}: original size={orig_w}x{orig_h}")
            print(
                f"[DEBUG]   Filtered scores: min={filtered_scores.min():.4f}, "
                f"max={filtered_scores.max():.4f}, mean={filtered_scores.mean():.4f}"
            )

        for idx, (score, box) in enumerate(zip(filtered_scores.cpu().tolist(), filtered_boxes.cpu().tolist())):
            cx, cy, w_norm, h_norm = box
            x = (cx - w_norm / 2.0) * orig_w
            y = (cy - h_norm / 2.0) * orig_h
            w = w_norm * orig_w
            h = h_norm * orig_h

            x = max(0.0, min(x, float(orig_w)))
            y = max(0.0, min(y, float(orig_h)))
            w = max(0.0, min(w, float(orig_w) - x))
            h = max(0.0, min(h, float(orig_h) - y))

            if w < 1 or h < 1:
                continue

            pred_dict = {
                'image_id': coco_image_id,
                'category_id': int(category_id),
                'bbox': [float(x), float(y), float(w), float(h)],
                'score': float(score),
                'id': pred_id
            }

            if debug and img_id == image_ids[0] and idx == 0:
                print(f"[DEBUG]   First original-res bbox: {pred_dict['bbox']}")

            coco_predictions.append(pred_dict)
            pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': dataset.coco_data.get('categories', [{'id': 1, 'name': 'object'}])
    }

    ann_id = 0
    coco_id_to_index = {int(coco_id): idx for idx, coco_id in enumerate(dataset.image_ids)}
    resolved_items = []
    if image_ids is None:
        resolved_items = [
            (int(coco_id), idx) for idx, coco_id in enumerate(dataset.image_ids)
        ]
    else:
        seen = set()
        for raw_id in image_ids:
            candidate = int(raw_id)
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in coco_id_to_index:
                resolved_items.append((candidate, coco_id_to_index[candidate]))
            elif 0 <= candidate < len(dataset):
                resolved_items.append((int(dataset.image_ids[candidate]), candidate))

    for coco_image_id, dataset_idx in resolved_items:
        del dataset_idx
        image_info = dataset.images.get(int(coco_image_id), dataset.images.get(str(coco_image_id)))
        if image_info is None:
            continue
        source_annotations = dataset.img_to_anns.get(
            int(coco_image_id),
            dataset.img_to_anns.get(str(coco_image_id), []),
        )
        orig_w = int(image_info["width"])
        orig_h = int(image_info["height"])

        coco_gt['images'].append({
            'id': int(coco_image_id),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for source_ann in source_annotations:
            x, y, w, h = source_ann.get('bbox', [0, 0, 0, 0])
            ann = {
                'id': ann_id,
                'image_id': int(coco_image_id),
                'category_id': int(source_ann.get('category_id', 1)),
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': int(source_ann.get('iscrowd', 0)),
                'ignore': int(source_ann.get('ignore', 0))
            }

            if 'segmentation' in source_ann:
                ann['segmentation'] = source_ann['segmentation']

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


def move_to_device(obj, device):
    """Recursively move objects to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            val = getattr(obj, field)
            setattr(obj, field, move_to_device(val, device))
        return obj
    return obj


def validate(config_path, weights_path, val_data_dir, num_samples=None,
             prob_threshold=0.3, nms_iou=0.7, merge_cracks=False, merge_iou=0.15,
             use_base_model=False):
    """Run validation with full metrics (mAP, cgF1) and SAM3 NMS

    Args:
        config_path: Path to config file (for LoRA settings only). Not required if use_base_model=True.
        weights_path: Path to LoRA weights. Not required if use_base_model=True.
        val_data_dir: Direct path to validation data directory containing _annotations.coco.json
                      (e.g., /workspace/data2/valid)
        num_samples: Optional limit for number of samples (for debugging)
        use_base_model: If True, use original SAM3 model without LoRA (default: False)

    Example (with LoRA):
        validate(
            config_path="configs/full_lora_config.yaml",
            weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
            val_data_dir="/workspace/data2/valid"
        )

    Example (base SAM3 model):
        validate(
            config_path=None,
            weights_path=None,
            val_data_dir="/workspace/data2/valid",
            use_base_model=True
        )
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = None

    # Load config for batch_size and other settings
    if use_base_model:
        # Use original SAM3 model without LoRA
        print("Using original SAM3 model (no LoRA)")
        # Use default batch_size for base model
        batch_size = 1
    else:
        # Apply LoRA and load weights
        if config_path is None or weights_path is None:
            raise ValueError("config_path and weights_path are required when use_base_model=False")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Get batch_size from config
        batch_size = config["training"]["batch_size"]

    model_cfg = {} if config is None else config.get("model", {})
    srf_cfg = model_cfg.get("srf_lite", {})

    # Build model
    print("\nBuilding SAM3 model...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False,
        use_srf_lite=bool(srf_cfg.get("enabled", False)),
        srf_num_levels=int(srf_cfg.get("num_levels", 4)),
        srf_bottleneck_dim=srf_cfg.get("bottleneck_dim", None),
        srf_interpolation_mode=str(srf_cfg.get("interpolation_mode", "bilinear")),
        srf_alpha_init=float(srf_cfg.get("alpha_init", 0.0)),
    )

    if use_base_model:
        stats = count_parameters(model)
        print(f"Total params: {stats['total_parameters']:,}")
    else:
        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)

        # Load weights
        print(f"\nLoading LoRA weights from {weights_path}...")
        load_lora_weights(model, weights_path)

        stats = count_parameters(model)
        print(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

    model.to(device)
    model.eval()

    # Load validation data directly from the specified directory
    print(f"\nLoading validation data from {val_data_dir}...")

    # Load COCO annotations directly
    from pathlib import Path
    ann_file = Path(val_data_dir) / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    # Create a simple dataset class that loads from the directory directly
    class DirectCOCODataset(COCOSegmentDataset):
        def __init__(self, data_dir):
            self.data_dir = Path(data_dir)
            self.split_dir = self.data_dir

            # Load COCO annotations
            ann_file = self.split_dir / "_annotations.coco.json"
            if not ann_file.exists():
                raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

            with open(ann_file, 'r') as f:
                self.coco_data = json.load(f)

            # Build index: image_id -> image info
            self.images = {img['id']: img for img in self.coco_data['images']}
            self.image_ids = sorted(list(self.images.keys()))

            # Build index: image_id -> list of annotations
            self.img_to_anns = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)

            # Load categories
            self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
            print(f"Loaded COCO dataset from {data_dir}")
            print(f"  Images: {len(self.image_ids)}")
            print(f"  Annotations: {len(self.coco_data['annotations'])}")
            print(f"  Categories: {self.categories}")

            self.resolution = 1008
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    val_ds = DirectCOCODataset(val_data_dir)

    if num_samples:
        print(f"\n[INFO] Limiting validation to {num_samples} samples for debugging")

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,  # Enable parallel data loading
        pin_memory=True  # Faster GPU transfer
    )

    # Create matcher for loss computation
    matcher = BinaryHungarianMatcherV2(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
    )

    # Run validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)

    metrics_result = {
        "use_base_model": bool(use_base_model),
        "config_path": config_path,
        "weights_path": weights_path,
        "val_data_dir": val_data_dir,
        "num_samples": num_samples,
        "prob_threshold": float(prob_threshold),
        "nms_iou": float(nms_iou),
        "merge_cracks": bool(merge_cracks),
        "merge_iou": float(merge_iou),
        "num_query_items": 0,
        "num_unique_images": 0,
        "num_unique_categories": 0,
        "num_predictions": 0,
        "num_predictions_288": 0,
        "num_predictions_original_res": 0,
        "has_gt_segmentation": None,
        "mAP": None,
        "mAP50": None,
        "mAP75": None,
        "APmask": None,
        "APmask50": None,
        "APmask75": None,
        "APbbox": None,
        "APbbox50": None,
        "APbbox75": None,
        "APbbox_288": None,
        "APbbox50_288": None,
        "APbbox75_288": None,
        "APbbox_original_res": None,
        "APbbox50_original_res": None,
        "APbbox75_original_res": None,
        "cgF1": None,
        "cgF1_50": None,
        "cgF1_75": None,
        "semantic_mIoU": None,
        "semantic_macro_F1": None,
        "semantic_micro_F1": None,
        "semantic_pixel_accuracy": None,
        "semantic_head_mIoU": None,
        "semantic_head_macro_F1": None,
        "semantic_head_micro_F1": None,
        "semantic_head_pixel_accuracy": None,
    }

    all_image_ids = []
    all_category_ids = []
    coco_predictions = []
    coco_predictions_bbox = []
    semantic_head_prob_maps = defaultdict(dict)

    # Use automatic mixed precision for faster inference
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(val_loader, desc="Validation")):
            if num_samples and batch_idx * batch_size >= num_samples:
                break

            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)

            # Forward pass with optional AMP
            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs_list = model(input_batch)
            else:
                outputs_list = model(input_batch)

            # Extract predictions
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]

                batch_size_actual = final_outputs['pred_logits'].shape[0]
                query_coco_image_ids = None
                query_category_ids = None
                if hasattr(input_batch, "find_metadatas") and len(input_batch.find_metadatas) > 0:
                    stage_meta = input_batch.find_metadatas[0]
                    if hasattr(stage_meta, "coco_image_id"):
                        meta_ids = stage_meta.coco_image_id
                        if isinstance(meta_ids, torch.Tensor):
                            query_coco_image_ids = meta_ids.detach().cpu().view(-1).tolist()
                        else:
                            query_coco_image_ids = list(meta_ids)
                    if hasattr(stage_meta, "original_category_id"):
                        meta_cats = stage_meta.original_category_id
                        if isinstance(meta_cats, torch.Tensor):
                            query_category_ids = meta_cats.detach().cpu().view(-1).tolist()
                        else:
                            query_category_ids = list(meta_cats)

                batch_predictions = []
                batch_image_ids = []
                batch_category_ids = []
                for i in range(batch_size_actual):
                    if query_coco_image_ids is not None and i < len(query_coco_image_ids):
                        img_id = int(query_coco_image_ids[i])
                    else:
                        # Fallback to sequential indexing when metadata is missing.
                        img_id = batch_idx * batch_size + i
                    if query_category_ids is not None and i < len(query_category_ids):
                        category_id = int(query_category_ids[i])
                    else:
                        category_id = 1
                    all_image_ids.append(img_id)
                    all_category_ids.append(category_id)
                    batch_image_ids.append(img_id)
                    batch_category_ids.append(category_id)
                    batch_predictions.append({
                        'pred_logits': final_outputs['pred_logits'][i].detach().cpu(),
                        'pred_boxes': final_outputs['pred_boxes'][i].detach().cpu(),
                        'pred_masks': final_outputs['pred_masks'][i].detach().cpu()
                    })

                    if "semantic_seg" in final_outputs:
                        semantic_prob = (
                            torch.sigmoid(final_outputs["semantic_seg"][i])
                            .squeeze()
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )
                        cached_prob = semantic_head_prob_maps[img_id].get(category_id)
                        if cached_prob is None:
                            semantic_head_prob_maps[img_id][category_id] = semantic_prob
                        else:
                            semantic_head_prob_maps[img_id][category_id] = np.maximum(
                                cached_prob, semantic_prob
                            )
                batch_coco_predictions = convert_predictions_to_coco_format(
                    batch_predictions,
                    batch_image_ids,
                    category_ids=batch_category_ids,
                    resolution=288,
                    prob_threshold=prob_threshold,
                    nms_iou_threshold=nms_iou,
                    max_detections=100,
                    merge_cracks=merge_cracks,
                    merge_iou_threshold=merge_iou,
                    verbose=False,
                )
                batch_coco_predictions_bbox = convert_predictions_to_coco_format_original_res(
                    batch_predictions,
                    batch_image_ids,
                    dataset=val_ds,
                    category_ids=batch_category_ids,
                    model_resolution=288,
                    prob_threshold=prob_threshold,
                    nms_iou_threshold=nms_iou,
                    max_detections=100,
                    merge_cracks=merge_cracks,
                    merge_iou_threshold=merge_iou,
                    debug=False,
                )
                coco_predictions.extend(batch_coco_predictions)
                coco_predictions_bbox.extend(batch_coco_predictions_bbox)

                # Free batch tensors early to avoid host memory accumulation
                del batch_predictions, batch_image_ids, batch_category_ids
                del batch_coco_predictions, batch_coco_predictions_bbox
                del final_outputs
                del final_stage
                del outputs_list

            if device.type == "cuda" and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()

    unique_image_ids = list(dict.fromkeys(int(x) for x in all_image_ids))
    unique_category_ids = sorted(set(int(x) for x in all_category_ids))
    print(
        f"\nCollected predictions for {len(all_image_ids)} query-items "
        f"across {len(unique_image_ids)} unique images and {len(unique_category_ids)} categories"
    )
    metrics_result["num_query_items"] = len(all_image_ids)
    metrics_result["num_unique_images"] = len(unique_image_ids)
    metrics_result["num_unique_categories"] = len(unique_category_ids)

    # Compute metrics
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    # Create COCO ground truth (downsampled to 288×288 - fast!)
    print(f"\n[INFO] Creating ground truth from validation dataset...")
    coco_gt_dict = create_coco_gt_from_dataset(
        val_ds,
        image_ids=unique_image_ids,
        mask_resolution=288
    )
    gt_annotations = coco_gt_dict.get("annotations", [])
    gt_annotations_with_seg = sum(1 for ann in gt_annotations if "segmentation" in ann)
    has_full_gt_segmentation = (
        len(gt_annotations) > 0 and gt_annotations_with_seg == len(gt_annotations)
    )
    metrics_result["has_gt_segmentation"] = bool(has_full_gt_segmentation)

    # Check prediction scores (optional - can be commented out for speed)
    # print(f"\n[INFO] Analyzing prediction scores...")
    # all_scores = []
    # for p in all_predictions:
    #     if 'pred_logits' in p and len(p['pred_logits']) > 0:
    #         scores = torch.sigmoid(p['pred_logits']).squeeze(-1)
    #         all_scores.extend(scores.tolist())
    # if all_scores:
    #     print(f"[INFO] Prediction scores: min={min(all_scores):.4f}, max={max(all_scores):.4f}, mean={np.mean(all_scores):.4f}")

    if merge_cracks:
        print(f"\n[INFO] Total predictions after CRACK MERGING (288 eval): {len(coco_predictions)}")
        print(f"[INFO] Total predictions after CRACK MERGING (original-res bbox eval): {len(coco_predictions_bbox)}")
    else:
        print(f"\n[INFO] Total predictions after SAM3 NMS filtering (288 eval): {len(coco_predictions)}")
        print(f"[INFO] Total predictions after SAM3 NMS filtering (original-res bbox eval): {len(coco_predictions_bbox)}")
    metrics_result["num_predictions"] = len(coco_predictions_bbox)
    metrics_result["num_predictions_288"] = len(coco_predictions)
    metrics_result["num_predictions_original_res"] = len(coco_predictions_bbox)

    if len(coco_predictions) > 0:
        # Save temporary files for COCO evaluation
        import tempfile
        import os

        # Create temp directory for evaluation files
        temp_dir = tempfile.mkdtemp(prefix="sam3_eval_")
        gt_file = os.path.join(temp_dir, "gt.json")
        pred_file = os.path.join(temp_dir, "pred.json")
        gt_bbox_file = os.path.join(temp_dir, "gt_bbox_original_res.json")
        pred_bbox_file = os.path.join(temp_dir, "pred_bbox_original_res.json")

        coco_gt_bbox_dict = create_coco_gt_from_dataset_original_res(
            val_ds,
            image_ids=unique_image_ids,
            debug=False,
        )

        with open(gt_file, 'w') as f:
            json.dump(coco_gt_dict, f)
        with open(pred_file, 'w') as f:
            json.dump(coco_predictions, f)
        with open(gt_bbox_file, 'w') as f:
            json.dump(coco_gt_bbox_dict, f)
        with open(pred_bbox_file, 'w') as f:
            json.dump(coco_predictions_bbox, f)

        # Compute mAP
        print("\n" + "="*80)
        print("COCO mAP EVALUATION")
        print("="*80)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_gt = COCO(str(gt_file))
                coco_dt = coco_gt.loadRes(str(pred_file))
                coco_gt_bbox = COCO(str(gt_bbox_file))
                coco_dt_bbox = (
                    coco_gt_bbox.loadRes(str(pred_bbox_file))
                    if len(coco_predictions_bbox) > 0
                    else None
                )

        print("\nBBox AP (bbox, 288 resolution):")
        coco_eval_bbox = run_coco_eval(coco_gt, coco_dt, 'bbox')
        map_bbox_288 = coco_eval_bbox.stats[0]
        map50_bbox_288 = coco_eval_bbox.stats[1]
        map75_bbox_288 = coco_eval_bbox.stats[2]
        metrics_result["APbbox_288"] = float(map_bbox_288)
        metrics_result["APbbox50_288"] = float(map50_bbox_288)
        metrics_result["APbbox75_288"] = float(map75_bbox_288)

        map_bbox = map_bbox_288
        map50_bbox = map50_bbox_288
        map75_bbox = map75_bbox_288
        has_original_bbox_eval = False
        if coco_dt_bbox is not None:
            print("\nBBox AP (bbox, original resolution):")
            coco_eval_bbox_original = run_coco_eval(coco_gt_bbox, coco_dt_bbox, 'bbox')
            map_bbox = coco_eval_bbox_original.stats[0]
            map50_bbox = coco_eval_bbox_original.stats[1]
            map75_bbox = coco_eval_bbox_original.stats[2]
            has_original_bbox_eval = True
            metrics_result["APbbox_original_res"] = float(map_bbox)
            metrics_result["APbbox50_original_res"] = float(map50_bbox)
            metrics_result["APbbox75_original_res"] = float(map75_bbox)
        else:
            print("\n[WARN] No original-res bbox predictions generated; APbbox uses 288-resolution fallback.")

        metrics_result["APbbox"] = float(map_bbox)
        metrics_result["APbbox50"] = float(map50_bbox)
        metrics_result["APbbox75"] = float(map75_bbox)
        if merge_cracks:
            print("[WARN] merge_cracks=True keeps merged masks but does not recompute merged boxes; APbbox may be less reliable.")

        if has_full_gt_segmentation:
            print("\nMask AP (segm):")
            coco_eval_segm = run_coco_eval(coco_gt, coco_dt, 'segm')
            map_segm = coco_eval_segm.stats[0]
            map50_segm = coco_eval_segm.stats[1]
            map75_segm = coco_eval_segm.stats[2]
            metrics_result["mAP"] = float(map_segm)
            metrics_result["mAP50"] = float(map50_segm)
            metrics_result["mAP75"] = float(map75_segm)
            metrics_result["APmask"] = float(map_segm)
            metrics_result["APmask50"] = float(map50_segm)
            metrics_result["APmask75"] = float(map75_segm)

            # Compute cgF1
            print("\n" + "="*80)
            print("cgF1 EVALUATION")
            print("="*80)

            cgf1_evaluator = CGF1Evaluator(
                gt_path=str(gt_file),
                iou_type='segm',
                verbose=True,
                use_cats=True,
            )
            cgf1_results = cgf1_evaluator.evaluate(str(pred_file))

            cgf1 = cgf1_results.get('cgF1_eval_segm_cgF1', 0.0)
            cgf1_50 = cgf1_results.get('cgF1_eval_segm_cgF1@0.5', 0.0)
            cgf1_75 = cgf1_results.get('cgF1_eval_segm_cgF1@0.75', 0.0)
            metrics_result["cgF1"] = float(cgf1)
            metrics_result["cgF1_50"] = float(cgf1_50)
            metrics_result["cgF1_75"] = float(cgf1_75)

            # Compute semantic segmentation metrics from the same masks
            print("\n" + "="*80)
            print("SEMANTIC mIoU / F1 EVALUATION")
            print("="*80)
            semantic_metrics = compute_multiclass_semantic_metrics(
                coco_gt_dict=coco_gt_dict,
                coco_predictions=coco_predictions,
            )
            semantic_miou = semantic_metrics["semantic_mIoU"]
            semantic_macro_f1 = semantic_metrics["semantic_macro_F1"]
            semantic_micro_f1 = semantic_metrics["semantic_micro_F1"]
            semantic_pixel_acc = semantic_metrics["semantic_pixel_accuracy"]
            semantic_num_classes = semantic_metrics["semantic_num_classes"]
            metrics_result["semantic_mIoU"] = float(semantic_miou)
            metrics_result["semantic_macro_F1"] = float(semantic_macro_f1)
            metrics_result["semantic_micro_F1"] = float(semantic_micro_f1)
            metrics_result["semantic_pixel_accuracy"] = float(semantic_pixel_acc)
            metrics_result["semantic_num_classes"] = int(semantic_num_classes)
            print(f"Semantic classes evaluated: {semantic_num_classes}")
            print(f"Multi-class semantic mIoU: {semantic_miou:.4f}")
            print(f"Multi-class pixel F1 (macro): {semantic_macro_f1:.4f}")
            print(f"Multi-class pixel F1 (micro): {semantic_micro_f1:.4f}")
            print(f"Pixel accuracy: {semantic_pixel_acc:.4f}")

            semantic_head_metrics = compute_multiclass_semantic_metrics_from_prob_maps(
                coco_gt_dict=coco_gt_dict,
                semantic_prob_maps=semantic_head_prob_maps,
                threshold=0.5,
            )
            semantic_head_miou = semantic_head_metrics["semantic_mIoU"]
            semantic_head_macro_f1 = semantic_head_metrics["semantic_macro_F1"]
            semantic_head_micro_f1 = semantic_head_metrics["semantic_micro_F1"]
            semantic_head_pixel_acc = semantic_head_metrics["semantic_pixel_accuracy"]
            metrics_result["semantic_head_mIoU"] = float(semantic_head_miou)
            metrics_result["semantic_head_macro_F1"] = float(semantic_head_macro_f1)
            metrics_result["semantic_head_micro_F1"] = float(semantic_head_micro_f1)
            metrics_result["semantic_head_pixel_accuracy"] = float(semantic_head_pixel_acc)
            print("\nDirect semantic head metrics:")
            print(f"Semantic-head mIoU: {semantic_head_miou:.4f}")
            print(f"Semantic-head pixel F1 (macro): {semantic_head_macro_f1:.4f}")
            print(f"Semantic-head pixel F1 (micro): {semantic_head_micro_f1:.4f}")
            print(f"Semantic-head pixel accuracy: {semantic_head_pixel_acc:.4f}")
        else:
            print(
                "\n[WARN] Ground truth annotations do not all contain segmentation masks "
                f"({gt_annotations_with_seg}/{len(gt_annotations)} with segmentation). "
                "Skipping APmask, cgF1, and semantic metrics."
            )

        # Print summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        if has_original_bbox_eval:
            print(f"APbbox (IoU 0.50:0.95, original res): {map_bbox:.4f}")
            print(f"APbbox@50 (original res): {map50_bbox:.4f}")
            print(f"APbbox@75 (original res): {map75_bbox:.4f}")
        else:
            print("APbbox original-res: skipped (no original-res bbox predictions)")
        print(f"APbbox (IoU 0.50:0.95, 288 eval): {map_bbox_288:.4f}")
        print(f"APbbox@50 (288 eval): {map50_bbox_288:.4f}")
        print(f"APbbox@75 (288 eval): {map75_bbox_288:.4f}")
        if has_full_gt_segmentation:
            print(f"APmask (IoU 0.50:0.95): {map_segm:.4f}")
            print(f"APmask@50: {map50_segm:.4f}")
            print(f"APmask@75: {map75_segm:.4f}")
            print(f"cgF1 (IoU 0.50:0.95): {cgf1:.4f}")
            print(f"cgF1@50: {cgf1_50:.4f}")
            print(f"cgF1@75: {cgf1_75:.4f}")
            print(f"Semantic mIoU: {semantic_miou:.4f}")
            print(f"Semantic pixel F1 (macro): {semantic_macro_f1:.4f}")
            print(f"Semantic pixel F1 (micro): {semantic_micro_f1:.4f}")
            print(f"Pixel accuracy: {semantic_pixel_acc:.4f}")
            print(f"Semantic-head mIoU: {semantic_head_miou:.4f}")
            print(f"Semantic-head pixel F1 (macro): {semantic_head_macro_f1:.4f}")
            print(f"Semantic-head pixel F1 (micro): {semantic_head_micro_f1:.4f}")
            print(f"Semantic-head pixel accuracy: {semantic_head_pixel_acc:.4f}")
        else:
            print("APmask / cgF1 / semantic metrics: skipped (bbox-only or mixed GT without full segmentation)")
        print("="*80)

        # Cleanup temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    else:
        print("\n[ERROR] No predictions generated! Cannot compute metrics.")

    return metrics_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone validation script for SAM3 LoRA model with APmask/APbbox, bbox-only GT support, cgF1, and SAM3 NMS"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (for LoRA settings). Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights file. Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Direct path to validation data directory containing _annotations.coco.json (e.g., /workspace/data2/valid)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use original SAM3 model without LoRA (for baseline comparison)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit validation to N samples (for debugging)"
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.3,
        help="Probability threshold for filtering predictions (default: 0.3)"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable aggressive merging of overlapping segments (recommended for crack detection)"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.15,
        help="IoU threshold for merging overlapping predictions (default: 0.15, lower = more aggressive)"
    )
    args = parser.parse_args()

    # Validate argument combinations
    if not args.use_base_model:
        if args.config is None or args.weights is None:
            parser.error("--config and --weights are required when not using --use-base-model")

    validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        use_base_model=args.use_base_model
    )
