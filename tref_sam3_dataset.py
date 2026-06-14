import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pycocotools.mask as mask_utils
import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset
from torchvision.transforms import v2

from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)


class ReferringSegmentDataset(Dataset):
    """Unified referring-expression segmentation dataset for TRef-SAM3.

    Expected directory layout:
      data_root/
        annotations.json
        referring_annotations.json

    ``annotations.json`` is COCO-style instance segmentation. ``referring_annotations``
    contains records with at least:
      - image_id
      - expression
      - target_ann_ids

    This keeps the SAM3 datapoint contract unchanged, but changes the query
    semantics from category-to-all-instances to expression-to-target-instances.
    """

    def __init__(
        self,
        data_dir,
        split: Optional[str] = "train",
        annotations_file: str = "annotations.json",
        refs_file: str = "referring_annotations.json",
        max_queries_per_image: int = 0,
        include_multi_target: bool = True,
        fallback_to_category_queries: bool = False,
        training: bool = True,
        resolution: int = 1008,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.training = bool(training)
        self.max_queries_per_image = int(max_queries_per_image or 0)
        self.include_multi_target = bool(include_multi_target)
        self.fallback_to_category_queries = bool(fallback_to_category_queries)
        self.resolution = int(resolution)

        ann_path = self._resolve_existing_path(annotations_file)
        ref_path = self._resolve_existing_path(refs_file)

        with open(ann_path, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)
        with open(ref_path, "r", encoding="utf-8") as f:
            refs = json.load(f)

        self.images: Dict[int, dict] = {
            int(img["id"]): img for img in self.coco_data.get("images", [])
        }
        self.categories: Dict[int, str] = {
            int(cat["id"]): str(cat["name"])
            for cat in self.coco_data.get("categories", [])
        }

        self.img_to_anns: Dict[int, List[dict]] = defaultdict(list)
        for ann in self.coco_data.get("annotations", []):
            self.img_to_anns[int(ann["image_id"])].append(ann)

        self.refs_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ref in refs:
            if not self._keep_ref_for_split(ref):
                continue
            if not self.include_multi_target and len(ref.get("target_ann_ids", [])) != 1:
                continue
            self.refs_by_image[int(ref["image_id"])].append(ref)

        if self.fallback_to_category_queries:
            self.image_ids = sorted(self.images.keys())
        else:
            self.image_ids = sorted(
                image_id for image_id in self.refs_by_image.keys() if image_id in self.images
            )

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        num_refs = sum(len(v) for v in self.refs_by_image.values())
        print(f"Loaded TRef dataset from {self.data_dir}")
        print(f"  Split: {self.split}")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data.get('annotations', []))}")
        print(f"  Referring queries: {num_refs}")
        print(f"  Categories: {self.categories}")

    def _resolve_existing_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            candidate = path
        else:
            candidate = self.data_dir / path
        if not candidate.exists():
            raise FileNotFoundError(f"Required TRef file not found: {candidate}")
        return candidate

    def _keep_ref_for_split(self, ref: dict) -> bool:
        if self.split is None:
            return True
        ref_split = str(ref.get("split", "")).strip().lower()
        if not ref_split:
            return True
        return ref_split == str(self.split).strip().lower()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = int(self.image_ids[idx])
        img_info = self.images[img_id]

        img_path = self._resolve_image_path(img_info["file_name"])
        pil_image_original = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image_original.size
        pil_image = pil_image_original.resize(
            (self.resolution, self.resolution), PILImage.BILINEAR
        )
        image_tensor = self.transform(pil_image)

        annotations = self.img_to_anns.get(img_id, [])
        objects = []
        object_class_names = []
        ann_id_to_object_id = {}

        scale_w = self.resolution / max(float(orig_w), 1.0)
        scale_h = self.resolution / max(float(orig_h), 1.0)

        for ann in annotations:
            bbox_coco = ann.get("bbox")
            if bbox_coco is None:
                continue
            x, y, w, h = [float(v) for v in bbox_coco]
            if w <= 0 or h <= 0:
                continue

            category_id = int(ann.get("category_id", 0))
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)

            # SAM3 loss in this training script expects normalized CxCyWH boxes.
            cx = x + w / 2.0
            cy = y + h / 2.0
            box_tensor = torch.tensor(
                [
                    cx * scale_w / self.resolution,
                    cy * scale_h / self.resolution,
                    w * scale_w / self.resolution,
                    h * scale_h / self.resolution,
                ],
                dtype=torch.float32,
            )
            box_tensor = box_tensor.clamp(0.0, 1.0)

            segment = self._decode_segment(ann, orig_h, orig_w)
            if segment is not None:
                segment = torch.nn.functional.interpolate(
                    segment.float().unsqueeze(0).unsqueeze(0),
                    size=(self.resolution, self.resolution),
                    mode="nearest",
                ).squeeze() > 0.5

            object_id = len(objects)
            ann_id_to_object_id[int(ann["id"])] = object_id
            objects.append(
                Object(
                    bbox=box_tensor,
                    area=float(box_tensor[2] * box_tensor[3]),
                    object_id=object_id,
                    segment=segment,
                )
            )

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution),
        )

        queries = self._build_referring_queries(
            img_id=img_id,
            refs=self.refs_by_image.get(img_id, []),
            ann_id_to_object_id=ann_id_to_object_id,
            object_class_names=object_class_names,
            orig_size=(orig_h, orig_w),
        )

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image],
        )

    def _resolve_image_path(self, file_name: str) -> Path:
        path_text = str(file_name).replace("\\", "/")
        path = Path(path_text)
        if path.is_absolute() and path.exists():
            return path

        # Portable handling for annotations generated on another OS, e.g.
        # E:/SAM3_LoRA/data/train/xxx.jpg on a Linux training machine.
        data_marker = "/data/"
        if path_text.startswith("data/"):
            repo_relative = Path(path_text)
            candidate = Path.cwd() / repo_relative
            if candidate.exists():
                return candidate
        elif data_marker in path_text:
            repo_relative = Path("data") / path_text.split(data_marker, 1)[1]
            candidate = Path.cwd() / repo_relative
            if candidate.exists():
                return candidate

        candidate = self.data_dir / path
        if candidate.exists():
            return candidate
        return candidate

    def _decode_segment(self, ann: dict, orig_h: int, orig_w: int):
        segmentation = ann.get("segmentation")
        if not segmentation:
            return None
        try:
            if isinstance(segmentation, dict):
                counts = segmentation.get("counts")
                if isinstance(counts, list):
                    # COCO may store uncompressed RLE with counts as a list.
                    # pycocotools.decode expects compressed RLE, so convert first.
                    rle = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                    mask_np = mask_utils.decode(rle)
                else:
                    mask_np = mask_utils.decode(segmentation)
            elif isinstance(segmentation, list):
                rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                rle = mask_utils.merge(rles)
                mask_np = mask_utils.decode(rle)
            else:
                return None
            if mask_np.ndim == 3:
                mask_np = mask_np.any(axis=2)
            return torch.from_numpy(mask_np).bool()
        except Exception as exc:
            print(f"Warning: failed to decode segmentation for ann {ann.get('id')}: {exc}")
            return None

    def _build_referring_queries(
        self,
        img_id: int,
        refs: Iterable[dict],
        ann_id_to_object_id: Dict[int, int],
        object_class_names: List[str],
        orig_size,
    ) -> List[FindQueryLoaded]:
        refs = list(refs)
        if self.max_queries_per_image > 0 and len(refs) > self.max_queries_per_image:
            if self.training:
                refs = random.sample(refs, self.max_queries_per_image)
            else:
                refs = refs[: self.max_queries_per_image]

        queries = []
        for order, ref in enumerate(refs):
            obj_ids = [
                ann_id_to_object_id[int(ann_id)]
                for ann_id in ref.get("target_ann_ids", [])
                if int(ann_id) in ann_id_to_object_id
            ]
            if not obj_ids:
                continue

            category_ids = [int(x) for x in ref.get("category_ids", [])]
            original_category_id = category_ids[0] if len(category_ids) == 1 else -1
            object_id_meta = obj_ids[0] if len(obj_ids) == 1 else -1

            queries.append(
                FindQueryLoaded(
                    query_text=str(ref.get("expression", "object")).strip().lower(),
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=bool(ref.get("is_exhaustive", True)),
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=original_category_id,
                        original_size=orig_size,
                        object_id=object_id_meta,
                        frame_index=-1,
                    ),
                )
            )

        if queries:
            return queries
        if self.fallback_to_category_queries:
            return self._build_category_fallback_queries(
                img_id=img_id,
                object_class_names=object_class_names,
                orig_size=orig_size,
            )
        return [
            FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=-1,
                    original_size=orig_size,
                    object_id=-1,
                    frame_index=-1,
                ),
            )
        ]

    def _build_category_fallback_queries(self, img_id, object_class_names, orig_size):
        class_to_object_ids = defaultdict(list)
        for object_id, class_name in enumerate(object_class_names):
            class_to_object_ids[class_name.lower()].append(object_id)

        queries = []
        for query_text, obj_ids in class_to_object_ids.items():
            queries.append(
                FindQueryLoaded(
                    query_text=query_text,
                    image_id=0,
                    object_ids_output=obj_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=img_id,
                        original_image_id=img_id,
                        original_category_id=-1,
                        original_size=orig_size,
                        object_id=-1,
                        frame_index=-1,
                    ),
                )
            )
        if queries:
            return queries
        return [
            FindQueryLoaded(
                query_text="object",
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=img_id,
                    original_image_id=img_id,
                    original_category_id=-1,
                    original_size=orig_size,
                    object_id=-1,
                    frame_index=-1,
                ),
            )
        ]
