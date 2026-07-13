#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Manual integration visualizer for COCO mask prompts through SAM3 transforms.

This script validates the phase-2/phase-3 data path:

COCO instances JSON -> COCO_FROM_JSON -> Sam3ImageDataset -> DecodeRle ->
crop/hflip/resize/pad -> transformed query.input_mask.

Transform note:
The base validation image transform mirrors the Roboflow/eval_base validation
configuration used in this repo: RandomResizeAPI(size=1008, square=True),
ToTensorAPI, and NormalizeAPI(mean/std=[0.5, 0.5, 0.5]). For this manual
stage-3 integration check, DecodeRle is prepended and existing SAM3 transform
implementations are used to exercise crop, hflip, resize, and pad in one
deterministic pipeline. The script does not reimplement those algorithms.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.sam3_image_dataset import Datapoint, Object, Sam3ImageDataset
from sam3.train.transforms.basic import get_random_resize_max_size
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    PadToSizeAPI,
    RandomHorizontalFlip,
    RandomResizeAPI,
    ToTensorAPI,
    crop,
)
from sam3.train.transforms.segmentation import DecodeRle


BASE_VAL_TRANSFORM_CONFIG = (
    "sam3/train/configs/eval_base.yaml and "
    "sam3/train/configs/roboflow_v100/roboflow_v100_eval.yaml "
    "base validation resize/tensor/normalize, augmented here with "
    "DecodeRle + full-image crop + deterministic hflip + PadToSizeAPI for "
    "manual phase-3 spatial validation."
)

RESIZE_SIZE = 1008
PAD_SIZE = 1024
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
ALIGNMENT_IOU_THRESHOLD = 0.99


class FullImageCropAPI:
    """Run the existing crop transform with a full-image crop region."""

    def __call__(self, datapoint: Datapoint, **kwargs):
        for image_index, image in enumerate(datapoint.images):
            width, height = _image_width_height(image.data)
            datapoint = crop(
                datapoint,
                image_index,
                (0, 0, height, width),
                check_validity=False,
                check_input_validity=False,
            )
        return datapoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize and validate COCO mask prompts after SAM3 dataset "
            "loading and spatial transforms."
        )
    )
    parser.add_argument("--coco_root", default="", help="Optional COCO root path.")
    parser.add_argument("--ann_file", required=True, help="COCO instances JSON.")
    parser.add_argument("--img_folder", required=True, help="COCO image folder.")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs.")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--category_chunk_size", type=int, default=2)
    parser.add_argument("--min_instances", type=int, default=2)
    return parser.parse_args()


def build_transforms() -> List[Any]:
    return [
        ComposeAPI(
            [
                DecodeRle(),
                FullImageCropAPI(),
                RandomHorizontalFlip(consistent_transform=False, p=1.0),
                RandomResizeAPI(
                    sizes=RESIZE_SIZE,
                    max_size=get_random_resize_max_size(RESIZE_SIZE),
                    square=True,
                    consistent_transform=False,
                ),
                PadToSizeAPI(
                    size=PAD_SIZE,
                    consistent_transform=False,
                    bottom_right=True,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=NORM_MEAN, std=NORM_STD),
            ]
        )
    ]


def build_dataset(args: argparse.Namespace) -> Sam3ImageDataset:
    coco_json_loader = functools.partial(
        COCO_FROM_JSON,
        include_negatives=False,
        category_chunk_size=args.category_chunk_size,
        use_mask_prompt=True,
        min_instances_per_prompt_category=args.min_instances,
        prompt_sampling_policy="median_area",
        fixed_val_prompt_policy="median_area",
    )
    dataset = Sam3ImageDataset(
        img_folder=args.img_folder,
        ann_file=args.ann_file,
        transforms=build_transforms(),
        max_ann_per_img=100000,
        multiplier=1,
        training=False,
        load_segmentation=True,
        max_val_queries=100000,
        use_caching=False,
        coco_json_loader=coco_json_loader,
    )
    dataset.set_curr_epoch(0)
    return dataset


def _image_width_height(image_data: Any) -> Tuple[int, int]:
    if isinstance(image_data, PILImage.Image):
        return image_data.size
    if isinstance(image_data, torch.Tensor):
        return int(image_data.shape[-1]), int(image_data.shape[-2])
    raise TypeError(f"Unsupported image data type: {type(image_data)!r}")


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _safe_to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return _to_int(value)
    except (TypeError, ValueError):
        return None


def _positive_label(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.item() == 1)
    return value == 1 or value is True


def _query_category_id(query: Any) -> Any:
    metadata = getattr(query, "inference_metadata", None)
    if metadata is not None:
        category_id = getattr(metadata, "original_category_id", None)
        if category_id is not None and category_id != -1:
            return category_id
    return query.query_text


def _dataset_datapoint_id(dataset: Sam3ImageDataset, dataset_index: int) -> int:
    return int(dataset.ids[dataset_index].item())


def _raw_image_record(dataset: Sam3ImageDataset, datapoint_id: int) -> Dict[str, Any]:
    coco = dataset.coco
    if hasattr(coco, "_datapoint_category_chunks") and coco._datapoint_category_chunks:
        image_index, _ = coco._datapoint_category_chunks[datapoint_id]
    else:
        image_index = datapoint_id
    return coco._raw_data[image_index]["image"]


def _raw_loader_records(
    dataset: Sam3ImageDataset, datapoint_id: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return dataset.coco.loadQueriesAndAnnotationsFromDatapoint(datapoint_id)


def _raw_mask_type(raw_mask: Any) -> str:
    if isinstance(raw_mask, list):
        return "polygon"
    if isinstance(raw_mask, dict):
        counts = raw_mask.get("counts")
        if isinstance(counts, list):
            return "uncompressed_rle"
        return "rle"
    return type(raw_mask).__name__


def _object_id(obj: Object) -> Optional[int]:
    return _safe_to_int(getattr(obj, "object_id", None))


def _object_by_id(objects: Iterable[Object]) -> Dict[int, Object]:
    object_by_id: Dict[int, Object] = {}
    for obj in objects:
        object_id = _object_id(obj)
        if object_id is not None:
            object_by_id[object_id] = obj
    return object_by_id


def _query_target_ids(query: Any) -> List[int]:
    return [_to_int(object_id) for object_id in query.object_ids_output]


def _mask_2d(mask: Any) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected tensor mask, got {type(mask)!r}")
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.dim() == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.dim() != 2:
        raise ValueError(f"Expected [H, W] mask after squeeze, got {tuple(mask.shape)}")
    return mask.detach().cpu()


def _mask_bool(mask: Any) -> torch.Tensor:
    mask_2d = _mask_2d(mask)
    if mask_2d.dtype == torch.bool:
        return mask_2d
    return mask_2d > 0.5 if mask_2d.is_floating_point() else mask_2d > 0


def _unique_values(mask: torch.Tensor) -> List[Any]:
    values = torch.unique(mask.detach().cpu())
    result: List[Any] = []
    for value in values.tolist():
        if isinstance(value, bool):
            result.append(bool(value))
        elif isinstance(value, float):
            result.append(float(value))
        else:
            result.append(int(value))
    return result


def _is_binary_unique(unique_values: Sequence[Any]) -> bool:
    allowed = {0, 1, 0.0, 1.0, False, True}
    return all(value in allowed for value in unique_values)


def _iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()
    intersection = torch.logical_and(mask_a, mask_b).sum().item()
    union = torch.logical_or(mask_a, mask_b).sum().item()
    return float(intersection / max(union, 1))


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    tensor = image_tensor.detach().cpu().float()
    mean = torch.tensor(NORM_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(NORM_STD, dtype=tensor.dtype).view(3, 1, 1)
    tensor = (tensor * std + mean).clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def _sanitize_filename_part(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:80] or "query"


def _bbox_to_xyxy_pixels(
    bbox: Any,
    height: int,
    width: int,
) -> Optional[Tuple[float, float, float, float, str]]:
    if bbox is None:
        return None
    box = torch.as_tensor(bbox, dtype=torch.float32).view(-1)
    if box.numel() != 4:
        return None

    values = box.tolist()
    min_value = min(values)
    max_value = max(values)
    if min_value >= -0.01 and max_value <= 1.01:
        cx, cy, bw, bh = values
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        return x1, y1, x2, y2, "normalized_cxcywh"

    x1, y1, x2, y2 = values
    return x1, y1, x2, y2, "absolute_xyxy"


def _apply_mask_overlay(
    image: np.ndarray,
    mask: torch.Tensor,
    color: Tuple[float, float, float],
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image.copy()
    mask_np = mask.detach().cpu().bool().numpy()
    color_array = np.asarray(color, dtype=np.float32)
    overlay[mask_np] = overlay[mask_np] * (1.0 - alpha) + color_array * alpha
    return overlay


def _draw_mask_contour(ax: Any, mask: torch.Tensor, color: str, linewidth: float = 1.5):
    mask_np = mask.detach().cpu().bool().numpy().astype(np.float32)
    if mask_np.any():
        ax.contour(mask_np, levels=[0.5], colors=[color], linewidths=linewidth)


def _draw_bbox(
    ax: Any,
    bbox: Optional[Tuple[float, float, float, float, str]],
    color: str,
    label: Optional[str] = None,
):
    if bbox is None:
        return
    import matplotlib.patches as patches

    x1, y1, x2, y2, _ = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    rect = patches.Rectangle(
        (x1, y1),
        width,
        height,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1,
            max(0.0, y1 - 4.0),
            label,
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
        )


def _mask_centroid(mask: torch.Tensor) -> Optional[Tuple[float, float]]:
    coords = torch.nonzero(mask.detach().cpu().bool(), as_tuple=False)
    if coords.numel() == 0:
        return None
    yx = coords.float().mean(dim=0)
    return float(yx[1].item()), float(yx[0].item())


def _label_mask(ax: Any, mask: torch.Tensor, label: str, color: str):
    centroid = _mask_centroid(mask)
    if centroid is None:
        return
    ax.text(
        centroid[0],
        centroid[1],
        label,
        color=color,
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
    )


def _set_axis(ax: Any):
    ax.set_xticks([])
    ax.set_yticks([])


def _make_visualization(
    output_path: Path,
    image_array: np.ndarray,
    sample_info: Dict[str, Any],
    prompt_mask: torch.Tensor,
    prompt_target_mask: torch.Tensor,
    prompt_object: Object,
    target_objects: Dict[int, Object],
    target_ids: Sequence[int],
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height, width = prompt_mask.shape
    prompt_id = sample_info["prompt_object_id"]
    prompt_bbox = _bbox_to_xyxy_pixels(prompt_object.bbox, height, width)
    colors = [
        "cyan",
        "yellow",
        "magenta",
        "orange",
        "lime",
        "deepskyblue",
        "white",
        "violet",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=140)
    axes = axes.reshape(2, 2)

    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title(
        "\n".join(
            [
                f"image_id={sample_info['image_id']} query={sample_info['query_text']}",
                f"dataset_index={sample_info['dataset_index']} shape={sample_info['image_shape']}",
            ]
        )
    )
    _set_axis(axes[0, 0])

    prompt_overlay = _apply_mask_overlay(image_array, prompt_mask, (1.0, 0.0, 0.0))
    axes[0, 1].imshow(prompt_overlay)
    _draw_mask_contour(axes[0, 1], prompt_mask, "red", linewidth=1.8)
    _label_mask(axes[0, 1], prompt_mask, f"id={prompt_id}", "red")
    axes[0, 1].set_title("Input mask prompt")
    _set_axis(axes[0, 1])

    target_overlay = _apply_mask_overlay(
        image_array, prompt_target_mask, (0.0, 1.0, 0.0)
    )
    axes[1, 0].imshow(target_overlay)
    _draw_mask_contour(axes[1, 0], prompt_target_mask, "lime", linewidth=1.8)
    _draw_bbox(axes[1, 0], prompt_bbox, "lime", label=f"id={prompt_id}")
    bbox_format = prompt_bbox[-1] if prompt_bbox is not None else "missing_bbox"
    axes[1, 0].set_title(
        f"Prompt target IoU={sample_info['prompt_target_iou']:.4f} bbox={bbox_format}"
    )
    _set_axis(axes[1, 0])

    axes[1, 1].imshow(image_array)
    for index, object_id in enumerate(target_ids):
        target_obj = target_objects.get(object_id)
        if target_obj is None:
            continue
        color = "red" if object_id == prompt_id else colors[index % len(colors)]
        if target_obj.segment is not None:
            target_mask = _mask_bool(target_obj.segment)
            _draw_mask_contour(axes[1, 1], target_mask, color, linewidth=1.4)
            _label_mask(axes[1, 1], target_mask, str(object_id), color)
        bbox = _bbox_to_xyxy_pixels(target_obj.bbox, height, width)
        _draw_bbox(axes[1, 1], bbox, color)
    axes[1, 1].set_title(f"All same-category targets num_targets={len(target_ids)}")
    _set_axis(axes[1, 1])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _make_contact_sheet(
    image_paths: Sequence[Path], output_path: Path, columns: int = 2
):
    if not image_paths:
        return
    thumbs = []
    max_width = 720
    for image_path in image_paths:
        image = PILImage.open(image_path).convert("RGB")
        ratio = max_width / float(image.width)
        thumb_size = (max_width, max(1, int(round(image.height * ratio))))
        image.thumbnail(thumb_size, PILImage.LANCZOS)
        canvas = PILImage.new("RGB", thumb_size, "white")
        canvas.paste(image, (0, 0))
        thumbs.append(canvas)

    rows = int(math.ceil(len(thumbs) / float(columns)))
    cell_width = max(image.width for image in thumbs)
    cell_height = max(image.height for image in thumbs)
    sheet = PILImage.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * cell_width
        y = (idx // columns) * cell_height
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def _skip(skips: List[Dict[str, Any]], dataset_index: int, reason: str, **extra):
    record = {"dataset_index": dataset_index, "reason": reason}
    record.update(extra)
    skips.append(record)


def _valid_queries(datapoint: Datapoint, min_instances: int) -> List[Any]:
    valid = []
    for query in datapoint.find_queries:
        prompt_id = _safe_to_int(query.input_mask_object_id)
        if (
            query.input_mask is not None
            and _positive_label(query.input_mask_label)
            and prompt_id is not None
            and len(query.object_ids_output) >= min_instances
        ):
            valid.append(query)
    return valid


def _category_query_counts(datapoint: Datapoint) -> Counter:
    return Counter(_query_category_id(query) for query in datapoint.find_queries)


def _build_raw_maps(
    raw_queries: Sequence[Dict[str, Any]],
) -> Dict[Any, Dict[str, Any]]:
    raw_query_by_category = {query["original_cat_id"]: query for query in raw_queries}
    return raw_query_by_category


def _annotation_category_map_for_datapoint(
    dataset: Sam3ImageDataset,
    datapoint_id: int,
    raw_annotations: Sequence[Dict[str, Any]],
) -> Dict[int, int]:
    """Map COCO_FROM_JSON local annotation ids to source category ids."""

    # COCO_FROM_JSON keeps annotation ids local to the datapoint. The returned
    # annotations do not store category ids, so reconstruct the map from the
    # same image/category chunk order used by COCO_FROM_JSON.
    coco = dataset.coco
    image_index, category_chunk = coco._datapoint_category_chunks[datapoint_id]
    raw_image_annotations = coco._raw_data[image_index]["annotations"]
    raw_annotations_by_category: Dict[int, List[Dict[str, Any]]] = {}
    for raw_annotation in raw_image_annotations:
        category_id = int(raw_annotation["category_id"])
        raw_annotations_by_category.setdefault(category_id, []).append(raw_annotation)

    category_by_object_id: Dict[int, int] = {}
    local_annotation_index = 0
    for category_id in category_chunk:
        for _ in raw_annotations_by_category.get(int(category_id), []):
            if local_annotation_index >= len(raw_annotations):
                break
            annotation = raw_annotations[local_annotation_index]
            category_by_object_id[int(annotation["id"])] = int(category_id)
            local_annotation_index += 1
    return category_by_object_id


def _sample_output_name(sample_index: int, image_id: Any, query_text: str) -> str:
    return (
        f"sample_{sample_index:03d}_{image_id}_"
        f"{_sanitize_filename_part(query_text)}.png"
    )


def process_dataset(
    dataset: Sam3ImageDataset,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Path]]:
    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    samples: List[Dict[str, Any]] = []
    skips: List[Dict[str, Any]] = []
    output_files: List[Path] = []

    for dataset_index in indices:
        if len(samples) >= args.num_samples:
            break

        datapoint_id = _dataset_datapoint_id(dataset, dataset_index)
        raw_image = _raw_image_record(dataset, datapoint_id)
        raw_queries, raw_annotations = _raw_loader_records(dataset, datapoint_id)
        raw_query_by_category = _build_raw_maps(raw_queries)
        raw_category_by_object_id = _annotation_category_map_for_datapoint(
            dataset, datapoint_id, raw_annotations
        )

        try:
            datapoint = dataset[dataset_index]
        except Exception as error:  # noqa: BLE001 - manual debug script records skips.
            _skip(skips, dataset_index, "dataset_load_error", error=str(error))
            continue

        valid_queries = _valid_queries(datapoint, args.min_instances)
        if not valid_queries:
            _skip(skips, dataset_index, "no_valid_mask_prompt_query")
            continue

        query = rng.choice(valid_queries)
        image = datapoint.images[query.image_id]
        category_counts = _category_query_counts(datapoint)
        query_category_id = _query_category_id(query)
        one_query_for_category = category_counts[query_category_id] == 1
        if not one_query_for_category:
            _skip(
                skips,
                dataset_index,
                "multiple_queries_for_image_category",
                query_category_id=query_category_id,
                count=category_counts[query_category_id],
            )
            continue

        prompt_object_id = _safe_to_int(query.input_mask_object_id)
        if prompt_object_id is None:
            _skip(skips, dataset_index, "missing_prompt_object_id")
            continue

        target_object_ids = _query_target_ids(query)
        if prompt_object_id not in target_object_ids:
            _skip(
                skips,
                dataset_index,
                "prompt_object_id_not_in_query_targets",
                prompt_object_id=prompt_object_id,
                target_object_ids=target_object_ids,
            )
            continue

        object_by_id = _object_by_id(image.objects)
        if prompt_object_id not in object_by_id:
            _skip(
                skips,
                dataset_index,
                "prompt_object_id_not_in_object_by_id",
                prompt_object_id=prompt_object_id,
                object_ids=sorted(object_by_id.keys()),
            )
            continue

        missing_targets = [
            object_id
            for object_id in target_object_ids
            if object_id not in object_by_id
        ]
        if missing_targets:
            _skip(
                skips,
                dataset_index,
                "target_object_id_not_in_object_by_id",
                missing_targets=missing_targets,
            )
            continue

        raw_query = raw_query_by_category.get(query_category_id)
        raw_mask_kind = (
            _raw_mask_type(raw_query.get("input_mask")) if raw_query else None
        )
        raw_target_ids = raw_query.get("object_ids_output", []) if raw_query else []

        category_mismatches = [
            object_id
            for object_id in target_object_ids
            if raw_category_by_object_id.get(object_id) != query_category_id
        ]
        if category_mismatches:
            _skip(
                skips,
                dataset_index,
                "target_category_mismatch",
                query_category_id=query_category_id,
                mismatched_object_ids=category_mismatches,
            )
            continue

        if len(raw_target_ids) != len(target_object_ids):
            _skip(
                skips,
                dataset_index,
                "target_count_mismatch",
                raw_num_targets=len(raw_target_ids),
                transformed_num_targets=len(target_object_ids),
            )
            continue

        if not isinstance(image.data, torch.Tensor):
            _skip(skips, dataset_index, "image_not_tensor_after_transforms")
            continue

        _, height, width = image.data.shape
        try:
            prompt_mask_tensor = _mask_2d(query.input_mask)
        except (TypeError, ValueError) as error:
            _skip(skips, dataset_index, "invalid_prompt_mask_shape", error=str(error))
            continue

        if tuple(prompt_mask_tensor.shape) != (height, width):
            _skip(
                skips,
                dataset_index,
                "prompt_mask_shape_mismatch",
                image_shape=list(image.data.shape),
                input_mask_shape=list(prompt_mask_tensor.shape),
            )
            continue

        unique_values = _unique_values(prompt_mask_tensor)
        if not _is_binary_unique(unique_values):
            _skip(
                skips,
                dataset_index,
                "prompt_mask_non_binary_values",
                input_mask_unique_values=unique_values[:20],
            )
            continue

        prompt_mask = _mask_bool(prompt_mask_tensor)
        input_mask_area = int(prompt_mask.sum().item())
        if input_mask_area <= 0:
            _skip(skips, dataset_index, "prompt_mask_empty_after_transform")
            continue

        prompt_object = object_by_id[prompt_object_id]
        if prompt_object.segment is None:
            _skip(skips, dataset_index, "prompt_target_segment_missing")
            continue

        prompt_target_mask = _mask_bool(prompt_object.segment)
        if tuple(prompt_target_mask.shape) != (height, width):
            _skip(
                skips,
                dataset_index,
                "prompt_target_shape_mismatch",
                target_mask_shape=list(prompt_target_mask.shape),
                image_shape=list(image.data.shape),
            )
            continue

        prompt_target_iou = _iou(prompt_mask, prompt_target_mask)
        alignment_pass = prompt_target_iou >= ALIGNMENT_IOU_THRESHOLD

        image_id = getattr(query.inference_metadata, "coco_image_id", raw_image["id"])
        output_name = _sample_output_name(len(samples), image_id, query.query_text)
        output_path = output_dir / output_name

        sample_info: Dict[str, Any] = {
            "dataset_index": dataset_index,
            "datapoint_id": datapoint_id,
            "image_id": image_id,
            "file_name": raw_image.get("file_name"),
            "query_text": query.query_text,
            "query_category_id": query_category_id,
            "raw_input_mask_type": raw_mask_kind,
            "prompt_object_id": prompt_object_id,
            "target_object_ids": target_object_ids,
            "raw_target_object_ids": raw_target_ids,
            "num_targets": len(target_object_ids),
            "num_queries_for_category": category_counts[query_category_id],
            "one_query_per_image_category_pass": one_query_for_category,
            "image_shape": list(image.data.shape),
            "input_mask_shape": list(prompt_mask_tensor.shape),
            "input_mask_dtype": str(prompt_mask_tensor.dtype),
            "input_mask_unique_values": unique_values,
            "input_mask_area": input_mask_area,
            "prompt_target_iou": prompt_target_iou,
            "alignment_pass": alignment_pass,
            "output_file": output_name,
        }

        image_array = _denormalize_image(image.data)
        target_objects = {
            object_id: object_by_id[object_id] for object_id in target_object_ids
        }
        _make_visualization(
            output_path=output_path,
            image_array=image_array,
            sample_info=sample_info,
            prompt_mask=prompt_mask,
            prompt_target_mask=prompt_target_mask,
            prompt_object=prompt_object,
            target_objects=target_objects,
            target_ids=target_object_ids,
        )

        samples.append(sample_info)
        output_files.append(output_path)

    return samples, skips, output_files


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    samples: Sequence[Dict[str, Any]],
    skips: Sequence[Dict[str, Any]],
    contact_sheet_path: Optional[Path],
) -> Path:
    all_alignment_passed = all(sample["alignment_pass"] for sample in samples)
    summary = {
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "successful_samples": len(samples),
        "skipped_samples": len(skips),
        "all_alignment_passed": all_alignment_passed,
        "alignment_iou_threshold": ALIGNMENT_IOU_THRESHOLD,
        "val_transform_config": BASE_VAL_TRANSFORM_CONFIG,
        "samples": list(samples),
        "skips": list(skips),
        "contact_sheet": contact_sheet_path.name if contact_sheet_path else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _iou_stats(samples: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    values = [float(sample["prompt_target_iou"]) for sample in samples]
    return min(values), sum(values) / len(values), max(values)


def _print_result(
    args: argparse.Namespace,
    output_dir: Path,
    samples: Sequence[Dict[str, Any]],
    skips: Sequence[Dict[str, Any]],
    contact_sheet_path: Optional[Path],
    summary_path: Path,
):
    iou_min, iou_mean, iou_max = _iou_stats(samples)
    passed = sum(1 for sample in samples if sample["alignment_pass"])
    print("COCO mask-prompt transform visualization complete")
    print(f"Requested samples: {args.num_samples}")
    print(f"Successful samples: {len(samples)}")
    print(f"Skipped samples: {len(skips)}")
    print(
        f"Prompt-target IoU min/mean/max: {iou_min:.4f} / {iou_mean:.4f} / {iou_max:.4f}"
    )
    print(f"Alignment passed: {passed} / {len(samples)}")
    print(f"Output directory: {output_dir}")
    print(
        f"Contact sheet: {contact_sheet_path if contact_sheet_path else 'not created'}"
    )
    print(f"Summary: {summary_path}")
    skip_counts = Counter(skip["reason"] for skip in skips)
    if skip_counts:
        print("Skip reasons:")
        for reason, count in sorted(skip_counts.items()):
            print(f"  {reason}: {count}")
    if any(not sample["alignment_pass"] for sample in samples):
        print("WARNING: spatial misalignment detected")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    samples, skips, output_files = process_dataset(dataset, args, output_dir)

    contact_sheet_path = output_dir / "contact_sheet.png" if output_files else None
    if contact_sheet_path is not None:
        _make_contact_sheet(output_files, contact_sheet_path)

    summary_path = write_summary(output_dir, args, samples, skips, contact_sheet_path)
    _print_result(args, output_dir, samples, skips, contact_sheet_path, summary_path)

    if len(samples) < args.num_samples:
        return 2
    if any(not sample["alignment_pass"] for sample in samples):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
