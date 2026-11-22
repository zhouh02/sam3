"""Same-image single-box PCS queries, matched to the mask Stage 1 protocol."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random

from pycocotools import mask as mask_util
import torch

from sam3.train.data.coco_json_loaders import COCO_FROM_JSON, ann_to_rle


def has_segmentation(ann):
    seg = ann.get("segmentation")
    return (isinstance(seg, list) and len(seg) > 0) or (
        isinstance(seg, dict) and bool(seg.get("counts")) and bool(seg.get("size")))


def eligible_categories(annotations, minimum=2):
    counts = Counter(a["category_id"] for a in annotations if has_segmentation(a))
    return {category for category, count in counts.items() if count >= minimum}


class BoxPromptCOCO(COCO_FROM_JSON):
    def __init__(self, annotation_file, min_instances_per_prompt_category=2,
                 category_chunk_size=2, prompt_sampling_policy="random", text_mode="category"):
        if prompt_sampling_policy not in ("random", "median_area") or text_mode not in ("category", "visual"):
            raise ValueError("Invalid prompt policy or text mode")
        if min_instances_per_prompt_category < 1 or category_chunk_size < 1:
            raise ValueError("Minimum instances and category chunk size must be positive")
        super().__init__(annotation_file, include_negatives=False, category_chunk_size=category_chunk_size)
        self.policy = prompt_sampling_policy
        self.text_mode = text_mode
        self.minimum = min_instances_per_prompt_category
        self.datapoints = []
        for index, item in enumerate(self._raw_data):
            categories = sorted(eligible_categories(item["annotations"], self.minimum))
            for start in range(0, len(categories), category_chunk_size):
                self.datapoints.append((index, categories[start:start + category_chunk_size]))
        if not self.datapoints:
            raise ValueError("No eligible image/category pairs: each needs at least two segmented instances by default")

    def getDatapointIds(self):
        return list(range(len(self.datapoints)))

    def _view(self, idx):
        index, categories = self.datapoints[idx]
        view = copy.copy(self)
        view._raw_data = [self._raw_data[index]]
        view.category_chunks = [categories]
        return view

    def loadImagesFromDatapoint(self, idx):
        return COCO_FROM_JSON.loadImagesFromDatapoint(self._view(idx), 0)

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        view = self._view(idx)
        queries, annotations = COCO_FROM_JSON.loadQueriesAndAnnotationsFromDatapoint(view, 0)
        image = view._raw_data[0]["image"]
        raw = view._raw_data[0]["annotations"]
        for query in queries:
            # Preserve source annotation order: the mask reference uses local IDs
            # for equal-area tie breaking, not COCO annotation IDs.
            candidates = [a for a in raw if a["category_id"] == query["original_cat_id"] and has_segmentation(a)]
            if self.policy == "random":
                chosen = random.choice(candidates)
            else:
                ordered = sorted(candidates, key=lambda a: a.get("area", a["bbox"][2] * a["bbox"][3] / image["width"] / image["height"]))
                chosen = ordered[len(ordered) // 2]
            # Derive the prompt from the very same mask, rather than assuming
            # annotation bbox is tight. Targets keep their original COCO boxes.
            rle = ann_to_rle(chosen["segmentation"], image)
            if float(mask_util.area(rle)) <= 0:
                raise ValueError(f"Empty prompt segmentation: annotation {chosen['id']}")
            box = torch.as_tensor(mask_util.toBbox(rle), dtype=torch.float32)
            box /= torch.tensor([image["width"], image["height"], image["width"], image["height"]])
            query["input_box"] = box.view(1, 4)
            query["input_box_label"] = [1]
            query["prompt_annotation_id"] = chosen["id"]  # diagnostic only; no loss weighting
            if self.text_mode == "visual":
                query["query_text"] = "visual"
        return queries, annotations


def audit_and_write_gt(annotation, image_root, destination, minimum=2, chunk_size=2):
    """Validate usable targets and restrict AP ground truth to queried categories."""
    source = Path(annotation)
    data = json.loads(source.read_bytes())
    loader = BoxPromptCOCO(str(source), minimum, chunk_size, "median_area")
    eligible = {item["image"]["id"]: eligible_categories(item["annotations"], minimum) for item in loader._raw_data}
    data["images"] = [im for im in data["images"] if eligible[im["id"]]]
    data["annotations"] = [a for a in data["annotations"] if a["category_id"] in eligible.get(a["image_id"], set())]
    images = {im["id"]: im for im in data["images"]}
    for im in images.values():
        if not (Path(image_root) / im["file_name"]).is_file():
            raise FileNotFoundError(Path(image_root) / im["file_name"])
    for ann in data["annotations"]:
        if not has_segmentation(ann) or float(mask_util.area(ann_to_rle(ann["segmentation"], images[ann["image_id"]]))) <= 0:
            raise ValueError(f"Missing/empty target segmentation: annotation {ann['id']}")
        if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
            raise ValueError(f"Invalid target bbox: annotation {ann['id']}")
    if destination is not None:
        Path(destination).write_text(json.dumps(data), encoding="utf-8")
    return {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "images": len(images),
            "targets": len(data["annotations"]), "datapoints": len(loader.datapoints)}
