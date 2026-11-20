"""SAM3 visual-only PCS: one GT box per positive image/category pair, COCO AP+."""

import argparse
import copy
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_path(root, info):
    path = (root / info["file_name"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes image root: {path}")
    return path


def prepare_queries(coco, image_ids, seed, saved=None):
    """Keep all annotated pairs, including crowd-only pairs as in COCO_FROM_JSON.

    Sampling is stable under changes to traversal order, JSON order and --limit.
    This deliberately does not claim the paper's unpublished exact RNG sequence.
    """
    grouped = defaultdict(list)
    selected_images = set(image_ids)
    for ann in coco.anns.values():
        if ann["image_id"] in selected_images:
            grouped[ann["image_id"], ann["category_id"]].append(ann)
    saved_by_pair = {}
    if saved is not None:
        for query in saved:
            pair = query["image_id"], query["category_id"]
            if pair in saved_by_pair:
                raise ValueError(f"Duplicate saved prompt: {pair}")
            saved_by_pair[pair] = query
        if set(saved_by_pair) != set(grouped):
            raise ValueError("Saved prompts must cover exactly the selected positive image/category pairs")
    queries = []
    for (image_id, category_id), anns in sorted(grouped.items()):
        if category_id not in coco.cats:
            raise ValueError(f"Unknown category: {category_id}")
        info = coco.imgs[image_id]
        anns = sorted(anns, key=lambda a: a["id"])
        # The official loader/visual-query transform does not exclude crowd objects.
        if saved is None:
            ann = random.Random(f"{seed}:{image_id}:{category_id}").choice(anns)
        else:
            old = saved_by_pair[image_id, category_id]
            matches = [a for a in anns if a["id"] == old["annotation_id"]]
            if not matches or list(matches[0]["bbox"]) != old["prompt_xywh"]:
                raise ValueError(f"Saved prompt no longer matches annotation: {old}")
            ann = matches[0]
        x, y, w, h = map(float, ann["bbox"])
        if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid GT bbox for annotation {ann['id']}: {ann['bbox']}")
        width, height = info["width"], info["height"]
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_id}")
        # Sam3ImageDataset clips annotation boxes to the image before transforms.
        x0, y0 = min(width, max(0., x)), min(height, max(0., y))
        x1, y1 = min(width, max(0., x+w)), min(height, max(0., y+h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"GT box is empty after clipping: annotation {ann['id']}")
        queries.append({"image_id": image_id, "category_id": category_id,
                        "category_name": coco.cats[category_id]["name"],
                        "file_name": info["file_name"], "annotation_id": ann["id"],
                        "prompt_iscrowd": bool(ann.get("iscrowd", 0)),
                        "prompt_xywh": [x, y, w, h],
                        "prompt_xyxy_clipped": [x0, y0, x1, y1],
                        "prompt_cxcywh_normalized": [(x0+x1)/(2*width), (y0+y1)/(2*height),
                                                    (x1-x0)/width, (y1-y0)/height],
                        "target_instance_count": len(anns)})
    return queries


def top_predictions(records, max_dets):
    """Global per-image top K across categories, matching PredictionDumper's scope."""
    return sorted(records, key=lambda p: p["score"], reverse=True)[:max_dets]


def evaluate(coco, predictions, image_ids, kinds, output):
    import numpy as np
    from pycocotools.coco import COCO
    from sam3.eval.coco_eval_offline import COCOevalCustom, COCO_METRICS

    metrics = {}
    for kind in kinds:
        records = copy.deepcopy(predictions)
        for row in records:
            row.pop("bbox" if kind == "segm" else "segmentation", None)
        if records:
            results = coco.loadRes(records)
        else:
            results = COCO()
            results.dataset = {"images": list(coco.imgs.values()),
                               "categories": list(coco.cats.values()), "annotations": []}
            results.createIndex()
        evaluator = COCOevalCustom(coco, results, iouType=kind, dt_only_positive=True)
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[kind] = {name: float(value) if value >= 0 else None
                         for name, value in zip(COCO_METRICS, evaluator.stats)}
        np.savez_compressed(output / f"coco_{kind}_curves.npz",
                            precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
                            scores=evaluator.eval["scores"], image_ids=image_ids,
                            category_ids=evaluator.params.catIds)
    return metrics


def infer_image(processor, image, queries, include_masks, max_dets, on_query):
    """Encode once; reset every query so boxes cannot accumulate between classes."""
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils

    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for query in queries:
            processor.reset_all_prompts(state)
            state = processor.add_geometric_prompt(query["prompt_cxcywh_normalized"], True, state)
            scores = state["scores"].float().cpu().numpy()
            boxes = state["boxes"].float().cpu().numpy()
            indices = np.argsort(-scores, kind="stable")[:max_dets]
            masks = state["masks"][:, 0].cpu().numpy() if include_masks else None
            for i in indices:
                x0, y0, x1, y1 = map(float, boxes[i])
                row = {"image_id": query["image_id"], "category_id": query["category_id"],
                       "score": float(scores[i]), "bbox": [x0, y0, max(0., x1-x0), max(0., y1-y0)]}
                if include_masks:
                    rle = mask_utils.encode(np.asfortranarray(masks[i].astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("ascii")
                    row["segmentation"] = rle
                records.append(row)
            records = top_predictions(records, max_dets)
            on_query({**query, "raw_prediction_count": len(scores)})
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", help="COCO root (defaults to val2017)")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--checkpoint", type=Path, default=REPO / "sam3.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompts", type=Path, help="Replay an existing prompts.json instead of sampling")
    parser.add_argument("--iou-types", nargs="+", choices=("bbox", "segm"), default=["bbox", "segm"])
    parser.add_argument("--max-dets-per-image", type=int, default=100,
                        help="Total predictions retained across all queried categories per image")
    parser.add_argument("--limit", type=int, default=0, help="First N sorted images; 0 = all")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are never overwritten")
    args = parser.parse_args()
    if args.limit < 0 or args.max_dets_per_image <= 0:
        parser.error("--limit must be >= 0 and --max-dets-per-image > 0")
    if not args.dataset and not (args.annotations and args.image_root):
        parser.error("Provide DATASET_ROOT or both --annotations and --image-root")
    if not args.prepare_only and not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    return args


def main():
    args = parse_args()
    from PIL import Image
    from pycocotools.coco import COCO

    annotations = (args.annotations or args.dataset / "annotations" / f"instances_{args.split}.json").resolve()
    coco = COCO(str(annotations))
    coco.dataset.setdefault("info", {})
    image_ids = sorted(coco.imgs)
    if args.limit:
        image_ids = image_ids[:args.limit]
    if not image_ids:
        raise ValueError("No images in the selected dataset")
    if args.image_root:
        root = args.image_root.resolve()
    else:
        candidates = [args.dataset / "images" / args.split, args.dataset / args.split]
        roots = [p.resolve() for p in candidates
                 if all(image_path(p, coco.imgs[i]).is_file() for i in image_ids)]
        if len(roots) != 1:
            raise ValueError(f"Cannot uniquely locate all selected images under {candidates}; use --image-root")
        root = roots[0]
    for image_id in image_ids:
        info = coco.imgs[image_id]
        with Image.open(image_path(root, info)) as image:
            if image.size != (info["width"], info["height"]):
                raise ValueError(f"Image/annotation size mismatch: {info['file_name']}")
    saved = json.loads(args.prompts.read_text(encoding="utf-8")) if args.prompts else None
    queries = prepare_queries(coco, image_ids, args.seed, saved)
    if not queries:
        raise ValueError("No positive image/category pairs")
    output = (args.output or REPO / "outputs/coco_single_box" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {"status": "prepared", "protocol": "visual-only single GT bbox per positive image/category pair; AP+",
              "annotations": str(annotations), "annotations_sha256": sha256(annotations),
              "image_root": str(root), "split": args.split, "dataset_images": len(coco.imgs),
              "evaluated_images": len(image_ids), "positive_queries": len(queries), "limit": args.limit,
              "seed": args.seed, "sampling": "random.Random(seed:image_id:category_id), sorted annotation IDs",
              "crowd_prompt_policy": "included, matching COCO_FROM_JSON + TextQueryToVisual",
              "replayed_prompts": str(args.prompts.resolve()) if args.prompts else None,
              "checkpoint": str(args.checkpoint.resolve()), "iou_types": args.iou_types,
              "max_dets_per_image_across_categories": args.max_dets_per_image,
              "coco_max_dets": [1, 10, 100], "score_threshold": 0.0, "mask_threshold": 0.5,
              "presence_score": True, "resolution": 1008, "text": "visual", "python": platform.python_version(),
              "note": "Public-protocol reproduction; not the authors' exact COCO sampling/config. Only positive queries are inferred."}
    write_json(output / "prompts.json", queries)
    config["prompts_sha256"] = sha256(output / "prompts.json")
    write_json(output / "config.json", config)
    print(f"Prepared {len(image_ids)} images / {len(queries)} positive queries. Output: {output}", flush=True)
    if args.split == "val2017" and len(coco.imgs) != 5000:
        print(f"WARNING: annotation JSON contains {len(coco.imgs)} images, not full COCO val2017 (5000).", flush=True)
    if args.prepare_only:
        return

    try:
        import numpy as np
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        # Fail before expensive inference if the official evaluator cannot be imported.
        from sam3.eval.coco_eval_offline import COCOevalCustom  # noqa: F401

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This evaluation uses the existing CUDA BF16 inference environment")
        random.seed(args.seed)
        np.random.seed(args.seed % (2**32))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        config.update(status="running", checkpoint_sha256=sha256(args.checkpoint),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), precision="CUDA BF16 autocast",
                      git_commit=revision.stdout.strip())
        write_json(output / "config.json", config)
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint.resolve()), load_from_HF=False,
                                      device="cuda", enable_inst_interactivity=False)
        model.eval()
        processor = Sam3Processor(model, resolution=1008, device="cuda", confidence_threshold=0.0)
        by_image = defaultdict(list)
        for query in queries:
            by_image[query["image_id"]].append(query)
        predictions = []
        started = time.perf_counter()
        with (output / "per_query.jsonl").open("w", encoding="utf-8") as log, \
                (output / "predictions.jsonl").open("w", encoding="utf-8") as pred_log:
            def log_query(row):
                log.write(json.dumps(row) + "\n")
                log.flush()

            for index, image_id in enumerate(image_ids, 1):
                if not by_image[image_id]:
                    continue
                with Image.open(image_path(root, coco.imgs[image_id])) as source:
                    image = source.convert("RGB")  # Stored coordinates; no EXIF transpose.
                rows = infer_image(processor, image, by_image[image_id], "segm" in args.iou_types,
                                   args.max_dets_per_image, log_query)
                predictions.extend(rows)
                for row in rows:
                    pred_log.write(json.dumps(row) + "\n")
                pred_log.flush()
                print(f"[{index}/{len(image_ids)}] image={image_id} queries={len(by_image[image_id])} predictions={len(rows)}", flush=True)
        write_json(output / "predictions.json", predictions)
        metrics = {"protocol": "AP+", "images": len(image_ids), "queries": len(queries),
                   "instances_including_crowd": sum(q["target_instance_count"] for q in queries),
                   "inference_seconds_excluding_model_load": time.perf_counter()-started,
                   "coco": evaluate(coco, predictions, image_ids, args.iou_types, output)}
        write_json(output / "metrics.json", metrics)
        config["status"] = "complete"
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as error:
        config.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "config.json", config)


if __name__ == "__main__":
    main()
