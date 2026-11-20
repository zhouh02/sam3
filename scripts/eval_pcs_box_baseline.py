"""Evaluate SAM3 PCS with one reproducibly sampled GT box per test image."""

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time


def tight_box(mask):
    """Binary mask -> pixel-edge xyxy box (exclusive right/bottom)."""
    import numpy as np
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot prompt with an empty mask")
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def choose_annotation(annotations, seed, image_id):
    # Independent of image traversal, --limit, and annotation JSON ordering.
    rng = random.Random(f"{seed}:{image_id}")
    return rng.choice(sorted(annotations, key=lambda ann: ann["id"]))


def binary_metrics(intersection, predicted, target):
    union = predicted + target - intersection
    return {"iou": intersection / union if union else 1.0,
            "dice": 2 * intersection / (predicted + target) if predicted + target else 1.0}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare_queries(coco, image_root, seed, limit):
    """Validate all masks and images before allocating GPU resources."""
    from PIL import Image
    queries, skipped = [], []
    image_ids = sorted(coco.imgs)
    if limit:
        image_ids = image_ids[:limit]
    for image_id in image_ids:
        info = coco.imgs[image_id]
        path = (image_root / info["file_name"]).resolve()
        if not path.is_relative_to(image_root.resolve()):
            raise ValueError(f"Image path escapes image root: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (info["width"], info["height"]):
                raise ValueError(f"Image/annotation size mismatch: {path}")
        candidates = []
        for ann in coco.imgToAnns[image_id]:
            if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                continue
            mask = coco.annToMask(ann)
            if mask.shape != (info["height"], info["width"]) or not mask.any():
                raise ValueError(f"Invalid/empty mask for annotation {ann['id']}")
            candidates.append(ann)
        if not candidates:
            skipped.append({"image_id": image_id, "reason": "no non-crowd target to sample"})
            continue
        selected = choose_annotation(candidates, seed, image_id)
        box = tight_box(coco.annToMask(selected))
        x0, y0, x1, y1 = box
        width, height = info["width"], info["height"]
        queries.append({"image_id": image_id, "file_name": info["file_name"],
                        "annotation_id": selected["id"], "category_id": selected["category_id"],
                        "prompt_xyxy": box,
                        "prompt_cxcywh_normalized": [(x0+x1)/(2*width), (y0+y1)/(2*height),
                                                    (x1-x0)/width, (y1-y0)/height],
                        "target_instance_count": sum(a["category_id"] == selected["category_id"] for a in candidates)})
    return queries, skipped


def evaluate_coco(coco, predictions, image_ids, output):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    summary = {}
    for kind in ("segm", "bbox"):
        records = copy.deepcopy(predictions)
        if kind == "segm":
            # COCO.loadRes prefers bbox area if both bbox and segmentation exist.
            # Remove boxes so segmentation size buckets use actual mask area.
            for record in records:
                record.pop("bbox", None)
        else:
            for record in records:
                record.pop("segmentation", None)
        if records:
            results = coco.loadRes(records)
        else:
            results = COCO()
            results.dataset = {"images": list(coco.imgs.values()),
                               "categories": list(coco.cats.values()), "annotations": []}
            results.createIndex()
        evaluator = COCOeval(coco, results, kind)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        summary[kind] = {key: float(value) if value >= 0 else None
                         for key, value in zip(names, evaluator.stats)}
        np.savez_compressed(output / f"coco_{kind}_curves.npz",
                            precision=evaluator.eval["precision"], recall=evaluator.eval["recall"],
                            scores=evaluator.eval["scores"])
    return summary


def query_ground_truth(coco, queries):
    """Only the prompted category is a target on each image."""
    from pycocotools.coco import COCO
    categories = {q["image_id"]: q["category_id"] for q in queries}
    filtered = COCO()
    filtered.dataset = {
        "info": copy.deepcopy(coco.dataset.get("info", {})),
        "images": [copy.deepcopy(coco.imgs[i]) for i in categories],
        "categories": [copy.deepcopy(coco.cats[c]) for c in sorted(set(categories.values()))],
        "annotations": [copy.deepcopy(a) for a in coco.anns.values()
                        if categories.get(a["image_id"]) == a["category_id"]]}
    filtered.createIndex()
    return filtered


def resolve_datasets(args):
    """Resolve explicit COCO files or conventional dataset roots, without guessing splits."""
    if args.datasets_config:
        if args.annotations or args.image_root or args.dataset:
            raise ValueError("--datasets-config cannot be combined with --dataset/--annotations/--image-root")
        base = args.datasets_config.resolve().parent
        entries = json.loads(args.datasets_config.read_text(encoding="utf-8"))
        if not isinstance(entries, list) or not entries:
            raise ValueError("Dataset config must be a nonempty JSON list")
    elif args.annotations or args.image_root:
        if args.dataset or not (args.annotations and args.image_root):
            raise ValueError("Use --annotations and --image-root together, without --dataset")
        base = Path.cwd()
        entries = [{"annotations": str(args.annotations), "image_root": str(args.image_root),
                    "split": args.split, "name": args.annotations.stem}]
    elif args.dataset:
        base = Path.cwd()
        entries = [{"root": str(root), "split": args.split} for root in args.dataset]
    else:
        raise ValueError("Specify --dataset, --datasets-config, or --annotations plus --image-root")
    datasets = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError("Every dataset config entry must be an object")
        root = (base / entry["root"]).resolve() if entry.get("root") else None
        split = entry.get("split", args.split)
        if not isinstance(split, str) or not re.fullmatch(r"[\w.-]+", split) or split in (".", ".."):
            raise ValueError(f"Invalid split name: {split}")
        if entry.get("annotations") or entry.get("image_root"):
            if not (entry.get("annotations") and entry.get("image_root")):
                raise ValueError("Explicit annotations and image_root must both be specified")
            annotation_path = (base / entry["annotations"]).resolve()
            image_root = (base / entry["image_root"]).resolve()
        elif root:
            annotation_path = root / "annotations" / f"instances_{split}.json"
            image_root = root / "images" / split
        else:
            raise ValueError("Each dataset needs root or annotations plus image_root")
        if not annotation_path.is_file() or not image_root.is_dir():
            raise ValueError(f"Missing annotations/image directory: {annotation_path}, {image_root}")
        name = str(entry.get("name") or (root.name if root else annotation_path.stem))
        safe_name = re.sub(r"[^\w.-]+", "_", name).strip("._")[:80] or "dataset"
        datasets.append({"name": name, "root": root, "split": split,
                         "annotations": annotation_path, "image_root": image_root,
                         "output_name": f"{index:02d}_{safe_name}"})
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_paths", type=Path, nargs="*", metavar="DATASET_ROOT",
                        help="Dataset root(s); automatically use annotations/instances_test2017.json and images/test2017/")
    parser.add_argument("--dataset", type=Path, action="append", help="COCO root; repeat for multiple datasets")
    parser.add_argument("--split", default="test2017", help="Root layout: annotations/instances_<split>.json, images/<split>/")
    parser.add_argument("--annotations", type=Path, help="Explicit COCO instance JSON (with --image-root)")
    parser.add_argument("--image-root", type=Path, help="Directory relative to which COCO file_name is resolved")
    parser.add_argument("--datasets-config", type=Path, help="JSON list of datasets with individual paths and splits")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parents[1] / "sam3.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Keep predictions above this score for COCO AP (default: all)")
    parser.add_argument("--union-threshold", type=float, default=0.5,
                        help="Score threshold for foreground-union IoU/Dice")
    parser.add_argument("--limit", type=int, default=0, help="First N sorted test images; 0 = full test set")
    parser.add_argument("--prepare-only", action="store_true", help="Validate data and save sampled prompts without inference")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args()
    # Both `script.py ROOT` and the existing `--dataset ROOT` interface work.
    args.dataset = (args.dataset or []) + args.dataset_paths
    if not 0 <= args.score_threshold <= args.union_threshold <= 1 or args.limit < 0:
        parser.error("Require 0 <= score-threshold <= union-threshold <= 1 and limit >= 0")
    if not args.prepare_only and not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    try:
        datasets = resolve_datasets(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output = (args.output or Path("outputs/pcs_baseline") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shared = {}
    summary = []
    for dataset in datasets:
        destination = output / dataset["output_name"] if len(datasets) > 1 else output
        destination.mkdir(exist_ok=True)
        print(f"Dataset: {dataset['name']} ({dataset['split']})", flush=True)
        print(f"Annotations: {dataset['annotations']}\nImages: {dataset['image_root']}", flush=True)
        result = run_dataset(args, dataset, destination, shared)
        summary.append({"name": dataset["name"], "split": dataset["split"],
                        "output": str(destination), **result})
        write_json(output / "datasets_summary.json", summary)


def run_dataset(args, dataset, output, shared):

    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    annotation_path = dataset["annotations"]
    image_root = dataset["image_root"]
    coco = COCO(str(annotation_path))
    # The union metric below has no crowd/ignore-region handling. Fail explicitly
    # instead of silently evaluating a different target set than COCOeval.
    if any(a.get("iscrowd", 0) or a.get("ignore", 0) for a in coco.anns.values()):
        raise ValueError("This baseline requires non-crowd, non-ignored instance annotations")
    queries, skipped = prepare_queries(coco, image_root, args.seed, args.limit)
    if not queries:
        raise ValueError("No eligible images")
    evaluation_gt = query_ground_truth(coco, queries)
    config = {"dataset": str(dataset["root"]) if dataset["root"] else None,
              "dataset_name": dataset["name"], "image_root": str(image_root), "annotations": str(annotation_path),
              "annotations_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
              "checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
              "score_threshold": args.score_threshold, "union_threshold": args.union_threshold,
              "split": dataset["split"], "split_images_total": len(coco.imgs), "evaluated_images": len(queries),
              "limit": args.limit, "skipped": skipped,
              "protocol": "one uniformly sampled nonempty GT instance mask tight axis-aligned box per image; no text; evaluate all instances of the sampled category on that image, including prompted instance",
              "mask_threshold": 0.5, "coco_max_dets": [1, 10, 100],
              "status": "prepared"}
    write_json(output / "config.json", config)
    write_json(output / "prompts.json", queries)
    write_json(output / "evaluation_gt.json", evaluation_gt.dataset)
    print(f"Prepared {len(queries)} queries; skipped {len(skipped)}. Output: {output}", flush=True)
    if args.prepare_only:
        return {"status": "prepared", "images": len(queries), "skipped_images": len(skipped)}

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SAM3 baseline")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    config.update(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  precision="CUDA BF16 autocast", status="running")
    write_json(output / "config.json", config)
    if "processor" not in shared:
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint.resolve()),
                                      load_from_HF=False, device="cuda", enable_inst_interactivity=False)
        shared["processor"] = Sam3Processor(model, device="cuda", confidence_threshold=args.score_threshold)
    processor = shared["processor"]
    predictions, rows = [], []
    totals = {"intersection": 0, "predicted": 0, "target": 0}
    started = time.perf_counter()
    with (output / "per_image.jsonl").open("w", encoding="utf-8") as log:
        for index, query in enumerate(queries, 1):
            image_id = query["image_id"]
            with Image.open(image_root / query["file_name"]) as source:
                # COCO annotations refer to stored pixel coordinates, not EXIF-transposed coordinates.
                image = source.convert("RGB")
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)
                state = processor.add_geometric_prompt(query["prompt_cxcywh_normalized"], True, state)
                masks = state["masks"][:, 0].cpu().numpy()
                boxes = state["boxes"].float().cpu().numpy()
                scores = state["scores"].float().cpu().numpy()
            del state
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            for mask, box, score in zip(masks, boxes, scores):
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                x0, y0, x1, y1 = box.tolist()
                predictions.append({"image_id": image_id, "category_id": query["category_id"],
                                    "score": float(score), "segmentation": rle,
                                    "bbox": [x0, y0, max(0., x1-x0), max(0., y1-y0)]})
            gt_union = np.zeros((image.height, image.width), dtype=bool)
            for ann in evaluation_gt.imgToAnns[image_id]:
                gt_union |= coco.annToMask(ann).astype(bool)
            pred_union = np.zeros_like(gt_union)
            for mask, score in zip(masks, scores):
                if score > args.union_threshold:
                    pred_union |= mask
            counts = {"intersection": int((pred_union & gt_union).sum()),
                      "predicted": int(pred_union.sum()), "target": int(gt_union.sum())}
            for key in totals:
                totals[key] += counts[key]
            row = {**query, "prediction_count": len(scores),
                   "union_prediction_count": int((scores > args.union_threshold).sum()),
                   "inference_seconds": elapsed, **counts, **binary_metrics(**counts)}
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[{index}/{len(queries)}] image={image_id} predictions={len(scores)} IoU={row['iou']:.4f} ({elapsed:.2f}s)", flush=True)
    write_json(output / "predictions.json", predictions)
    metrics = {"images": len(rows), "instances": sum(q["target_instance_count"] for q in queries),
               "coco": evaluate_coco(evaluation_gt, predictions, [q["image_id"] for q in queries], output),
               "foreground_union": {"threshold": args.union_threshold,
                                    "macro_iou": float(np.mean([r["iou"] for r in rows])),
                                    "macro_dice": float(np.mean([r["dice"] for r in rows])),
                                    "micro": binary_metrics(**totals)},
               "inference_seconds_total": sum(r["inference_seconds"] for r in rows),
               "elapsed_seconds_excluding_model_load": time.perf_counter() - started}
    write_json(output / "metrics.json", metrics)
    config["status"] = "complete"
    write_json(output / "config.json", config)
    print(json.dumps(metrics, indent=2), flush=True)
    return {"status": "complete", **metrics}


if __name__ == "__main__":
    main()
