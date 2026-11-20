"""Regression tests for multi-category visual prompts and official AP+ semantics."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("single_box", REPO / "scripts/eval_coco_single_box.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)


def dataset():
    coco = COCO()
    anns = []
    for ann_id, image_id, category_id in ((1, 1, 1), (2, 1, 1), (3, 1, 2), (4, 2, 2)):
        mask = np.zeros((32, 32), dtype=np.uint8)
        x = ann_id * 4
        mask[2:6, x:x+3] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode()
        anns.append({"id": ann_id, "image_id": image_id, "category_id": category_id,
                     "bbox": [x+0.25, 2.5, 3.0, 4.0], "area": 12, "iscrowd": 0, "segmentation": rle})
    coco.dataset = {"info": {}, "images": [{"id": i, "file_name": f"{i}.jpg", "width": 32, "height": 32}
                                           for i in (1, 2)],
                    "categories": [{"id": i, "name": str(i)} for i in (1, 2)], "annotations": anns}
    coco.createIndex()
    return coco


class SingleBoxTests(unittest.TestCase):
    def test_pairs_float_boxes_and_replay(self):
        coco = dataset()
        queries = script.prepare_queries(coco, [1, 2], 42)
        self.assertEqual([(q["image_id"], q["category_id"]) for q in queries], [(1, 1), (1, 2), (2, 2)])
        self.assertEqual(queries[0]["target_instance_count"], 2)
        for query in queries:
            self.assertEqual(query["prompt_xywh"], coco.anns[query["annotation_id"]]["bbox"])
        self.assertEqual(queries, script.prepare_queries(coco, [2, 1], 999, saved=queries))
        self.assertEqual(queries[:2], script.prepare_queries(coco, [1], 42))
        with self.assertRaises(ValueError):
            script.prepare_queries(coco, [1, 2], 42, saved=queries[:2])
        bad = copy.deepcopy(queries)
        bad[0]["prompt_xywh"][0] += 1
        with self.assertRaises(ValueError):
            script.prepare_queries(coco, [1, 2], 42, saved=bad)

    def test_crowd_and_clipping(self):
        coco = dataset()
        coco.anns[3]["iscrowd"] = 1
        coco.anns[3]["bbox"] = [-1.5, 2, 5, 4]
        queries = script.prepare_queries(coco, [1], 42)
        self.assertTrue(queries[1]["prompt_iscrowd"])
        self.assertEqual(queries[1]["prompt_xyxy_clipped"], [0, 2, 3.5, 6])

    def test_official_positive_ap_and_empty_predictions(self):
        coco = dataset()
        predictions = [{"image_id": a["image_id"], "category_id": a["category_id"],
                        "bbox": a["bbox"], "segmentation": a["segmentation"], "score": 0.9}
                       for a in coco.anns.values()]
        # High-confidence detection of absent class 1 on image 2 must be ignored by AP+.
        negative = copy.deepcopy(predictions[0])
        negative.update(image_id=2, score=0.99)
        predictions.append(negative)
        with tempfile.TemporaryDirectory(dir=REPO) as tmp:
            metrics = script.evaluate(coco, predictions, [1, 2], ["bbox", "segm"], Path(tmp))
            self.assertAlmostEqual(metrics["bbox"]["AP"], 1.)
            self.assertAlmostEqual(metrics["segm"]["AP"], 1.)
            ordinary = COCOeval(coco, coco.loadRes(copy.deepcopy(predictions)), "bbox")
            ordinary.evaluate()
            ordinary.accumulate()
            ordinary.summarize()
            self.assertLess(ordinary.stats[0], metrics["bbox"]["AP"])
            empty = script.evaluate(coco, [], [1, 2], ["bbox", "segm"], Path(tmp))
            self.assertEqual(empty["bbox"]["AP"], 0.)
            self.assertEqual(empty["segm"]["AP"], 0.)

    def test_cap_is_across_categories(self):
        rows = [{"category_id": c, "score": s} for c, s in [(1, .9), (1, .7), (2, .8), (2, .1)]]
        self.assertEqual([r["score"] for r in script.top_predictions(rows, 2)], [.9, .8])

    def test_inference_encodes_once_and_resets_every_query(self):
        import torch

        class Processor:
            encoded = 0
            resets = 0

            def set_image(self, image):
                self.encoded += 1
                return {"prompts": []}

            def reset_all_prompts(self, state):
                self.resets += 1
                state["prompts"] = []

            def add_geometric_prompt(self, box, positive, state):
                assert positive and not state["prompts"]
                state["prompts"].append(box)
                state.update(scores=torch.tensor([.9]), boxes=torch.tensor([[1., 2., 4., 6.]]))
                return state

        processor = Processor()
        queries = script.prepare_queries(dataset(), [1], 42)
        logs = []
        rows = script.infer_image(processor, None, queries, False, 100, logs.append)
        self.assertEqual(processor.encoded, 1)
        self.assertEqual(processor.resets, 2)
        self.assertEqual({r["category_id"] for r in rows}, {1, 2})
        self.assertEqual(len(logs), 2)


if __name__ == "__main__":
    unittest.main()
