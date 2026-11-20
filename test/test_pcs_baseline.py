"""COCO dataset routing and prompted-category evaluation regression tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

spec = importlib.util.spec_from_file_location(
    "baseline", Path(__file__).resolve().parents[1] / "scripts/eval_pcs_box_baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(dir=self.workspace, prefix="baseline_test_")
        self.root = Path(self.temp.name).resolve()
        assert self.root.parent == self.workspace
        self.args = SimpleNamespace(dataset=None, datasets_config=None, annotations=None,
                                    image_root=None, split="test")

    def tearDown(self):
        self.temp.cleanup()

    def make_dataset(self):
        images = self.root / "images/test"
        images.mkdir(parents=True)
        (self.root / "annotations").mkdir()
        data = {"images": [], "annotations": [],
                "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
        for image_id in (1, 2):
            Image.new("RGB", (20, 20)).save(images / f"{image_id}.png")
            data["images"].append({"id": image_id, "width": 20, "height": 20, "file_name": f"{image_id}.png"})
            for category_id in (1, 2):
                mask = np.zeros((20, 20), dtype=np.uint8)
                x = (category_id - 1) * 10
                mask[2:8, x:x+6] = 1
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode()
                data["annotations"].append({"id": image_id*10+category_id, "image_id": image_id,
                                            "category_id": category_id, "segmentation": rle,
                                            "bbox": [x, 2, 6, 6], "area": 36, "iscrowd": 0})
        path = self.root / "annotations/instances_test.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path, images

    def test_roots_and_explicit_paths(self):
        annotation, images = self.make_dataset()
        self.args.dataset = [self.root, self.root]
        entries = baseline.resolve_datasets(self.args)
        self.assertEqual(entries[0]["annotations"], annotation)
        self.assertNotEqual(entries[0]["output_name"], entries[1]["output_name"])
        self.args.dataset = None
        self.args.annotations, self.args.image_root = annotation, images
        self.assertEqual(baseline.resolve_datasets(self.args)[0]["image_root"], images)

    def test_config_relative_paths_and_missing_source(self):
        with self.assertRaises(ValueError):
            baseline.resolve_datasets(self.args)
        annotation, images = self.make_dataset()
        config = self.root / "datasets.json"
        config.write_text(json.dumps([{"name": "../custom", "annotations": "annotations/instances_test.json",
                                      "image_root": "images/test", "split": "test"}]))
        self.args.datasets_config = config
        entry = baseline.resolve_datasets(self.args)[0]
        self.assertEqual(entry["annotations"], annotation)
        self.assertNotIn("/", entry["output_name"])

    def test_multicategory_query_targets_and_metrics(self):
        annotation, images = self.make_dataset()
        coco = COCO(str(annotation))
        queries, skipped = baseline.prepare_queries(coco, images, 42, 0)
        partial, _ = baseline.prepare_queries(coco, images, 42, 1)
        self.assertEqual(partial[0], queries[0])
        self.assertFalse(skipped)
        self.assertTrue(all(q["target_instance_count"] == 1 for q in queries))
        filtered = baseline.query_ground_truth(coco, queries)
        self.assertEqual(len(filtered.anns), 2)
        for query in queries:
            self.assertEqual(filtered.imgToAnns[query["image_id"]][0]["category_id"], query["category_id"])
        predictions = [{"image_id": a["image_id"], "category_id": a["category_id"],
                        "bbox": a["bbox"], "segmentation": a["segmentation"], "score": 1.}
                       for a in filtered.anns.values()]
        metrics = baseline.evaluate_coco(filtered, predictions, [1, 2], self.root)
        self.assertAlmostEqual(metrics["segm"]["AP"], 1.)
        self.assertAlmostEqual(metrics["bbox"]["AP"], 1.)
        empty = baseline.evaluate_coco(filtered, [], [1, 2], self.root)
        self.assertEqual(empty["segm"]["AP"], 0.)


if __name__ == "__main__":
    unittest.main()
