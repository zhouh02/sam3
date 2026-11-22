"""Run with python -m unittest discover -s test -p test_box_prompt_training.py."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pycocotools import mask as mask_util
import torch
from torch import nn

from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import _create_geometry_encoder
from sam3.train.box_prompt.data import BoxPromptCOCO, audit_and_write_gt
from sam3.train.box_prompt.model import MODULES, build_optimizer, freeze_for_box_training, load_official

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_box", ROOT / "scripts/train_box_prompt.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def make_dataset(root):
    annotations = []
    # IDs deliberately out of order: equal-area ties must use source order.
    for identity, cat, x in ((91, 3, 2), (12, 3, 10), (7, 6, 18), (4, 6, 24), (5, 9, 1)):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[4:12, x:x + 4] = 1
        rle = mask_util.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode()
        annotations.append({"id": identity, "category_id": cat, "image_id": 1,
                            "bbox": [x, 4, 5, 9], "area": 32, "iscrowd": 0, "segmentation": rle})
    data = {"images": [{"id": 1, "file_name": "sample.png", "height": 32, "width": 32}],
            "annotations": annotations, "categories": [{"id": c, "name": f"line{c}"} for c in (3, 6, 9)]}
    Image.new("RGB", (32, 32), (80, 120, 40)).save(root / "sample.png")
    path = root / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class BoxTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "outputs").mkdir(exist_ok=True)

    def test_sampling_collation_and_eval_population(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            root = Path(directory)
            path = make_dataset(root)
            loader = BoxPromptCOCO(str(path), prompt_sampling_policy="median_area")
            queries, anns = loader.loadQueriesAndAnnotationsFromDatapoint(0)
            self.assertEqual([q["prompt_annotation_id"] for q in queries], [12, 4])
            self.assertEqual([len(q["object_ids_output"]) for q in queries], [2, 2])
            self.assertEqual(len(anns), 4)
            torch.testing.assert_close(queries[0]["input_box"], torch.tensor([[10, 4, 4, 8]]) / 32)
            report = audit_and_write_gt(path, root, root / "gt.json")
            self.assertEqual(report["targets"], 4)
            gt = json.loads((root / "gt.json").read_text())
            self.assertEqual({a["category_id"] for a in gt["annotations"]}, {3, 6})
            cfg = launcher.configuration(launcher.arguments(["train", "--data-root", str(root)]))
            ds = cfg.trainer.data.train.dataset
            ds.img_folder, ds.ann_file = str(root), str(path)
            cfg.scratch.resolution = 32
            item = instantiate(ds)[0]
            batch = instantiate(cfg.trainer.data.train.collate_fn)([item])["all"]
            self.assertEqual(batch.find_inputs[0].input_boxes.shape, (1, 2, 4))
            self.assertEqual(batch.find_targets[0].num_boxes.tolist(), [2, 2])
            self.assertFalse(batch.find_inputs[0].input_boxes_mask.any())
            self.assertEqual(set(batch.find_text_batch), {"line3", "line6"})

    def test_gradients_cross_frozen_fusion_and_only_frontend_updates(self):
        torch.manual_seed(7)
        model = nn.Module()
        model.geometry_encoder = _create_geometry_encoder()
        model.readout = nn.Linear(256, 3)
        freeze_for_box_training(model)
        cfg = OmegaConf.load(ROOT / "sam3/train/box_prompt/config.yaml")
        optim = build_optimizer(model, cfg.trainer.optim)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        prompt = Prompt(box_embeddings=torch.tensor([[[.5, .5, .5, .5]]]))
        token, _ = model.geometry_encoder(prompt, [torch.randn(64, 1, 256)], [(8, 8)])
        loss = model.readout(token).square().mean()
        loss.backward()
        for module in MODULES:
            params = getattr(model.geometry_encoder, module).parameters()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in params), module)
        optim.step_schedulers(0, step=0)
        optim.optimizer.step()
        for n, p in model.named_parameters():
            if not p.requires_grad:
                self.assertTrue(torch.equal(p, before[n]), n)
        for module in MODULES:
            self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                                if n.startswith(f"geometry_encoder.{module}.")), module)

    def test_resume_keeps_protocol_and_extends_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            cfg = launcher.configuration(launcher.arguments(["train", "--data-root", directory,
                  "--output", directory, "--lr", "0.00003", "--epochs", "3"]))
            OmegaConf.save(cfg, Path(directory) / "config.yaml")
            resumed = launcher.configuration(launcher.arguments(["resume", "--output", directory, "--epochs", "6"]))
            self.assertEqual(resumed.trainer.max_epochs, 6)
            self.assertEqual(resumed.scratch.stage1_lr, 0.00003)
            self.assertEqual(resumed.trainer.model.initialization, "resume")

    def test_official_loading_rejects_missing_frozen_weights(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as directory:
            model = nn.Linear(2, 3)
            checkpoint = Path(directory) / "official.pt"
            torch.save({"model": {"detector.weight": model.weight}}, checkpoint)
            with self.assertRaisesRegex(ValueError, "missing"):
                load_official(model, checkpoint)


if __name__ == "__main__":
    unittest.main()
