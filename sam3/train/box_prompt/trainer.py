"""Native trainer with an audited front-end-only optimizer and resume contract."""
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch

from sam3.train.trainer import Trainer, unwrap_ddp_if_wrapped
from sam3.train.utils.distributed import all_gather
from .model import CONTRACT, MODULES, build_optimizer, validate_trainable_parameters


def capture_rng():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return {"numpy": (algorithm, keys.tolist(), position, has_gauss, cached),
            "python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    algorithm, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


class BoxPromptTrainer(Trainer):
    def _construct_optimizers(self):
        self.optim = build_optimizer(self.model, self.optim_conf)
        names = validate_trainable_parameters(self.model)
        if self.distributed_rank == 0:
            root = Path(self.checkpoint_conf.save_dir).parent
            root.mkdir(parents=True, exist_ok=True)
            params = dict(self.model.named_parameters())
            report = {"training_contract": CONTRACT,
                      "trainable_numel": sum(params[n].numel() for n in names),
                      "parameters": {n: list(params[n].shape) for n in names}}
            (root / "trainable_parameters.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            logging.info("Experiment B: %d trainable parameters in %d tensors", report["trainable_numel"], len(names))
        if self.mode != "val":
            self._check_first_update()

    def _check_first_update(self):
        params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        before = {}

        def pre_step(optimizer, args, kwargs):
            for module in MODULES:
                gradients = [p.grad for n, p in params.items() if n.startswith(f"geometry_encoder.{module}.") and p.grad is not None]
                if not gradients or not all(g.isfinite().all().item() for g in gradients) or not any(g.count_nonzero().item() for g in gradients):
                    raise RuntimeError(f"No finite nonzero gradient reached {module}")
            before.update({n: p.detach().clone() for n, p in params.items()})

        def post_step(optimizer, args, kwargs):
            changed = [n for n, p in params.items() if not torch.equal(p, before[n])]
            for module in MODULES:
                if not any(n.startswith(f"geometry_encoder.{module}.") for n in changed):
                    raise RuntimeError(f"First AdamW update did not change {module}")
            logging.info("Experiment B first update verified all four modules (%d tensors changed)", len(changed))
            before.clear()
            pre.remove()
            post.remove()

        pre = self.optim.optimizer.register_step_pre_hook(pre_step)
        post = self.optim.optimizer.register_step_post_hook(post_step)

    def _optimizer_names(self):
        names = {id(p): n for n, p in unwrap_ddp_if_wrapped(self.model).named_parameters()}
        return [[names[id(p)] for p in group["params"]] for group in self.optim.optimizer.param_groups]

    def save_checkpoint(self, epoch, checkpoint_names=None):
        if self.skip_saving_ckpts:
            return
        self._rng_by_rank = all_gather(capture_rng(), force_cpu=True)
        super().save_checkpoint(epoch, checkpoint_names)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint.update(training_contract=CONTRACT, optimizer_names=self._optimizer_names(),
                          rng_by_rank=self._rng_by_rank)
        super()._save_checkpoint(checkpoint, checkpoint_path)

    def load_checkpoint(self):
        if self.model_conf.initialization == "resume":
            self._load_resuming_checkpoint(str(Path(self.checkpoint_conf.save_dir) / "checkpoint.pt"))
        else:
            self._init_model_state()

    def _load_resuming_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("training_contract") != CONTRACT or checkpoint.get("optimizer_names") != self._optimizer_names():
            raise ValueError("Resume checkpoint does not match experiment B optimizer contract")
        if checkpoint["epoch"] >= self.max_epochs:
            raise ValueError("--epochs must exceed the number of completed epochs")
        if len(checkpoint["rng_by_rank"]) != torch.distributed.get_world_size():
            raise ValueError("Resume requires the original GPU count")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch, self.steps = checkpoint["epoch"], checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)
        self.best_meter_values = checkpoint.get("best_meter_values", {})
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng_by_rank"][self.distributed_rank])
