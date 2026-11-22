"""Enforce the experiment B parameter and checkpoint boundaries."""
import logging

import torch
from torch import nn

MODULES = ("boxes_direct_project", "boxes_pool_project", "boxes_pos_enc_project", "label_embed")
PREFIXES = tuple(f"geometry_encoder.{name}." for name in MODULES)
CONTRACT = "box-pcs/frontend-only/v1"


def trainable_frontend(model):
    return nn.ModuleDict({name: getattr(model.geometry_encoder, name) for name in MODULES})


def freeze_for_box_training(model):
    frontend = trainable_frontend(model)
    if any(module is None for module in frontend.values()):
        raise ValueError("Experiment B requires all three native box projections and label embedding")
    model.requires_grad_(False)
    frontend.requires_grad_(True)
    validate_trainable_parameters(model)


def validate_trainable_parameters(model):
    expected = {name for name, _ in model.named_parameters() if name.startswith(PREFIXES)}
    actual = {name for name, p in model.named_parameters() if p.requires_grad}
    if not expected or actual != expected:
        raise RuntimeError(f"Experiment B trainable boundary mismatch: {actual ^ expected}")
    return sorted(actual)


def build_optimizer(model, conf):
    from sam3.train.optim.optimizer import construct_optimizer

    validate_trainable_parameters(model)
    optim = construct_optimizer(trainable_frontend(model), conf.optimizer, conf.options)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optim.optimizer.param_groups for p in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError("Optimizer must contain each box front-end parameter exactly once")
    return optim


def load_official(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    detector = {k.removeprefix("detector."): v for k, v in state.items() if k.startswith("detector.")}
    if not detector:
        raise ValueError("Expected the official SAM3 checkpoint with detector.* weights")
    incompatible = model.load_state_dict(detector, strict=False)
    # Official weights also include SAM2 pyramid convolutions unused by PCS.
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("backbone.vision_backbone.sam2_convs.")]
    if incompatible.missing_keys or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={unexpected}")


def build_model(checkpoint_path, bpe_path, initialization="official", device="cpu", eval_mode=False):
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(device=device, bpe_path=bpe_path, load_from_HF=False,
                                  eval_mode=eval_mode, enable_segmentation=True)
    if initialization == "official":
        load_official(model, checkpoint_path)
    elif initialization == "trained":
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state.get("training_contract") != CONTRACT:
            raise ValueError("Expected an experiment B checkpoint")
        model.load_state_dict(state["model"], strict=True)
    elif initialization != "resume":
        raise ValueError(f"Unknown initialization: {initialization}")
    freeze_for_box_training(model)
    # Only the backbones can skip autograd. Frozen fusion layers must pass gradients.
    model.backbone.forward_image = torch.no_grad()(model.backbone.forward_image)
    model.backbone.forward_text = torch.no_grad()(model.backbone.forward_text)
    logging.info("Experiment B trainable parameters: %s", validate_trainable_parameters(model))
    return model
