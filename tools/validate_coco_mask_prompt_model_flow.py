"""Validate the real COCO mask-prompt path through the SAM3 image model."""

from __future__ import annotations

import argparse
import copy
import functools
import json
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.coco_json_loaders import COCO_FROM_JSON
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Sam3ImageDataset
from sam3.train.matcher import BinaryHungarianMatcherV2
from sam3.train.transforms.basic import get_random_resize_max_size
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from sam3.train.transforms.segmentation import DecodeRle


RESOLUTION = 1008
MASK_ENCODER_PREFIX = "geometry_encoder.mask_prompt_encoder."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann_file", type=Path, required=True)
    parser.add_argument("--img_folder", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("./sam3.pt"))
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("./debug_outputs/coco_mask_prompt_model_flow/summary.json"),
    )
    args = parser.parse_args()
    if args.num_samples < 8:
        parser.error("--num_samples must be at least 8 for this validation")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_validation_dataset(args: argparse.Namespace) -> Sam3ImageDataset:
    loader = functools.partial(
        COCO_FROM_JSON,
        use_mask_prompt=True,
        min_instances_per_prompt_category=2,
        prompt_sampling_policy="median_area",
        fixed_val_prompt_policy="median_area",
        include_negatives=False,
        category_chunk_size=2,
    )
    transforms = [
        ComposeAPI(
            [
                DecodeRle(),
                RandomResizeAPI(
                    sizes=RESOLUTION,
                    max_size=get_random_resize_max_size(RESOLUTION),
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
    ]
    dataset = Sam3ImageDataset(
        img_folder=str(args.img_folder),
        ann_file=str(args.ann_file),
        transforms=transforms,
        max_ann_per_img=100000,
        multiplier=1,
        training=False,
        load_segmentation=True,
        max_val_queries=100000,
        use_caching=False,
        coco_json_loader=loader,
    )
    dataset.set_curr_epoch(0)
    return dataset


def validate_pre_collator_datapoint(datapoint: Any) -> int:
    checked = 0
    for query in datapoint.find_queries:
        if query.input_mask is None:
            continue
        if query.input_mask_object_id not in query.object_ids_output:
            raise AssertionError(
                "input_mask_object_id is not present in object_ids_output: "
                f"{query.input_mask_object_id=} {query.object_ids_output=}"
            )
        if int(query.input_mask_label) != 1:
            raise AssertionError(
                f"Expected positive mask label 1, got {query.input_mask_label}"
            )
        if not torch.is_tensor(query.input_mask) or query.input_mask.sum() <= 0:
            raise AssertionError(
                "Transformed query.input_mask is empty or is not a tensor"
            )
        checked += 1
    if checked == 0:
        raise AssertionError("Datapoint contains no transformed mask prompt query")
    return checked


def collect_datapoints(dataset: Sam3ImageDataset, requested_queries: int) -> list[Any]:
    datapoints = []
    query_count = 0
    for index in range(len(dataset)):
        datapoint = dataset[index]
        query_count += validate_pre_collator_datapoint(datapoint)
        datapoints.append(datapoint)
        if query_count >= requested_queries:
            break
    if query_count < requested_queries:
        raise AssertionError(
            f"Only found {query_count} eligible COCO mask queries; "
            f"requested {requested_queries}"
        )
    print(
        f"Selected {len(datapoints)} transformed COCO datapoints containing "
        f"{query_count} mask-prompt queries."
    )
    return datapoints


def validate_collated_batch(batch: Any) -> dict[str, Any]:
    if len(batch.find_inputs) != 1:
        raise AssertionError(f"Expected one find stage, got {len(batch.find_inputs)}")
    stage = batch.find_inputs[0]
    masks = stage.input_masks
    if masks.ndim != 5 or masks.shape[2] != 1:
        raise AssertionError(f"Expected masks [N,B,1,H,W], got {tuple(masks.shape)}")
    if tuple(masks.shape[-2:]) != tuple(batch.img_batch.shape[-2:]):
        raise AssertionError(
            f"Mask/image spatial mismatch: {tuple(masks.shape[-2:])} vs "
            f"{tuple(batch.img_batch.shape[-2:])}"
        )
    expected_labels = (masks.shape[0], masks.shape[1])
    expected_padding = (masks.shape[1], masks.shape[0])
    if tuple(stage.input_masks_label.shape) != expected_labels:
        raise AssertionError(
            f"Mask label shape mismatch: {tuple(stage.input_masks_label.shape)} "
            f"vs {expected_labels}"
        )
    if tuple(stage.input_masks_mask.shape) != expected_padding:
        raise AssertionError(
            f"Mask padding shape mismatch: {tuple(stage.input_masks_mask.shape)} "
            f"vs {expected_padding}"
        )

    valid = ~stage.input_masks_mask.transpose(0, 1)
    valid_masks = masks[valid]
    if valid_masks.numel() == 0 or torch.any(valid_masks.flatten(1).sum(-1) <= 0):
        raise AssertionError("At least one valid mask prompt is empty")
    if not torch.isfinite(masks.float()).all():
        raise AssertionError("Mask prompt tensor contains NaN or Inf")
    unique_values = set(torch.unique(valid_masks).cpu().tolist())
    if not unique_values.issubset({0, 1}):
        raise AssertionError(f"Mask prompts are not binary: {sorted(unique_values)}")
    return {
        "image_shape": list(batch.img_batch.shape),
        "mask_shape": list(masks.shape),
        "valid_mask_count": int(valid.sum()),
        "mask_unique_values": sorted(unique_values),
    }


def preprocess_checkpoint(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    return {
        key.replace("detector.", ""): value
        for key, value in checkpoint.items()
        if "detector" in key
    }


def inspect_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    state_dict = preprocess_checkpoint(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    mask_missing = [key for key in missing if key.startswith(MASK_ENCODER_PREFIX)]
    other_missing = [key for key in missing if not key.startswith(MASK_ENCODER_PREFIX)]
    if other_missing:
        raise AssertionError(
            f"Non-mask modules are missing checkpoint keys: {other_missing}"
        )
    if len(mask_missing) != len(missing):
        raise AssertionError("Not all newly missing keys belong to MaskPromptEncoder")
    baseline_unexpected = [
        key
        for key in unexpected
        if key.startswith("backbone.vision_backbone.sam2_convs.")
    ]
    nonbaseline_unexpected = [
        key for key in unexpected if key not in baseline_unexpected
    ]
    if nonbaseline_unexpected:
        raise AssertionError(
            f"Unexpected keys outside baseline sam2_convs: {nonbaseline_unexpected}"
        )
    print(
        f"Checkpoint: {len(missing)} missing keys (all MaskPromptEncoder), "
        f"{len(unexpected)} baseline-existing unexpected sam2_convs keys."
    )
    return {
        "missing_keys": list(missing),
        "mask_prompt_encoder_missing_keys": mask_missing,
        "unexpected_keys": list(unexpected),
        "baseline_existing_unexpected_keys": baseline_unexpected,
    }


def iter_named_tensors(value: Any, prefix: str = "output"):
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, SAM3Output):
        yield from iter_named_tensors(value.output, f"{prefix}.output")
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from iter_named_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from iter_named_tensors(child, f"{prefix}[{index}]")


def summarize_output(output: SAM3Output) -> tuple[dict[str, Any], bool]:
    if not isinstance(output, SAM3Output):
        raise AssertionError(f"Expected SAM3Output, got {type(output).__name__}")
    summaries = {}
    all_finite = True
    for name, tensor in iter_named_tensors(output):
        if not torch.is_floating_point(tensor):
            continue
        detached = tensor.detach().float()
        finite = bool(torch.isfinite(detached).all())
        all_finite &= finite
        if any(key in name for key in ("pred_logits", "pred_boxes", "pred_masks")):
            summaries[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "min": float(detached.min().cpu()),
                "max": float(detached.max().cpu()),
                "mean": float(detached.mean().cpu()),
                "finite": finite,
            }
    if not summaries:
        raise AssertionError("SAM3Output contained no prediction tensors to summarize")
    if not all_finite:
        raise AssertionError("SAM3Output contains NaN or Inf")
    return summaries, all_finite


class FlowRecorder:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.mask_prompt_encoder_calls = 0
        self.box_encoder_calls = 0
        self.dense_mask_encoder_calls = 0
        self.mask_prompt_method_calls = 0
        self.mask_token_shapes = []
        self.mask_tokens = []
        self.final_proj_input_shapes = []
        self.shared_calls = {
            "final_proj": 0,
            "norm": 0,
            "transformer": 0,
            "encode_norm": 0,
        }

    def _mask_forward_hook(self, _module, _inputs, output) -> None:
        self.mask_prompt_encoder_calls += 1
        self.mask_token_shapes.append(list(output.shape))
        self.mask_tokens.append(output.detach().cpu())

    def _final_proj_hook(self, _module, inputs, _output) -> None:
        self.shared_calls["final_proj"] += 1
        self.final_proj_input_shapes.append(list(inputs[0].shape))

    def __enter__(self):
        geometry = self.model.geometry_encoder
        self.stack = ExitStack()
        self.stack.enter_context(
            mock.patch.object(
                geometry,
                "_encode_boxes",
                wraps=self._counted("box_encoder_calls", geometry._encode_boxes),
            )
        )
        self.stack.enter_context(
            mock.patch.object(
                geometry,
                "_encode_masks",
                wraps=self._counted("dense_mask_encoder_calls", geometry._encode_masks),
            )
        )
        self.stack.enter_context(
            mock.patch.object(
                geometry,
                "_encode_mask_prompts",
                wraps=self._counted(
                    "mask_prompt_method_calls", geometry._encode_mask_prompts
                ),
            )
        )
        self.handles = [
            geometry.mask_prompt_encoder.register_forward_hook(self._mask_forward_hook),
            geometry.final_proj.register_forward_hook(self._final_proj_hook),
            geometry.norm.register_forward_hook(
                lambda *_: self._increment_shared("norm")
            ),
            geometry.encode[0].register_forward_hook(
                lambda *_: self._increment_shared("transformer")
            ),
            geometry.encode_norm.register_forward_hook(
                lambda *_: self._increment_shared("encode_norm")
            ),
        ]
        return self

    def _counted(self, attribute: str, function):
        def wrapper(*args, **kwargs):
            setattr(self, attribute, getattr(self, attribute) + 1)
            return function(*args, **kwargs)

        return wrapper

    def _increment_shared(self, name: str) -> None:
        self.shared_calls[name] += 1

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        return self.stack.__exit__(exc_type, exc_value, traceback)


def validate_mask_mode_recording(recorder: FlowRecorder, stage: Any) -> None:
    if recorder.mask_prompt_encoder_calls != 1:
        raise AssertionError(
            f"Expected one MaskPromptEncoder call, got {recorder.mask_prompt_encoder_calls}"
        )
    if recorder.mask_prompt_method_calls != 1:
        raise AssertionError(
            f"Expected one _encode_mask_prompts call, got {recorder.mask_prompt_method_calls}"
        )
    if recorder.box_encoder_calls != 0 or recorder.dense_mask_encoder_calls != 0:
        raise AssertionError(
            "Mask mode called an incompatible encoder: "
            f"boxes={recorder.box_encoder_calls}, dense={recorder.dense_mask_encoder_calls}"
        )
    expected = [stage.input_masks.shape[0], stage.input_masks.shape[1], 256]
    if recorder.mask_token_shapes != [expected]:
        raise AssertionError(
            f"Expected one mask-token shape {expected}, got {recorder.mask_token_shapes}"
        )
    if not torch.isfinite(recorder.mask_tokens[0]).all():
        raise AssertionError("MaskPromptEncoder output contains NaN or Inf")
    if not all(count > 0 for count in recorder.shared_calls.values()):
        raise AssertionError(
            f"Mask tokens skipped shared geometry path: {recorder.shared_calls}"
        )
    points = stage.input_points.shape[1] if stage.input_points is not None else 0
    expected_sequence = points + stage.input_masks.shape[0] + 1
    if recorder.final_proj_input_shapes[-1][0] != expected_sequence:
        raise AssertionError(
            f"final_proj sequence length {recorder.final_proj_input_shapes[-1][0]} "
            f"does not match points + masks + CLS = {expected_sequence}"
        )


def forward_mask_batch(
    model: torch.nn.Module,
    datapoints: list[Any],
    device: torch.device,
) -> tuple[Any, SAM3Output, FlowRecorder, dict[str, Any]]:
    batch = collate_fn_api(datapoints, dict_key="validation")["validation"]
    batch_info = validate_collated_batch(batch)
    stage = batch.find_inputs[0]
    recorder = FlowRecorder(model)
    with recorder, torch.no_grad():
        output = model(copy_data_to_device(batch, device, non_blocking=True))
    validate_mask_mode_recording(recorder, stage)
    output_stats, all_finite = summarize_output(output)
    batch_info.update(
        {
            "mask_token_shape": recorder.mask_token_shapes[0],
            "final_proj_input_shape": recorder.final_proj_input_shapes[-1],
            "shared_geometry_calls": recorder.shared_calls,
            "output_tensors": output_stats,
            "all_outputs_finite": all_finite,
        }
    )
    return batch, output, recorder, batch_info


def find_prompt_pair(datapoints: list[Any]) -> tuple[Any, Any]:
    for datapoint in datapoints:
        for query_index, query in enumerate(datapoint.find_queries):
            if query.input_mask is None or len(query.object_ids_output) < 2:
                continue
            alternatives = [
                object_index
                for object_index in query.object_ids_output
                if object_index != query.input_mask_object_id
                and datapoint.images[query.image_id].objects[object_index].segment
                is not None
                and datapoint.images[query.image_id].objects[object_index].segment.sum()
                > 0
            ]
            if not alternatives:
                continue
            alternate_index = alternatives[0]
            version_a = copy.deepcopy(datapoint)
            version_b = copy.deepcopy(datapoint)
            query_b = version_b.find_queries[query_index]
            query_b.input_mask = (
                version_b.images[query_b.image_id]
                .objects[alternate_index]
                .segment.clone()
            )
            query_b.input_mask_label = 1
            query_b.input_mask_object_id = alternate_index
            version_a.find_queries = [version_a.find_queries[query_index]]
            version_b.find_queries = [query_b]
            validate_pre_collator_datapoint(version_a)
            validate_pre_collator_datapoint(version_b)
            return version_a, version_b
    raise AssertionError(
        "Could not find two same-category target masks in selected data"
    )


def sensitivity_check(
    model: torch.nn.Module,
    pair: tuple[Any, Any],
    device: torch.device,
) -> dict[str, Any]:
    tokens = []
    outputs = []
    for datapoint in (pair[0], pair[0], pair[1]):
        _, output, recorder, _ = forward_mask_batch(model, [datapoint], device)
        tokens.append(recorder.mask_tokens[0])
        outputs.append(output)
    deterministic = bool(torch.allclose(tokens[0], tokens[1]))
    different = not bool(torch.allclose(tokens[0], tokens[2]))
    if not deterministic:
        raise AssertionError("Repeated identical mask prompt produced different tokens")
    if not different:
        raise AssertionError("Different real COCO masks produced identical mask tokens")

    first_pred = next(
        tensor.detach().float().cpu()
        for name, tensor in iter_named_tensors(outputs[0])
        if "pred_logits" in name
    )
    other_pred = next(
        tensor.detach().float().cpu()
        for name, tensor in iter_named_tensors(outputs[2])
        if "pred_logits" in name
    )
    return {
        "same_prompt_deterministic": deterministic,
        "different_prompts_produce_different_tokens": different,
        "token_max_abs_difference": float((tokens[0] - tokens[2]).abs().max()),
        "output_pred_logits_max_abs_difference": float(
            (first_pred - other_pred).abs().max()
        ),
    }


def get_probe_tensor(output: SAM3Output) -> tuple[str, torch.Tensor]:
    candidates = [
        (name, tensor)
        for name, tensor in iter_named_tensors(output)
        if torch.is_floating_point(tensor)
        and ("pred_logits" in name or "pred_masks" in name)
    ]
    if not candidates:
        raise AssertionError(
            "No continuous prediction tensor is available for backward"
        )
    return candidates[0]


def backward_check(
    model: torch.nn.Module,
    datapoint: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    if model.matcher is None:
        model.matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    for parameter in model.parameters():
        parameter.requires_grad = False
        parameter.grad = None
    mask_encoder = model.geometry_encoder.mask_prompt_encoder
    for parameter in mask_encoder.parameters():
        parameter.requires_grad = True

    batch = collate_fn_api([datapoint], dict_key="backward")["backward"]
    validate_collated_batch(batch)
    original_forward_image = model.backbone.forward_image
    original_forward_text = model.backbone.forward_text

    def frozen_forward_image(*args, **kwargs):
        with torch.no_grad():
            return original_forward_image(*args, **kwargs)

    def frozen_forward_text(*args, **kwargs):
        with torch.no_grad():
            return original_forward_text(*args, **kwargs)

    # The repository's inference-only fused ViT MLP raises whenever global
    # autograd is enabled. Keep the frozen backbone under no_grad while the
    # actual model output remains differentiable through MaskPromptEncoder and
    # the shared geometry/decoder path.
    with mock.patch.object(
        model.backbone, "forward_image", side_effect=frozen_forward_image
    ), mock.patch.object(
        model.backbone, "forward_text", side_effect=frozen_forward_text
    ):
        output = model(copy_data_to_device(batch, device, non_blocking=True))
    probe_name, probe_tensor = get_probe_tensor(output)
    probe_loss = probe_tensor.float().mean()
    probe_loss.backward()

    branch_prefixes = {
        "visual": "mask_visual_project",
        "shape": "mask_shape_",
        "position": "mask_pos_enc_project",
        "label": "mask_label_embed",
    }
    grad_details = {}
    nonzero_names = []
    for name, parameter in mask_encoder.named_parameters():
        grad = parameter.grad
        finite = grad is not None and bool(torch.isfinite(grad).all())
        grad_sum = 0.0 if grad is None else float(grad.detach().abs().sum().cpu())
        nonzero = finite and grad_sum > 0
        if nonzero:
            nonzero_names.append(name)
        grad_details[name] = {
            "present": grad is not None,
            "finite": finite,
            "abs_sum": grad_sum,
            "nonzero": nonzero,
        }
    branch_status = {
        branch: any(
            details["nonzero"]
            for name, details in grad_details.items()
            if name.startswith(prefix)
        )
        for branch, prefix in branch_prefixes.items()
    }
    frozen_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    connected = bool(nonzero_names)
    if not connected:
        raise AssertionError(
            "No MaskPromptEncoder parameter received a nonzero gradient"
        )
    if frozen_with_grad:
        raise AssertionError(
            f"Frozen parameters unexpectedly have gradients: {frozen_with_grad}"
        )
    model.eval()
    return {
        "backward_grad_connected": connected,
        "probe_output": probe_name,
        "probe_loss": float(probe_loss.detach().cpu()),
        "trainable_mask_parameters_with_nonzero_grad": nonzero_names,
        "branch_nonzero_grad": branch_status,
        "parameter_grad_details": grad_details,
        "frozen_parameters_with_grad": frozen_with_grad,
    }


def make_box_datapoint(datapoint: Any) -> Any:
    result = copy.deepcopy(datapoint)
    query = result.find_queries[0]
    object_index = query.object_ids_output[0]
    query.input_bbox = (
        result.images[query.image_id].objects[object_index].bbox.view(1, 4)
    )
    query.input_bbox_label = torch.ones(1, dtype=torch.long)
    query.input_mask = None
    query.input_mask_label = None
    query.input_mask_object_id = None
    result.find_queries = [query]
    return result


def legacy_box_check(
    model: torch.nn.Module,
    datapoint: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    batch = collate_fn_api([make_box_datapoint(datapoint)], dict_key="box")["box"]
    recorder = FlowRecorder(model)
    with recorder, torch.no_grad():
        output = model(copy_data_to_device(batch, device, non_blocking=True))
    output_stats, finite = summarize_output(output)
    if recorder.mask_prompt_encoder_calls != 0:
        raise AssertionError("Legacy box mode called MaskPromptEncoder")
    if recorder.box_encoder_calls != 1:
        raise AssertionError(
            f"Legacy box mode expected one BoxEncoder call, got {recorder.box_encoder_calls}"
        )
    if recorder.dense_mask_encoder_calls != 0:
        raise AssertionError("Legacy box mode called dense mask encoder")
    return {
        "legacy_box_forward_passed": True,
        "legacy_mask_prompt_encoder_calls": recorder.mask_prompt_encoder_calls,
        "legacy_box_encoder_calls": recorder.box_encoder_calls,
        "legacy_dense_mask_encoder_calls": recorder.dense_mask_encoder_calls,
        "legacy_all_outputs_finite": finite,
        "legacy_output_tensors": output_stats,
    }


def print_batch_result(
    index: int, info: dict[str, Any], recorder: FlowRecorder
) -> None:
    print(f"Batch {index}:")
    print(f"  image shape: {info['image_shape']}")
    print(f"  raw mask shape: {info['mask_shape']}")
    print(f"  valid masks: {info['valid_mask_count']}")
    print(f"  mask token shape: {info['mask_token_shape']}")
    print(f"  final_proj input shape: {info['final_proj_input_shape']}")
    print(
        "  calls: "
        f"mask_prompt={recorder.mask_prompt_encoder_calls}, "
        f"box={recorder.box_encoder_calls}, dense_mask={recorder.dense_mask_encoder_calls}"
    )
    print(f"  shared path calls: {info['shared_geometry_calls']}")
    for name, stats in info["output_tensors"].items():
        print(
            f"  {name}: shape={stats['shape']} dtype={stats['dtype']} "
            f"min={stats['min']:.6g} max={stats['max']:.6g} "
            f"mean={stats['mean']:.6g} finite={stats['finite']}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path_name in ("ann_file", "img_folder", "checkpoint"):
        path = getattr(args, path_name)
        if not path.exists():
            raise FileNotFoundError(f"--{path_name} does not exist: {path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = build_validation_dataset(args)
    datapoints = collect_datapoints(dataset, args.num_samples)

    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
    )
    model.eval()
    checkpoint_info = inspect_checkpoint(model, args.checkpoint)

    total_calls = {
        "mask_prompt_encoder": 0,
        "mask_prompt_method": 0,
        "box_encoder_in_mask_mode": 0,
        "dense_mask_encoder_in_mask_mode": 0,
    }
    batch_results = []
    successful_queries = 0
    for batch_index, start in enumerate(range(0, len(datapoints), args.batch_size), 1):
        current = datapoints[start : start + args.batch_size]
        batch, _output, recorder, info = forward_mask_batch(model, current, device)
        successful_queries += info["valid_mask_count"]
        total_calls["mask_prompt_encoder"] += recorder.mask_prompt_encoder_calls
        total_calls["mask_prompt_method"] += recorder.mask_prompt_method_calls
        total_calls["box_encoder_in_mask_mode"] += recorder.box_encoder_calls
        total_calls[
            "dense_mask_encoder_in_mask_mode"
        ] += recorder.dense_mask_encoder_calls
        info["find_text_batch"] = list(batch.find_text_batch)
        batch_results.append(info)
        print_batch_result(batch_index, info, recorder)
        if successful_queries >= args.num_samples:
            break
    if successful_queries < args.num_samples:
        raise AssertionError(
            f"Only {successful_queries} real COCO queries completed forward; "
            f"required {args.num_samples}"
        )

    pair = find_prompt_pair(datapoints)
    sensitivity = sensitivity_check(model, pair, device)
    print(f"Sensitivity: {sensitivity}")
    backward = backward_check(model, pair[0], device)
    print(
        "Backward: "
        f"connected={backward['backward_grad_connected']}, "
        f"branch_nonzero_grad={backward['branch_nonzero_grad']}"
    )
    legacy = legacy_box_check(model, pair[0], device)
    print(
        "Legacy box: "
        f"passed={legacy['legacy_box_forward_passed']}, "
        f"box_calls={legacy['legacy_box_encoder_calls']}, "
        f"mask_calls={legacy['legacy_mask_prompt_encoder_calls']}"
    )

    return {
        "passed": True,
        "num_requested_samples": args.num_samples,
        "num_successful_samples": successful_queries,
        "num_failed_samples": 0,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mask_token_shapes": [result["mask_token_shape"] for result in batch_results],
        "mask_prompt_encoder_calls": total_calls["mask_prompt_encoder"],
        "mask_prompt_method_calls": total_calls["mask_prompt_method"],
        "box_encoder_calls_in_mask_mode": total_calls["box_encoder_in_mask_mode"],
        "dense_mask_encoder_calls_in_mask_mode": total_calls[
            "dense_mask_encoder_in_mask_mode"
        ],
        "all_model_outputs_finite": all(
            result["all_outputs_finite"] for result in batch_results
        ),
        **sensitivity,
        **backward,
        **checkpoint_info,
        **legacy,
        "batches": batch_results,
        "failures": [],
    }


def main() -> None:
    args = parse_args()
    summary = {
        "passed": False,
        "num_requested_samples": args.num_samples,
        "num_successful_samples": 0,
        "num_failed_samples": args.num_samples,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "failures": [],
    }
    try:
        summary = run(args)
    except Exception as error:
        summary["failures"].append(f"{type(error).__name__}: {error}")
        print(f"VALIDATION FAILED: {type(error).__name__}: {error}")
        raise
    finally:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Summary written to {args.output_json.resolve()}")
    print(
        "VALIDATION PASSED: "
        f"samples={summary['num_successful_samples']}, "
        f"token_shapes={summary['mask_token_shapes']}, "
        f"finite={summary['all_model_outputs_finite']}, "
        f"backward={summary['backward_grad_connected']}, "
        f"legacy_box={summary['legacy_box_forward_passed']}"
    )


if __name__ == "__main__":
    main()
