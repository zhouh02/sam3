"""Interactive SAM3 image demo with text and box prompts.

Example:
    python scripts/interactive_image_box_demo.py --image assets/images/truck.jpg --text "truck"
"""

from __future__ import annotations

import argparse
import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


MASK_COLORS = np.array(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ],
    dtype=np.float32,
)
DEFAULT_TEXT_PROMPT = "visual"


def resolve_text_prompt(text_prompt: str | None) -> str:
    text = (text_prompt or "").strip()
    return text or DEFAULT_TEXT_PROMPT


@dataclass(frozen=True)
class BoxPrompt:
    box: list[float]
    label: bool = True
    display_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PointPrompt:
    point: list[float]
    label: bool = True
    display_point: tuple[float, float] | None = None


class _NumpyPromptTensor:
    def __init__(self, array: np.ndarray):
        self._array = array

    def numpy(self) -> np.ndarray:
        return self._array


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _clamp_xyxy(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x_min, x_max = sorted((float(x0), float(x1)))
    y_min, y_max = sorted((float(y0), float(y1)))
    x_min = min(max(x_min, 0.0), float(image_width))
    x_max = min(max(x_max, 0.0), float(image_width))
    y_min = min(max(y_min, 0.0), float(image_height))
    y_max = min(max(y_max, 0.0), float(image_height))
    return x_min, y_min, x_max, y_max


def xyxy_to_normalized_cxcywh(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert a dragged pixel-space xyxy box to SAM3 normalized cxcywh format."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    x_min, y_min, x_max, y_max = _clamp_xyxy(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        image_width=image_width,
        image_height=image_height,
    )
    box_width = x_max - x_min
    box_height = y_max - y_min
    center_x = x_min + box_width / 2.0
    center_y = y_min + box_height / 2.0

    return [
        round(center_x / image_width, 6),
        round(center_y / image_height, 6),
        round(box_width / image_width, 6),
        round(box_height / image_height, 6),
    ]


def normalized_cxcywh_to_xyxy(
    box: Sequence[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = [float(value) for value in box]
    box_width = width * image_width
    box_height = height * image_height
    x0 = center_x * image_width - box_width / 2.0
    y0 = center_y * image_height - box_height / 2.0
    x1 = x0 + box_width
    y1 = y0 + box_height
    return _clamp_xyxy(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        image_width=image_width,
        image_height=image_height,
    )


def parse_box_prompt(
    value: str,
    image_width: int,
    image_height: int,
    label: bool = True,
    box_format: str = "xyxy",
) -> BoxPrompt:
    values = [
        float(part)
        for part in value.replace(",", " ").split()
        if part.strip()
    ]
    if len(values) != 4:
        raise ValueError(
            f"Box prompt must contain four numbers, got {len(values)}: {value!r}"
        )

    if box_format == "xyxy":
        x0, y0, x1, y1 = values
        display_box = _clamp_xyxy(x0, y0, x1, y1, image_width, image_height)
        box = xyxy_to_normalized_cxcywh(
            *display_box,
            image_width=image_width,
            image_height=image_height,
        )
    elif box_format == "xywh":
        x, y, width, height = values
        display_box = _clamp_xyxy(
            x,
            y,
            x + width,
            y + height,
            image_width,
            image_height,
        )
        box = xyxy_to_normalized_cxcywh(
            *display_box,
            image_width=image_width,
            image_height=image_height,
        )
    elif box_format == "cxcywh_norm":
        box = [round(min(max(float(part), 0.0), 1.0), 6) for part in values]
        display_box = normalized_cxcywh_to_xyxy(box, image_width, image_height)
    else:
        raise ValueError(f"Unsupported box_format: {box_format}")

    return BoxPrompt(box=box, label=bool(label), display_box=display_box)


def parse_point_prompt(
    value: str,
    image_width: int,
    image_height: int,
    label: bool = True,
    point_format: str = "xy",
) -> PointPrompt:
    values = [
        float(part)
        for part in value.replace(",", " ").split()
        if part.strip()
    ]
    if len(values) != 2:
        raise ValueError(
            f"Point prompt must contain two numbers, got {len(values)}: {value!r}"
        )

    if point_format == "xy":
        x, y = values
        x = min(max(x, 0.0), float(image_width))
        y = min(max(y, 0.0), float(image_height))
        point = [round(x / image_width, 6), round(y / image_height, 6)]
        display_point = (x, y)
    elif point_format == "xy_norm":
        point = [round(min(max(float(part), 0.0), 1.0), 6) for part in values]
        display_point = (point[0] * image_width, point[1] * image_height)
    else:
        raise ValueError(f"Unsupported point_format: {point_format}")

    return PointPrompt(point=point, label=bool(label), display_point=display_point)


def outputs_to_numpy(state: dict) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Convert SAM3 processor outputs to NumPy arrays for visualization."""
    masks_value = state.get("masks")
    boxes_value = state.get("boxes")
    scores_value = state.get("scores")

    if masks_value is None:
        masks = []
    else:
        masks_np = _to_numpy(masks_value).astype(bool)
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None, ...]
        masks = [mask for mask in masks_np]

    if boxes_value is None:
        boxes = np.empty((0, 4), dtype=np.float32)
    else:
        boxes = _to_numpy(boxes_value).astype(np.float32).reshape(-1, 4)

    if scores_value is None:
        scores = np.empty((0,), dtype=np.float32)
    else:
        scores = _to_numpy(scores_value).astype(np.float32).reshape(-1)

    return masks, boxes, scores


def overlay_masks(
    image_rgb: np.ndarray,
    masks: Iterable[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend boolean masks over an RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 RGB array")

    overlay = image.astype(np.float32).copy()
    for idx, mask in enumerate(masks):
        mask_np = np.asarray(mask).astype(bool)
        if mask_np.shape != image.shape[:2]:
            raise ValueError(
                f"mask shape {mask_np.shape} does not match image shape {image.shape[:2]}"
            )
        color = MASK_COLORS[idx % len(MASK_COLORS)]
        overlay[mask_np] = overlay[mask_np] * (1.0 - alpha) + color * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _inference_context():
    try:
        import torch
    except ModuleNotFoundError:
        return nullcontext()
    return torch.inference_mode()


def _set_text_prompt_without_grounding(processor, state: dict, text_prompt: str) -> dict:
    text_prompt = resolve_text_prompt(text_prompt)
    text_outputs = processor.model.backbone.forward_text(
        [text_prompt],
        device=processor.device,
    )
    state["backbone_out"].update(text_outputs)
    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()
    return state


def _box_tensor(box: Sequence[float], device):
    try:
        import torch
    except ModuleNotFoundError:
        return _NumpyPromptTensor(np.asarray(box, dtype=np.float32).reshape(1, 1, 4))
    return torch.tensor(box, device=device, dtype=torch.float32).view(1, 1, 4)


def _box_label_tensor(label: bool, device):
    try:
        import torch
    except ModuleNotFoundError:
        return _NumpyPromptTensor(np.asarray([[bool(label)]], dtype=bool))
    return torch.tensor([label], device=device, dtype=torch.bool).view(1, 1)


def _point_tensor(point: Sequence[float], device):
    try:
        import torch
    except ModuleNotFoundError:
        return _NumpyPromptTensor(np.asarray(point, dtype=np.float32).reshape(1, 1, 2))
    return torch.tensor(point, device=device, dtype=torch.float32).view(1, 1, 2)


def _point_label_tensor(label: bool, device):
    try:
        import torch
    except ModuleNotFoundError:
        return _NumpyPromptTensor(np.asarray([[1 if label else 0]], dtype=np.int64))
    return torch.tensor([1 if label else 0], device=device, dtype=torch.long).view(1, 1)


def _append_box_prompt(processor, state: dict, prompt: BoxPrompt) -> None:
    state["geometric_prompt"].append_boxes(
        _box_tensor(prompt.box, processor.device),
        _box_label_tensor(prompt.label, processor.device),
    )


def _append_point_prompt(processor, state: dict, prompt: PointPrompt) -> None:
    state["geometric_prompt"].append_points(
        _point_tensor(prompt.point, processor.device),
        _point_label_tensor(prompt.label, processor.device),
    )


def run_sam3_inference(
    processor,
    image: Image.Image,
    text_prompt: str | None,
    box_prompts: Sequence[BoxPrompt] | None = None,
    point_prompts: Sequence[PointPrompt] | None = None,
) -> dict:
    """Run the FSS-style single-image SAM3 prompt flow.

    Text features and geometric prompts are assembled first; only then do we run
    final grounding for display.
    """
    with _inference_context():
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        text_prompt = resolve_text_prompt(text_prompt)
        state = _set_text_prompt_without_grounding(processor, state, text_prompt)
        for prompt in box_prompts or []:
            _append_box_prompt(processor, state, prompt)
        for prompt in point_prompts or []:
            _append_point_prompt(processor, state, prompt)
        return processor._forward_grounding(state)


class InteractiveSam3BoxDemo:
    def __init__(
        self,
        image: Image.Image,
        processor,
        text_prompt: str,
        initial_box_prompts: Sequence[BoxPrompt] | None = None,
        initial_point_prompts: Sequence[PointPrompt] | None = None,
        positive_box: bool = True,
        min_box_size: float = 2.0,
    ) -> None:
        self.image = image.convert("RGB")
        self.image_rgb = np.asarray(self.image)
        self.processor = processor
        self.text_prompt = resolve_text_prompt(text_prompt)
        self.box_prompts = list(initial_box_prompts or [])
        self.point_prompts = list(initial_point_prompts or [])
        self.prompt_kind = "box"
        self.prompt_label = bool(positive_box)
        self.positive_box = self.prompt_label
        self.min_box_size = min_box_size

        self.state: dict | None = None
        self.prompted_boxes: list[dict] = [
            {"box": prompt.display_box, "label": prompt.label}
            for prompt in self.box_prompts
            if prompt.display_box is not None
        ]
        self.prompted_points: list[dict] = [
            {"point": prompt.display_point, "label": prompt.label}
            for prompt in self.point_prompts
            if prompt.display_point is not None
        ]
        self.box_start: tuple[float, float] | None = None
        self.preview_rect = None

        self.fig = None
        self.ax = None
        self.patches = None
        self.widgets = []
        self.text_box = None

    @property
    def image_width(self) -> int:
        return self.image.size[0]

    @property
    def image_height(self) -> int:
        return self.image.size[1]

    @property
    def has_geometric_prompts(self) -> bool:
        return bool(self.box_prompts or self.point_prompts)

    def run_initial_inference(self) -> None:
        self.start_inference()

    def start_inference(self, event=None) -> None:
        self._sync_text_box_prompt()
        self.state = run_sam3_inference(
            processor=self.processor,
            image=self.image,
            text_prompt=self.text_prompt,
            box_prompts=self.box_prompts,
            point_prompts=self.point_prompts,
        )
        self.draw()

    def reset_prompts(self, event=None) -> None:
        if self.preview_rect is not None:
            self.preview_rect.remove()
            self.preview_rect = None
        self.box_prompts.clear()
        self.point_prompts.clear()
        self.prompted_boxes.clear()
        self.prompted_points.clear()
        self.box_start = None
        self.state = None
        print("Cleared box and point prompts.")
        self.draw()

    def update_text_prompt(self, text: str) -> None:
        resolved_text = resolve_text_prompt(text)
        self.text_prompt = resolved_text
        if (text or "").strip():
            print(f'Updated text prompt: "{self.text_prompt}"')
        else:
            print(f'Using default text prompt: "{self.text_prompt}"')
            if (
                self.text_box is not None
                and hasattr(self.text_box, "set_val")
                and getattr(self.text_box, "text", None) != self.text_prompt
            ):
                self.text_box.set_val(self.text_prompt)
        self.draw()

    def set_prompt_mode(self, kind: str, label: bool) -> None:
        if kind not in {"box", "point"}:
            raise ValueError(f"Unsupported prompt mode: {kind}")
        self.prompt_kind = kind
        self.prompt_label = bool(label)
        self.positive_box = self.prompt_label
        mode = self._prompt_mode_label()
        print(f"Selected prompt mode: {mode}")
        self.draw()

    def _sync_text_box_prompt(self) -> None:
        if self.text_box is None or not hasattr(self.text_box, "text"):
            return
        self.text_prompt = resolve_text_prompt(str(self.text_box.text))
        if (
            not str(self.text_box.text).strip()
            and hasattr(self.text_box, "set_val")
            and getattr(self.text_box, "text", None) != self.text_prompt
        ):
            self.text_box.set_val(self.text_prompt)

    def add_box_prompt(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        label: bool | None = None,
    ) -> bool:
        x_min, y_min, x_max, y_max = _clamp_xyxy(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        if (x_max - x_min) < self.min_box_size or (y_max - y_min) < self.min_box_size:
            return False

        prompt_label = self.prompt_label if label is None else bool(label)
        box = xyxy_to_normalized_cxcywh(
            x0=x_min,
            y0=y_min,
            x1=x_max,
            y1=y_max,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        prompt = BoxPrompt(
            box=box,
            label=prompt_label,
            display_box=(x_min, y_min, x_max, y_max),
        )
        self.box_prompts.append(prompt)
        self.prompted_boxes.append({"box": prompt.display_box, "label": prompt_label})
        self.state = None
        print(f"Added {'positive' if prompt_label else 'negative'} box prompt: {box}")
        self.draw()
        return True

    def add_point_prompt(
        self,
        x: float,
        y: float,
        label: bool | None = None,
    ) -> None:
        prompt_label = self.prompt_label if label is None else bool(label)
        prompt = parse_point_prompt(
            f"{x},{y}",
            image_width=self.image_width,
            image_height=self.image_height,
            label=prompt_label,
            point_format="xy",
        )
        self.point_prompts.append(prompt)
        self.prompted_points.append(
            {"point": prompt.display_point, "label": prompt_label}
        )
        self.state = None
        print(
            f"Added {'positive' if prompt_label else 'negative'} point prompt: "
            f"{prompt.point}"
        )
        self.draw()

    def _prompt_mode_label(self) -> str:
        sign = "+" if self.prompt_label else "-"
        return f"{self.prompt_kind.capitalize()} {sign}"

    def show(self) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        self.patches = patches
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.subplots_adjust(bottom=0.18)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._create_controls()
        self.draw()
        plt.show()

    def _create_controls(self) -> None:
        if self.fig is None:
            return

        from matplotlib.widgets import Button, TextBox

        class ResizeSafeTextBox(TextBox):
            def connect_event(self, event, callback):
                if event != "resize_event":
                    return super().connect_event(event, callback)

                def safe_resize_callback(resize_event):
                    if not hasattr(resize_event, "inaxes"):
                        resize_event = copy.copy(resize_event)
                        resize_event.inaxes = self.ax
                    return callback(resize_event)

                return super().connect_event(event, safe_resize_callback)

        self.widgets.clear()
        text_ax = self.fig.add_axes([0.08, 0.04, 0.28, 0.055])
        self.text_box = ResizeSafeTextBox(text_ax, "Text", initial=self.text_prompt)
        self.text_box.on_submit(self.update_text_prompt)
        self.widgets.append(self.text_box)

        button_specs = [
            ("Box +", 0.39, lambda event: self.set_prompt_mode("box", True)),
            ("Box -", 0.48, lambda event: self.set_prompt_mode("box", False)),
            ("Point +", 0.57, lambda event: self.set_prompt_mode("point", True)),
            ("Point -", 0.68, lambda event: self.set_prompt_mode("point", False)),
            ("Start", 0.80, self.start_inference),
            ("Reset", 0.89, self.reset_prompts),
        ]
        for label, left, callback in button_specs:
            width = 0.08 if label != "Point +" and label != "Point -" else 0.10
            button_ax = self.fig.add_axes([left, 0.04, width, 0.055])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.widgets.append(button)

    def draw(self) -> None:
        if self.ax is None:
            return

        masks, boxes, scores = outputs_to_numpy(self.state or {})
        self.ax.clear()
        self.ax.imshow(overlay_masks(self.image_rgb, masks))
        self._draw_prediction_boxes(boxes=boxes, scores=scores)
        self._draw_prompted_boxes()
        self._draw_prompted_points()

        mode = self._prompt_mode_label()
        prompt_summary = (
            f"{len(self.box_prompts)} boxes, {len(self.point_prompts)} points"
        )
        if self.state is None:
            title = (
                f'SAM3 prompt: "{self.text_prompt}" | {mode} | '
                f"{prompt_summary} | click Start"
            )
        else:
            title = (
                f'SAM3 prompt: "{self.text_prompt}" | {len(masks)} masks | '
                f"{mode} | {prompt_summary}"
            )
        self.ax.set_title(title)
        self.ax.set_xlim(0, self.image_width)
        self.ax.set_ylim(self.image_height, 0)
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def _draw_prediction_boxes(self, boxes: np.ndarray, scores: np.ndarray) -> None:
        if self.patches is None:
            return
        for idx, box in enumerate(boxes):
            x0, y0, x1, y1 = box.tolist()
            color = MASK_COLORS[idx % len(MASK_COLORS)] / 255.0
            rect = self.patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1.6,
                edgecolor=color,
                facecolor="none",
            )
            self.ax.add_patch(rect)
            score = scores[idx] if idx < len(scores) else None
            label = f"{idx}: {score:.2f}" if score is not None else str(idx)
            self.ax.text(
                x0,
                max(y0 - 4, 0),
                label,
                color="white",
                fontsize=8,
                bbox={"facecolor": color, "alpha": 0.8, "pad": 2},
            )

    def _draw_prompted_boxes(self) -> None:
        if self.patches is None:
            return
        for prompt in self.prompted_boxes:
            x0, y0, x1, y1 = prompt["box"]
            color = "lime" if prompt["label"] else "red"
            rect = self.patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=2.2,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
            )
            self.ax.add_patch(rect)

    def _draw_prompted_points(self) -> None:
        for prompt in self.prompted_points:
            point = prompt["point"]
            if point is None:
                continue
            x, y = point
            color = "lime" if prompt["label"] else "red"
            marker = "*" if prompt["label"] else "x"
            self.ax.scatter(
                [x],
                [y],
                c=color,
                marker=marker,
                s=90,
                edgecolors="white",
                linewidths=1.0,
            )

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self.prompt_kind == "point":
            self.add_point_prompt(float(event.xdata), float(event.ydata))
            return
        self.box_start = (float(event.xdata), float(event.ydata))

    def _on_motion(self, event) -> None:
        if (
            self.prompt_kind != "box"
            or
            self.box_start is None
            or event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
            or self.patches is None
        ):
            return

        if self.preview_rect is not None:
            self.preview_rect.remove()

        x0, y0 = self.box_start
        x1, y1 = float(event.xdata), float(event.ydata)
        x_min, y_min, x_max, y_max = _clamp_xyxy(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        color = "lime" if self.prompt_label else "red"
        self.preview_rect = self.patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2.0,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
        )
        self.ax.add_patch(self.preview_rect)
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self.prompt_kind != "box" or self.box_start is None:
            return

        if self.preview_rect is not None:
            self.preview_rect.remove()
            self.preview_rect = None

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.box_start = None
            self.draw()
            return

        x0, y0 = self.box_start
        x1, y1 = float(event.xdata), float(event.ydata)
        self.box_start = None
        self.add_box_prompt(x0, y0, x1, y1)

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key == "b":
            self.set_prompt_mode("box", True)
        elif key == "n":
            self.set_prompt_mode("box", False)
        elif key == "p":
            self.set_prompt_mode("point", True)
        elif key == "x":
            self.set_prompt_mode("point", False)
        elif key in {"enter", "return"}:
            self.start_inference()
        elif key == "r":
            self.reset_prompts()


def default_checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[1] / "sam3.pt"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_processor(
    checkpoint_path: Path,
    device: str,
    confidence_threshold: float,
):
    os.environ.setdefault("USE_PERFLIB", "0")
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        load_from_HF=False,
    )
    return Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build SAM3 image prompts interactively, then click Start to run "
            "segmentation."
        )
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--text",
        default=None,
        help=f'Text prompt for SAM3. Defaults to "{DEFAULT_TEXT_PROMPT}".',
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to SAM3 checkpoint. Defaults to sam3.pt in the repository root.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cuda, or cpu. Default: auto.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Mask confidence threshold passed to Sam3Processor. Default: 0.5.",
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="Start the interactive box mode with negative prompts selected.",
    )
    parser.add_argument(
        "--box",
        action="append",
        default=[],
        help=(
            "Positive box prompt. Repeatable. Default format is pixel xyxy: "
            "'x0,y0,x1,y1'."
        ),
    )
    parser.add_argument(
        "--negative-box",
        action="append",
        default=[],
        help=(
            "Negative box prompt. Repeatable. Uses the same format as --box."
        ),
    )
    parser.add_argument(
        "--box-format",
        default="xyxy",
        choices=["xyxy", "xywh", "cxcywh_norm"],
        help=(
            "Format for --box and --negative-box. xyxy/xywh are pixel coordinates; "
            "cxcywh_norm is normalized SAM3 format. Default: xyxy."
        ),
    )
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help=(
            "Positive point prompt. Repeatable. Default format is pixel xy: 'x,y'."
        ),
    )
    parser.add_argument(
        "--negative-point",
        action="append",
        default=[],
        help="Negative point prompt. Repeatable. Uses the same format as --point.",
    )
    parser.add_argument(
        "--point-format",
        default="xy",
        choices=["xy", "xy_norm"],
        help=(
            "Format for --point and --negative-point. xy is pixel coordinates; "
            "xy_norm is normalized coordinates. Default: xy."
        ),
    )
    return parser.parse_args(argv)


def parse_box_prompts_from_args(args, image: Image.Image) -> list[BoxPrompt]:
    image_width, image_height = image.size
    prompts = [
        parse_box_prompt(
            value,
            image_width=image_width,
            image_height=image_height,
            label=True,
            box_format=args.box_format,
        )
        for value in args.box
    ]
    prompts.extend(
        parse_box_prompt(
            value,
            image_width=image_width,
            image_height=image_height,
            label=False,
            box_format=args.box_format,
        )
        for value in args.negative_box
    )
    return prompts


def parse_point_prompts_from_args(args, image: Image.Image) -> list[PointPrompt]:
    image_width, image_height = image.size
    prompts = [
        parse_point_prompt(
            value,
            image_width=image_width,
            image_height=image_height,
            label=True,
            point_format=args.point_format,
        )
        for value in args.point
    ]
    prompts.extend(
        parse_point_prompt(
            value,
            image_width=image_width,
            image_height=image_height,
            label=False,
            point_format=args.point_format,
        )
        for value in args.negative_point
    )
    return prompts


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    image_path = Path(args.image).expanduser().resolve()
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else default_checkpoint_path()
    )

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Pass --checkpoint to override."
        )

    device = resolve_device(args.device)
    image = Image.open(image_path).convert("RGB")
    text_prompt = resolve_text_prompt(args.text)
    box_prompts = parse_box_prompts_from_args(args, image)
    point_prompts = parse_point_prompts_from_args(args, image)

    print(f"Loading SAM3 checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    processor = build_processor(
        checkpoint_path=checkpoint_path,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    demo = InteractiveSam3BoxDemo(
        image=image,
        processor=processor,
        text_prompt=text_prompt,
        initial_box_prompts=box_prompts,
        initial_point_prompts=point_prompts,
        positive_box=not args.negative,
    )
    if demo.has_geometric_prompts:
        print(
            f'Loaded text prompt "{text_prompt}" with provided geometry. '
            "Click Start to run segmentation."
        )
    else:
        print(
            f'Loaded text prompt "{text_prompt}". '
            "Click Start for text-only segmentation, or add box/point prompts first."
        )
    demo.show()


if __name__ == "__main__":
    main()
