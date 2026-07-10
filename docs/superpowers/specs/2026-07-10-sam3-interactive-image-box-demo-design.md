# SAM3 Interactive Prompt Demo Design

## Goal

Extend `scripts/interactive_image_box_demo.py` into a small interactive prompt-building demo. The window should let the user edit the text prompt, add any number of positive or negative box and point prompts, then click `Start` to run a single final SAM3 segmentation pass with all prompts fused.

## User Flow

Run:

```bash
python scripts/interactive_image_box_demo.py --image path/to/image.jpg --text "person"
```

The demo opens on the raw image and does not run segmentation until the user clicks `Start`, unless static `--box` or `--point` prompts are supplied and the user explicitly starts from the UI. The text prompt is editable in the Matplotlib window. The user can switch between `Box +`, `Box -`, `Point +`, and `Point -` modes, add prompts on the image, and click `Start` to run:

```text
set_image -> reset_all_prompts -> text forward -> append boxes -> append points -> _forward_grounding
```

After results appear, the user can add more prompts and click `Start` again. `Reset` clears all geometric prompts and current results while keeping the current text prompt.

## Architecture

Keep the implementation in `scripts/interactive_image_box_demo.py` because it is a standalone demo. Preserve the existing helper layer for prompt parsing, coordinate conversion, tensor conversion, and overlay rendering so tests remain lightweight. Add UI-state methods to `InteractiveSam3BoxDemo` for selecting prompt mode, updating text, resetting prompts, adding points, and starting inference.

## Controls

- `TextBox`: edits `self.text_prompt`; changing it does not immediately run inference.
- `Box +` / `Box -`: select positive or negative box drawing mode.
- `Point +` / `Point -`: select positive or negative point placement mode.
- `Start`: fuses the current text, boxes, and points and runs final grounding.
- `Reset`: clears boxes, points, and current model outputs.
- Keyboard shortcuts: `b` selects positive box, `n` selects negative box, `p` selects positive point, `x` selects negative point, `enter` starts inference, and `r` resets prompts.

## Constraints

- Use the existing `build_sam3_image_model` and `Sam3Processor` APIs.
- Use local checkpoint `sam3.pt` by default and pass `load_from_HF=False`.
- Interpret drawn boxes and points in image pixel coordinates and clamp them to image bounds.
- Store box prompts in normalized SAM3 `cxcywh` format and point prompts in normalized `xy` format.
- Do not run text-only segmentation before geometric prompts are provided.
- Keep automated tests free of model/checkpoint loading.
- Use Matplotlib as the desktop interactive UI.

## Testing

Automated tests cover:

- Pixel box and point parsing/conversion.
- Output tensor conversion and mask overlay rendering.
- The prompt fusion order used by final inference.
- Starting with only text does not run inference.
- `Start` runs inference with collected prompts.
- `Reset` clears geometry and results but preserves text.
- Text updates preserve existing geometry and are used by the next `Start`.
- Box and point modes add positive and negative prompts without triggering inference.

Manual verification is running the script with `assets/images/truck.jpg`, adding multiple boxes and points, changing text, clicking `Start`, and checking that results refresh only after `Start`.
