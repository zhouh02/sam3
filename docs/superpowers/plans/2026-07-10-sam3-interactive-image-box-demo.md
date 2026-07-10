# SAM3 Interactive Prompt Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add interactive text editing, reset, point prompts, prompt-mode buttons, and a `Start` button to the SAM3 image demo.

**Architecture:** Keep all demo behavior in `scripts/interactive_image_box_demo.py`. Extend `InteractiveSam3BoxDemo` with prompt-mode state and UI callback methods; keep model execution in `run_sam3_inference` so tests can verify prompt fusion without loading SAM3.

**Tech Stack:** Python, Pillow, NumPy, Matplotlib widgets, SAM3 `Sam3Processor`, pytest/unittest.

## Global Constraints

- Use `build_sam3_image_model` and `Sam3Processor`.
- Default checkpoint path is `sam3.pt` in the repository root.
- Pass `load_from_HF=False` when building the model.
- Convert dragged pixel boxes to normalized `[center_x, center_y, width, height]`.
- Convert clicked points to normalized `[x, y]`.
- Do not run text-only segmentation before geometric prompts are provided.
- Run segmentation only when `Start` is clicked or `run_initial_inference()` is explicitly called by tests/CLI setup.
- Keep automated tests free of model/checkpoint loading.

---

### Task 1: Prompt Builder State Tests

**Files:**
- Modify: `test/test_interactive_image_box_demo.py`
- Modify: `scripts/interactive_image_box_demo.py`

**Interfaces:**
- `InteractiveSam3BoxDemo.set_prompt_mode(kind: str, label: bool) -> None`
- `InteractiveSam3BoxDemo.update_text_prompt(text: str) -> None`
- `InteractiveSam3BoxDemo.reset_prompts(event=None) -> None`
- `InteractiveSam3BoxDemo.start_inference(event=None) -> None`

- [ ] **Step 1: Write failing tests**

Add tests that instantiate `InteractiveSam3BoxDemo` with `FakeProcessor`, update text, reset prompts, and start inference. Assert reset preserves text and clears `box_prompts`, `point_prompts`, display prompts, and `state`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_interactive_image_box_demo.py -v`

Expected: FAIL because the new state methods do not exist yet.

- [ ] **Step 3: Implement minimal state methods**

Add `prompt_kind`, `prompt_label`, `set_prompt_mode`, `update_text_prompt`, `reset_prompts`, and `start_inference` to `InteractiveSam3BoxDemo`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_interactive_image_box_demo.py -v`

Expected: PASS.

### Task 2: Interactive Box And Point Collection

**Files:**
- Modify: `test/test_interactive_image_box_demo.py`
- Modify: `scripts/interactive_image_box_demo.py`

**Interfaces:**
- `InteractiveSam3BoxDemo.add_point_prompt(x: float, y: float, label: bool | None = None) -> None`
- `InteractiveSam3BoxDemo.add_box_prompt(x0: float, y0: float, x1: float, y1: float, label: bool | None = None) -> bool`

- [ ] **Step 1: Write failing tests**

Add tests that call `add_box_prompt` and `add_point_prompt` directly. Assert prompts are normalized, display prompts are stored, positive/negative labels are respected, and the processor is not called until `start_inference`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_interactive_image_box_demo.py -v`

Expected: FAIL because prompt collection methods do not exist yet.

- [ ] **Step 3: Implement prompt collection methods and event behavior**

Route mouse release through `add_box_prompt` in box mode and mouse click through `add_point_prompt` in point mode. Remove automatic inference from `_on_release`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test/test_interactive_image_box_demo.py -v`

Expected: PASS.

### Task 3: Matplotlib Controls

**Files:**
- Modify: `scripts/interactive_image_box_demo.py`

**Interfaces:**
- `InteractiveSam3BoxDemo._create_controls() -> None`
- `InteractiveSam3BoxDemo._on_key(event) -> None`

- [ ] **Step 1: Add UI controls**

Use Matplotlib `Button` and `TextBox` widgets for `Box +`, `Box -`, `Point +`, `Point -`, `Start`, `Reset`, and text prompt editing.

- [ ] **Step 2: Add keyboard shortcuts**

Map `b`, `n`, `p`, `x`, `enter`, and `r` to the same callbacks as the controls.

- [ ] **Step 3: Run syntax and unit checks**

Run: `python -m py_compile scripts/interactive_image_box_demo.py test/test_interactive_image_box_demo.py`

Expected: exits with code 0.

Run: `python -m pytest test/test_interactive_image_box_demo.py -v`

Expected: PASS.

### Task 4: End-To-End Verification

**Files:**
- Verify: `scripts/interactive_image_box_demo.py`
- Verify: `test/test_interactive_image_box_demo.py`
- Verify: `test/test_perflib_fused.py`

- [ ] **Step 1: Verify help output**

Run: `python scripts/interactive_image_box_demo.py --help`

Expected: help prints without loading the model.

- [ ] **Step 2: Verify conda test suite**

Run: `conda run -n sam3 python -m pytest test/test_interactive_image_box_demo.py test/test_perflib_fused.py -v`

Expected: PASS.

- [ ] **Step 3: Verify real SAM3 forward path**

Run a small `conda run -n sam3 python -c ...` script that builds the processor, supplies one box and one point prompt, calls `run_sam3_inference`, and prints masks/boxes/scores shapes.

Expected: command exits with code 0 and prints non-error output shapes.
