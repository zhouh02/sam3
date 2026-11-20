"""Dependency-free checks for coordinate mapping and PCS prompt lifecycle."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "pcs_gui", Path(__file__).resolve().parents[1] / "scripts/sam3_pcs_gui.py")
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


class FakeProcessor:
    def reset_all_prompts(self, state):
        state.pop("text", None)
        state["prompts"] = []
        state["points"] = []

    def set_confidence_threshold(self, threshold):
        self.threshold = threshold

    def set_text_prompt(self, prompt, state):
        state["text"] = prompt
        return state

    def add_geometric_prompt(self, box, label, state):
        state["prompts"].append((box, label))
        return state

    def add_point_prompt(self, point, label, state):
        state["points"].append((point, label))
        return state


class PromptTests(unittest.TestCase):
    def test_zoom_preserves_pixel_under_cursor(self):
        offset, size, new_size, anchor = (100, 50), (400, 300), (800, 600), (250, 200)
        changed = gui.zoom_offset(offset, size, new_size, anchor)
        before = gui.canvas_point(*anchor, offset, size, (1600, 1200))
        after = gui.canvas_point(*anchor, changed, new_size, (1600, 1200))
        self.assertEqual(before, after)

    def test_point_labels_normalization_and_reset(self):
        state = {}
        processor = FakeProcessor()
        gui.apply_prompts(processor, state, "", [], (400, 200), 0.5,
                          [((100, 100), True), ((300, 50), False)])
        self.assertEqual(state["points"], [([0.25, 0.5], True), ([0.75, 0.25], False)])
        gui.apply_prompts(processor, state, "car", [], (400, 200), 0.5)
        self.assertEqual(state["points"], [])

    def test_letterbox_and_edge_clipping(self):
        self.assertEqual(gui.canvas_point(300, 200, (100, 50), (400, 300),
                                          (1600, 1200)), (800, 600))
        self.assertEqual(gui.canvas_point(-50, 700, (100, 50), (400, 300),
                                          (1600, 1200)), (0, 1200))

    def test_normalized_coordinates(self):
        self.assertEqual(gui.normalized_box((100, 50, 300, 150), 400, 200),
                         [0.5, 0.5, 0.5, 0.5])

    def test_combined_prompts_and_reset_between_runs(self):
        processor = FakeProcessor()
        embedding = object()
        state = {"backbone_out": embedding}
        boxes = [((0, 0, 100, 50), True), ((100, 50, 200, 100), False)]
        result = gui.apply_prompts(processor, state, "car", boxes, (200, 100), 0.3)
        self.assertEqual(result["text"], "car")
        self.assertEqual(result["prompts"], [([0.25, 0.25, 0.5, 0.5], True),
                                              ([0.75, 0.75, 0.5, 0.5], False)])
        gui.apply_prompts(processor, state, "", boxes[:1], (200, 100), 0.8)
        self.assertNotIn("text", state)
        self.assertEqual(len(state["prompts"]), 1)
        self.assertIs(state["backbone_out"], embedding)
        self.assertEqual(processor.threshold, 0.8)
        gui.apply_prompts(processor, state, "dog", [], (200, 100), 0.5)
        self.assertEqual(state["prompts"], [])

    def test_empty_prompt_rejected(self):
        with self.assertRaises(ValueError):
            gui.apply_prompts(FakeProcessor(), {}, "", [], (200, 100), 0.5)


if __name__ == "__main__":
    unittest.main()
