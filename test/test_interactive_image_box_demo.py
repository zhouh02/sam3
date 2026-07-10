import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile

import numpy as np
from PIL import Image

from scripts.interactive_image_box_demo import (
    BoxPrompt,
    DEFAULT_TEXT_PROMPT,
    InteractiveSam3BoxDemo,
    PointPrompt,
    main,
    outputs_to_numpy,
    overlay_masks,
    parse_box_prompt,
    parse_point_prompt,
    run_sam3_inference,
    xyxy_to_normalized_cxcywh,
)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self.device = "cpu"
        self.model = FakeModel(self.calls)

    def set_image(self, image):
        self.calls.append(("set_image", image))
        return {"image": image, "backbone_out": {}}

    def reset_all_prompts(self, state):
        self.calls.append(("reset_all_prompts", state))

    def _forward_grounding(self, state):
        self.calls.append(("_forward_grounding", state["geometric_prompt"].events[:]))
        state["final_forward"] = True
        return state


class FakeBackbone:
    def __init__(self, calls):
        self.calls = calls

    def forward_text(self, prompts, device):
        self.calls.append(("forward_text", prompts, device))
        return {"language_features": prompts[0]}


class FakePrompt:
    def __init__(self, calls):
        self.calls = calls
        self.events = []

    def append_boxes(self, boxes, labels):
        self.events.append(("box", boxes.numpy().reshape(-1).tolist(), bool(labels.numpy()[0][0])))

    def append_points(self, points, labels):
        self.events.append(
            ("point", points.numpy().reshape(-1).tolist(), int(labels.numpy()[0][0]))
        )


class FakeModel:
    def __init__(self, calls):
        self.calls = calls
        self.backbone = FakeBackbone(calls)

    def _get_dummy_prompt(self):
        self.calls.append(("_get_dummy_prompt",))
        return FakePrompt(self.calls)


class TestInteractiveImageBoxDemoHelpers(unittest.TestCase):
    def test_xyxy_to_normalized_cxcywh_clamps_and_handles_reversed_drag(self):
        box = xyxy_to_normalized_cxcywh(
            x0=120.0,
            y0=80.0,
            x1=-20.0,
            y1=220.0,
            image_width=200,
            image_height=100,
        )
        self.assertEqual(box, [0.3, 0.9, 0.6, 0.2])

    def test_outputs_to_numpy_accepts_sam3_tensors(self):
        state = {
            "masks": FakeTensor(
                [[[[True, False], [False, True]]], [[[False, True], [True, False]]]]
            ),
            "boxes": FakeTensor(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
            ),
            "scores": FakeTensor([0.9, 0.25]),
        }

        masks, boxes, scores = outputs_to_numpy(state)

        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[0].dtype, np.bool_)
        np.testing.assert_array_equal(boxes[1], np.array([5.0, 6.0, 7.0, 8.0]))
        np.testing.assert_allclose(scores, np.array([0.9, 0.25], dtype=np.float32))

    def test_overlay_masks_preserves_shape_and_changes_masked_pixels(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        masks = [np.array([[True, False], [False, False]])]

        overlay = overlay_masks(image, masks, alpha=0.5)

        self.assertEqual(overlay.shape, image.shape)
        self.assertTrue(np.any(overlay[0, 0] != image[0, 0]))
        np.testing.assert_array_equal(overlay[1, 1], image[1, 1])

    def test_parse_box_prompt_accepts_pixel_xyxy(self):
        prompt = parse_box_prompt(
            "10,20,110,220",
            image_width=200,
            image_height=400,
            label=True,
            box_format="xyxy",
        )

        self.assertEqual(prompt.box, [0.3, 0.3, 0.5, 0.5])
        self.assertTrue(prompt.label)

    def test_parse_box_prompt_accepts_pixel_xywh(self):
        prompt = parse_box_prompt(
            "10,20,100,200",
            image_width=200,
            image_height=400,
            label=False,
            box_format="xywh",
        )

        self.assertEqual(prompt.box, [0.3, 0.3, 0.5, 0.5])
        self.assertFalse(prompt.label)

    def test_parse_point_prompt_accepts_pixel_xy(self):
        prompt = parse_point_prompt(
            "50,80",
            image_width=200,
            image_height=400,
            label=False,
            point_format="xy",
        )

        self.assertEqual(prompt.point, [0.25, 0.2])
        self.assertEqual(prompt.display_point, (50.0, 80.0))
        self.assertFalse(prompt.label)

    def test_run_sam3_inference_fuses_prompts_before_final_forward(self):
        processor = FakeProcessor()
        box_prompt = BoxPrompt(box=[0.3, 0.3, 0.5, 0.5], label=True)
        point_prompt = PointPrompt(point=[0.25, 0.2], label=False)

        state = run_sam3_inference(
            processor=processor,
            image="image-object",
            text_prompt="car",
            box_prompts=[box_prompt],
            point_prompts=[point_prompt],
        )

        self.assertEqual(
            [call[0] for call in processor.calls],
            [
                "set_image",
                "reset_all_prompts",
                "forward_text",
                "_get_dummy_prompt",
                "_forward_grounding",
            ],
        )
        self.assertTrue(state["final_forward"])
        events = processor.calls[-1][1]
        self.assertEqual(events[0][0], "box")
        np.testing.assert_allclose(events[0][1], [0.3, 0.3, 0.5, 0.5])
        self.assertTrue(events[0][2])
        self.assertEqual(events[1][0], "point")
        np.testing.assert_allclose(events[1][1], [0.25, 0.2])
        self.assertEqual(events[1][2], 0)

    def test_interactive_demo_starts_without_inference_when_no_geometric_prompt(self):
        processor = FakeProcessor()
        image = Image.new("RGB", (8, 8), "black")

        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="car",
        )

        self.assertFalse(demo.has_geometric_prompts)
        self.assertIsNone(demo.state)
        self.assertEqual(processor.calls, [])

    def test_main_waits_for_start_even_when_static_geometry_is_provided(self):
        processor = FakeProcessor()
        demos = []

        def capture_show(demo):
            demos.append(demo)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "image.png"
            checkpoint_path = tmp_path / "sam3.pt"
            Image.new("RGB", (100, 100), "black").save(image_path)
            checkpoint_path.write_bytes(b"fake checkpoint")

            with patch(
                "scripts.interactive_image_box_demo.resolve_device",
                return_value="cpu",
            ), patch(
                "scripts.interactive_image_box_demo.build_processor",
                return_value=processor,
            ), patch.object(
                InteractiveSam3BoxDemo,
                "show",
                capture_show,
            ):
                main(
                    [
                        "--image",
                        str(image_path),
                        "--text",
                        "car",
                        "--checkpoint",
                        str(checkpoint_path),
                        "--box",
                        "10,10,40,40",
                    ]
                )

        self.assertEqual(processor.calls, [])
        self.assertEqual(len(demos), 1)
        self.assertEqual(len(demos[0].box_prompts), 1)
        self.assertIsNone(demos[0].state)

    def test_main_defaults_missing_text_to_visual_prompt(self):
        processor = FakeProcessor()
        demos = []

        def capture_show(demo):
            demos.append(demo)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "image.png"
            checkpoint_path = tmp_path / "sam3.pt"
            Image.new("RGB", (100, 100), "black").save(image_path)
            checkpoint_path.write_bytes(b"fake checkpoint")

            with patch(
                "scripts.interactive_image_box_demo.resolve_device",
                return_value="cpu",
            ), patch(
                "scripts.interactive_image_box_demo.build_processor",
                return_value=processor,
            ), patch.object(
                InteractiveSam3BoxDemo,
                "show",
                capture_show,
            ):
                main(
                    [
                        "--image",
                        str(image_path),
                        "--checkpoint",
                        str(checkpoint_path),
                    ]
                )

        self.assertEqual(processor.calls, [])
        self.assertEqual(len(demos), 1)
        self.assertEqual(demos[0].text_prompt, DEFAULT_TEXT_PROMPT)

    def test_text_box_resize_event_without_inaxes_does_not_crash(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backend_bases import ResizeEvent

        processor = FakeProcessor()
        image = Image.new("RGB", (100, 100), "black")
        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="visual",
        )
        demo.fig, demo.ax = plt.subplots()

        demo._create_controls()
        resize_event = ResizeEvent("resize_event", demo.fig.canvas)

        demo.fig.canvas.callbacks.process("resize_event", resize_event)

        plt.close(demo.fig)

    def test_start_inference_uses_current_text_and_existing_geometry(self):
        processor = FakeProcessor()
        image = Image.new("RGB", (200, 400), "black")
        box_prompt = BoxPrompt(
            box=[0.3, 0.3, 0.5, 0.5],
            label=True,
            display_box=(10.0, 20.0, 110.0, 220.0),
        )
        point_prompt = PointPrompt(
            point=[0.25, 0.2],
            label=False,
            display_point=(50.0, 80.0),
        )
        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="car",
            initial_box_prompts=[box_prompt],
            initial_point_prompts=[point_prompt],
        )

        demo.update_text_prompt("truck")
        demo.start_inference()

        self.assertEqual(demo.text_prompt, "truck")
        self.assertTrue(demo.state["final_forward"])
        self.assertEqual(processor.calls[2], ("forward_text", ["truck"], "cpu"))
        events = processor.calls[-1][1]
        self.assertEqual([event[0] for event in events], ["box", "point"])

    def test_start_inference_reads_unsubmitted_text_box_value(self):
        processor = FakeProcessor()
        image = Image.new("RGB", (200, 400), "black")
        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="car",
            initial_box_prompts=[
                BoxPrompt(
                    box=[0.3, 0.3, 0.5, 0.5],
                    label=True,
                    display_box=(10.0, 20.0, 110.0, 220.0),
                )
            ],
        )
        demo.text_box = type("FakeTextBox", (), {"text": "truck"})()

        demo.start_inference()

        self.assertEqual(demo.text_prompt, "truck")
        self.assertEqual(processor.calls[2], ("forward_text", ["truck"], "cpu"))

    def test_start_inference_runs_text_only_when_no_geometry_is_provided(self):
        processor = FakeProcessor()
        image = Image.new("RGB", (200, 400), "black")
        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="car",
        )

        demo.start_inference()

        self.assertTrue(demo.state["final_forward"])
        self.assertEqual(processor.calls[2], ("forward_text", ["car"], "cpu"))
        self.assertEqual(processor.calls[-1], ("_forward_grounding", []))

    def test_start_inference_uses_visual_prompt_when_text_is_empty(self):
        processor = FakeProcessor()
        image = Image.new("RGB", (200, 400), "black")
        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="",
        )
        demo.text_box = type("FakeTextBox", (), {"text": ""})()

        demo.start_inference()

        self.assertEqual(demo.text_prompt, DEFAULT_TEXT_PROMPT)
        self.assertEqual(
            processor.calls[2],
            ("forward_text", [DEFAULT_TEXT_PROMPT], "cpu"),
        )

    def test_reset_prompts_clears_geometry_and_results_but_preserves_text(self):
        processor = FakeProcessor()
        image = Image.new("RGB", (200, 400), "black")
        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="car",
            initial_box_prompts=[
                BoxPrompt(
                    box=[0.3, 0.3, 0.5, 0.5],
                    label=True,
                    display_box=(10.0, 20.0, 110.0, 220.0),
                )
            ],
            initial_point_prompts=[
                PointPrompt(
                    point=[0.25, 0.2],
                    label=False,
                    display_point=(50.0, 80.0),
                )
            ],
        )
        demo.update_text_prompt("truck")
        demo.start_inference()

        demo.reset_prompts()

        self.assertEqual(demo.text_prompt, "truck")
        self.assertEqual(demo.box_prompts, [])
        self.assertEqual(demo.point_prompts, [])
        self.assertEqual(demo.prompted_boxes, [])
        self.assertEqual(demo.prompted_points, [])
        self.assertIsNone(demo.state)

    def test_collects_box_and_point_prompts_without_running_inference(self):
        processor = FakeProcessor()
        image = Image.new("RGB", (200, 400), "black")
        demo = InteractiveSam3BoxDemo(
            image=image,
            processor=processor,
            text_prompt="car",
        )

        demo.set_prompt_mode("box", False)
        added_box = demo.add_box_prompt(10.0, 20.0, 110.0, 220.0)
        demo.set_prompt_mode("point", True)
        demo.add_point_prompt(50.0, 80.0)

        self.assertTrue(added_box)
        self.assertEqual(processor.calls, [])
        self.assertEqual(demo.box_prompts[0].box, [0.3, 0.3, 0.5, 0.5])
        self.assertFalse(demo.box_prompts[0].label)
        self.assertEqual(demo.prompted_boxes[0]["box"], (10.0, 20.0, 110.0, 220.0))
        self.assertEqual(demo.point_prompts[0].point, [0.25, 0.2])
        self.assertTrue(demo.point_prompts[0].label)
        self.assertEqual(demo.prompted_points[0]["point"], (50.0, 80.0))

        demo.start_inference()

        events = processor.calls[-1][1]
        self.assertEqual([event[0] for event in events], ["box", "point"])


if __name__ == "__main__":
    unittest.main()
