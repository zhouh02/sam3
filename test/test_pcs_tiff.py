"""TIFF decoding regression checks; run in the SAM3 environment."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import tifffile

spec = importlib.util.spec_from_file_location(
    "pcs_gui", Path(__file__).resolve().parents[1] / "scripts/sam3_pcs_gui.py")
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


class TiffTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory(dir=self.workspace, prefix="tiff_test_")
        self.folder = Path(self.temp.name).resolve()
        self.assertEqual(self.folder.parent, self.workspace)

    def tearDown(self):
        self.temp.cleanup()

    def read(self, data, **kwargs):
        path = self.folder / "测试.TIF"
        tifffile.imwrite(path, data, **kwargs)
        return gui.read_input_image(path)

    def test_uint16_gray_preserves_contrast(self):
        data = np.linspace(1000, 50000, 100, dtype=np.uint16).reshape(10, 10)
        image, info = self.read(data)
        pixels = np.asarray(image)
        self.assertEqual(image.mode, "RGB")
        self.assertGreater(len(np.unique(pixels)), 80)
        self.assertEqual((pixels.min(), pixels.max()), (0, 255))
        self.assertEqual(info["source_dtype"], "uint16")

    def test_float_invalid_and_constant(self):
        data = np.array([[np.nan, np.inf, -np.inf], [1., 5., 9.]], dtype=np.float32)
        image, _ = self.read(data)
        self.assertTrue((np.asarray(image)[0] == 0).all())
        image, _ = self.read(np.full((8, 8), 1000, dtype=np.uint16))
        self.assertTrue((np.asarray(image) == 0).all())
        with self.assertRaisesRegex(ValueError, "没有有效像素"):
            self.read(np.full((8, 8), np.nan, dtype=np.float32))

    def test_rgb8_unchanged(self):
        data = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
        image, _ = self.read(data, photometric="rgb")
        np.testing.assert_array_equal(np.asarray(image), data)

    def test_rgb16_planar_and_interleaved(self):
        data = np.arange(180, dtype=np.uint16).reshape(6, 10, 3) * 200
        image, _ = self.read(data, photometric="rgb")
        planar, _ = self.read(data.transpose(2, 0, 1), photometric="rgb", planarconfig="separate")
        np.testing.assert_array_equal(np.asarray(image), np.asarray(planar))
        self.assertEqual(image.size, (10, 6))

    def test_lzw_gray_without_imagecodecs(self):
        path = self.folder / "lzw.tif"
        data = np.arange(100, dtype=np.uint16).reshape(10, 10) * 100
        Image.fromarray(data).save(path, compression="tiff_lzw")
        image, _ = gui.read_input_image(path)
        self.assertGreater(len(np.unique(image)), 80)

    def test_multiple_pages(self):
        path = self.folder / "pages.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((8, 10), dtype=np.uint8))
            writer.write(np.full((6, 12), 173, dtype=np.uint8))
        image, info = gui.read_input_image(path, 1)
        self.assertEqual(image.size, (12, 6))
        self.assertTrue((np.asarray(image) == 173).all())
        self.assertEqual(info["page_count"], 2)
        self.assertEqual(info["page_index"], 1)

    def test_orientation(self):
        data = np.arange(60, dtype=np.uint16).reshape(6, 10) * 100
        plain, _ = self.read(data)
        oriented, _ = self.read(data, extratags=[(274, "H", 1, 6, False)])
        np.testing.assert_array_equal(np.asarray(oriented), np.asarray(plain.transpose(Image.Transpose.ROTATE_270)))

    def test_multiband_and_nodata(self):
        data = np.arange(300, dtype=np.uint16).reshape(6, 10, 5)
        data[0, 0] = 65535
        image, info = self.read(data, photometric="minisblack", planarconfig="contig",
                               extratags=[(42113, "s", 0, "65535", False)])
        self.assertEqual(info["display_bands"], [1, 2, 3])
        self.assertEqual(info["source_channels"], 5)
        self.assertTrue((np.asarray(image)[0, 0] == 0).all())

    def test_miniswhite_and_png(self):
        data = np.arange(60, dtype=np.uint8).reshape(6, 10)
        image, _ = self.read(data, photometric="miniswhite")
        np.testing.assert_array_equal(np.asarray(image)[..., 0], 255 - data)
        path = self.folder / "normal.png"
        Image.fromarray(data).save(path)
        image, info = gui.read_input_image(path)
        self.assertEqual(info["format"], "PNG")
        np.testing.assert_array_equal(np.asarray(image)[..., 0], data)


if __name__ == "__main__":
    unittest.main()
