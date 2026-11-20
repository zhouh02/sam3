"""Standalone SAM3 PCS UI. See sam3_pcs_gui.md for setup and usage."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import traceback


def is_tiff(path):
    with open(path, "rb") as stream:
        return stream.read(4) in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def tiff_module():
    try:
        import tifffile
    except ImportError as error:
        raise RuntimeError("读取 TIFF 需要 tifffile，请在 sam3 环境运行：python -m pip install tifffile") from error
    return tifffile


def to_display_uint8(array, nodata=None):
    """Map high-depth data to display RGB without clipping everything above 255."""
    import numpy as np
    if array.dtype == np.uint8 and nodata is None:
        return array, None
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255, None
    if array.dtype.kind not in "uif":
        raise ValueError(f"不支持的 TIFF 数据类型：{array.dtype}")
    data = array.astype(np.float64)
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    # Preserve 8-bit intensities even when masking a NoData value.
    if array.dtype == np.uint8:
        return np.where(valid, array, 0).astype(np.uint8), None
    if not valid.any():
        raise ValueError("TIFF 当前页没有有效像素（全部为 NaN、Inf 或 NoData）。")
    low, high = np.percentile(data[valid], [1, 99])
    if high <= low:
        low, high = data[valid].min(), data[valid].max()
    mapped = np.zeros(data.shape, dtype=np.float64)
    if high > low:
        mapped[valid] = np.clip((data[valid] - low) / (high - low), 0, 1) * 255
    return np.rint(mapped).astype(np.uint8), [float(low), float(high)]


def read_input_image(path, page_index=0):
    """Decode one image/page to RGB; return reproducible conversion metadata."""
    import numpy as np
    from PIL import Image, ImageOps

    if not is_tiff(path):
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB"), {"format": source.format}
    tifffile = tiff_module()
    with tifffile.TiffFile(path) as source:
        page = source.pages[page_index]
        metadata = {"format": "TIFF", "page_index": page_index,
                    "page_count": len(source.pages), "source_dtype": str(page.dtype),
                    "source_shape": list(page.shape), "source_axes": page.axes}
        # Pillow correctly interprets palette, CMYK, YCbCr, and TIFF orientation.
        # Do not route high-bit RGB here: Pillow can silently reduce its bit depth.
        if page.dtype == np.dtype("uint8") and int(page.photometric) in (2, 3, 5, 6) and page.samplesperpixel <= 4:
            try:
                with Image.open(path) as decoded:
                    decoded.seek(page_index)
                    image = ImageOps.exif_transpose(decoded).convert("RGB")
                return image, metadata
            except (OSError, ValueError):
                pass
        if int(page.photometric) not in (0, 1, 2):
            raise ValueError(f"暂不支持该 TIFF 色彩编码：{page.photometric.name}（{page.dtype}）")
        try:
            array = page.asarray()
        except (ValueError, ImportError) as error:
            # libtiff in Pillow supports common LZW/Deflate files even when the
            # optional tifffile codec package is absent. Only use lossless modes.
            try:
                with Image.open(path) as decoded:
                    decoded.seek(page_index)
                    if page.samplesperpixel != 1:
                        raise ValueError("Cannot safely decode high-bit RGB through Pillow")
                    array = np.array(decoded)
                    if array.dtype.kind != page.dtype.kind or array.dtype.itemsize != page.dtype.itemsize:
                        raise ValueError("Pillow changed TIFF bit depth")
                # Pillow already applies TIFF orientation.
                axes = "YX"
                orientation = 1
            except Exception:
                raise RuntimeError(
                    "TIFF 解码失败；部分压缩格式需要 imagecodecs。请在 sam3 环境运行："
                    "python -m pip install imagecodecs\n原始错误：" + str(error)
                ) from error
        else:
            axes = page.axes
            orientation_tag = page.tags.get("Orientation")
            orientation = int(orientation_tag.value) if orientation_tag else 1
        if axes == "SYX":
            array = np.moveaxis(array, 0, -1)
        elif axes not in ("YX", "YXS"):
            raise ValueError(f"该 TIFF 页维度暂不支持：{axes} {array.shape}，请先导出二维图像。")
        channels = 1 if array.ndim == 2 else array.shape[-1]
        if channels == 1:
            array = array.reshape(array.shape[:2])
        elif channels == 2:
            array = array[..., 0]  # Gray + alpha: match the existing alpha-discarding RGB conversion.
        else:
            array = array[..., :3]
        metadata["display_bands"] = list(range(1, min(channels, 3) + 1)) if channels != 2 else [1]
        metadata["source_channels"] = channels
        nodata_tag = page.tags.get(42113)  # GDAL_NODATA, common in GeoTIFF.
        nodata = None
        if nodata_tag:
            try:
                nodata = float(str(nodata_tag.value).strip("\x00"))
            except ValueError:
                pass
        array, stretch = to_display_uint8(array, nodata)
        if int(page.photometric) == 0:
            array = 255 - array  # MINISWHITE
        metadata["display_range_1_99_percentile"] = stretch
        metadata["orientation"] = orientation
        image = Image.fromarray(array).convert("RGB")
        operations = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
                      4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
                      6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
                      8: Image.Transpose.ROTATE_90}
        if orientation in operations:
            image = image.transpose(operations[orientation])
        return image, metadata


def normalized_box(box, width, height):
    """Original-image xyxy -> normalized cxcywh expected by Sam3Processor."""
    x0, y0, x1, y1 = box
    return [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
            (x1 - x0) / width, (y1 - y0) / height]


def canvas_point(x, y, offset, display_size, image_size):
    """Map canvas coordinates, including letterboxing, to clamped image pixels."""
    return tuple(max(0, min(size, (v - origin) * size / shown))
                 for v, origin, shown, size in
                 zip((x, y), offset, display_size, image_size))


def zoom_offset(offset, old_size, new_size, anchor):
    """Keep the same image pixel under the cursor when zooming."""
    return tuple(a - (a - o) * new / old
                 for o, old, new, a in zip(offset, old_size, new_size, anchor))


def apply_prompts(processor, state, text, boxes, image_size, threshold, points=()):
    """Rebuild prompts every run; preserve the cached image embedding only."""
    if not text and not boxes and not points:
        raise ValueError("Enter text or add at least one box/point.")
    processor.reset_all_prompts(state)
    processor.set_confidence_threshold(threshold)
    if text:
        state = processor.set_text_prompt(prompt=text, state=state)
    for box, positive in boxes:
        state = processor.add_geometric_prompt(
            box=normalized_box(box, *image_size), label=positive, state=state)
    for (x, y), positive in points:
        state = processor.add_point_prompt(
            point=[x / image_size[0], y / image_size[1]], label=positive, state=state)
    return state


class Predictor:
    """Accessed only by the single inference worker, never the UI thread."""

    def __init__(self, args):
        self.args = args
        self.processor = None
        self.state = None
        self.cached_image = None

    def predict(self, image, text, boxes, threshold, points=()):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable. Activate your SAM3 CUDA environment.")
        if self.processor is None:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            model = build_sam3_image_model(
                checkpoint_path=self.args.checkpoint,
                load_from_HF=self.args.checkpoint is None,
                device="cuda", enable_inst_interactivity=False)
            self.processor = Sam3Processor(model, device="cuda")
        # SAM3's vision backbone uses BF16 internally. Autocast must be entered
        # in this worker thread (its context does not propagate from the UI).
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.state is None or self.cached_image is not image:
                # Drop the previous embedding before encoding the new image.
                self.state = None
                self.cached_image = None
                self.state = self.processor.set_image(image)
                self.cached_image = image
            result = apply_prompts(self.processor, self.state, text, boxes,
                                   image.size, threshold, points)
            return {key: (result[key] if key == "masks" else result[key].float()).detach().cpu().numpy()
                    for key in ("masks", "boxes", "scores")}


class App:
    COLORS = [(255, 95, 85), (45, 190, 255), (255, 195, 55),
              (160, 105, 255), (50, 220, 145), (255, 100, 210)]

    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.args = root, args
        self.image = None
        self.image_path = None
        self.image_metadata = {}
        self.predictor = Predictor(args)
        self.boxes = []
        self.points = []
        self.history = []
        self.zoom = 1.0
        self.pan = (0.0, 0.0)
        self.pan_start = None
        self.result = self.overlay = self.snapshot = None
        self.busy = False
        self.start = None
        self.events = queue.Queue()
        self.text = tk.StringVar(value=args.text)
        self.positive = tk.BooleanVar(value=True)
        self.tool = tk.StringVar(value="box")
        self.zoom_label = tk.StringVar(value="结果缩放 100%")
        self.threshold = tk.StringVar(value=str(args.threshold))
        self.status = tk.StringVar(value="框模式拖动，点模式单击；右侧滚轮缩放、左键拖动平移。")
        root.title("SAM3 PCS — text + box prompts")
        root.geometry("1280x820")
        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        self.open_button = ttk.Button(top, text="打开图像", command=self.open_image)
        self.open_button.pack(side="left", padx=(0, 8))
        root.bind("<Control-o>", lambda event: self.open_image())
        ttk.Label(top, text="文本提示:").pack(side="left")
        ttk.Entry(top, textvariable=self.text, width=40).pack(side="left", padx=6)
        ttk.Label(top, text="置信度:").pack(side="left")
        ttk.Entry(top, textvariable=self.threshold, width=6).pack(side="left", padx=6)
        self.run_button = ttk.Button(top, text="运行 PCS", command=self.run, state="disabled")
        self.run_button.pack(side="left", padx=6)
        self.save_button = ttk.Button(top, text="保存结果", command=self.save, state="disabled")
        self.save_button.pack(side="left")
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        ttk.Radiobutton(controls, text="提示框", variable=self.tool, value="box").pack(side="left")
        ttk.Radiobutton(controls, text="提示点", variable=self.tool, value="point").pack(side="left")
        ttk.Radiobutton(controls, text="正提示（目标）", variable=self.positive,
                        value=True).pack(side="left")
        ttk.Radiobutton(controls, text="负提示（排除）", variable=self.positive,
                        value=False).pack(side="left")
        ttk.Button(controls, text="撤销上一个提示", command=self.undo).pack(side="left", padx=8)
        ttk.Button(controls, text="清空全部提示", command=self.clear).pack(side="left")
        ttk.Button(controls, text="重置结果视图", command=self.reset_view).pack(side="left", padx=8)
        ttk.Label(controls, textvariable=self.zoom_label).pack(side="left")
        panels = ttk.Frame(root)
        panels.pack(fill="both", expand=True)
        self.canvases = []
        for label in ("输入图像 / 提示框", "分割结果 / 检测框 / 置信度"):
            frame = ttk.LabelFrame(panels, text=label)
            frame.pack(side="left", fill="both", expand=True)
            canvas = tk.Canvas(frame, bg="#20242b", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda event: self.redraw())
            self.canvases.append(canvas)
        self.canvases[0].bind("<ButtonPress-1>", self.press)
        self.canvases[0].bind("<B1-Motion>", self.drag)
        self.canvases[0].bind("<ButtonRelease-1>", self.release)
        self.canvases[1].bind("<MouseWheel>", self.wheel)
        self.canvases[1].bind("<Button-4>", self.wheel)
        self.canvases[1].bind("<Button-5>", self.wheel)
        self.canvases[1].bind("<ButtonPress-1>", self.pan_press)
        self.canvases[1].bind("<B1-Motion>", self.pan_drag)
        self.canvases[1].bind("<ButtonRelease-1>", self.pan_release)
        self.canvases[1].bind("<Double-Button-1>", lambda event: self.reset_view())
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1200).pack(fill="x")
        self.photos = [None, None]
        if args.image:
            self.load_image(args.image, clear_text=False)
        else:
            self.status.set("点击“打开图像”或按 Ctrl+O 选择图片，然后添加提示并运行。")
        root.after(100, self.poll)

    def open_image(self):
        from tkinter import filedialog
        if self.busy:
            return
        path = filedialog.askopenfilename(
            parent=self.root, title="打开要分割的图像",
            initialdir=str(self.image_path.parent) if self.image_path else str(Path.cwd()),
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                       ("所有文件", "*.*")])
        if path:
            self.load_image(path)

    def load_image(self, path, clear_text=True):
        from tkinter import messagebox, simpledialog
        if self.busy:
            return False
        # Decode first so a cancelled dialog or invalid file preserves current work.
        try:
            image_path = Path(path).resolve()
            page_index = 0
            if is_tiff(image_path):
                with tiff_module().TiffFile(image_path) as source:
                    page_count = len(source.pages)
                if page_count > 1:
                    page_number = simpledialog.askinteger(
                        "选择 TIFF 页面", f"此文件有 {page_count} 页，请输入要分割的页码（1～{page_count}）：",
                        initialvalue=1, minvalue=1, maxvalue=page_count, parent=self.root)
                    if page_number is None:
                        return False
                    page_index = page_number - 1
            image, metadata = read_input_image(image_path, page_index)
        except Exception as error:
            messagebox.showerror("打开图像失败", str(error), parent=self.root)
            return False
        self.image, self.image_path = image, image_path
        self.image_metadata = metadata
        self.boxes.clear()
        self.points.clear()
        self.history.clear()
        self.start = None
        if clear_text:
            self.text.set("")
        self.invalidate()
        self.reset_view()
        self.redraw()
        self.run_button.configure(state="normal")
        self.root.title(f"SAM3 PCS — {image_path.name} ({image.width} × {image.height})")
        details = ""
        if metadata.get("format") == "TIFF":
            details = f'；第 {metadata["page_index"] + 1}/{metadata["page_count"]} 页，{metadata["source_dtype"]}'
            if metadata.get("display_range_1_99_percentile") is not None:
                details += "，已映射亮度至 8 位 RGB"
            if metadata.get("source_channels", 0) > 4:
                details += "，显示及检测使用前 3 个波段"
        self.status.set(f"已打开：{image_path}{details}。请添加提示后运行 PCS。")
        return True

    def layout(self, canvas):
        w, h = self.image.size
        scale = min(max(1, canvas.winfo_width()) / w,
                    max(1, canvas.winfo_height()) / h)
        if canvas is self.canvases[1]:
            scale *= self.zoom
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        offset = ((canvas.winfo_width() - size[0]) / 2,
                  (canvas.winfo_height() - size[1]) / 2)
        if canvas is self.canvases[1]:
            offset = tuple(o + p for o, p in zip(offset, self.pan))
        return offset, size

    def redraw(self, only_result=False):
        from PIL import Image, ImageTk
        for i, canvas in enumerate(self.canvases):
            if only_result and i == 0:
                continue
            if self.image is None:
                canvas.delete("all")
                canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
                                   text="点击“打开图像”选择图片", fill="white")
                continue
            offset, size = self.layout(canvas)
            source = self.image if i == 0 or self.overlay is None else self.overlay
            if i == 1:
                # Render just the viewport; a large zoom never allocates a huge bitmap.
                sx, sy = source.width / size[0], source.height / size[1]
                view = source.transform(
                    (max(1, canvas.winfo_width()), max(1, canvas.winfo_height())),
                    Image.Transform.AFFINE,
                    (sx, 0, -offset[0] * sx, 0, sy, -offset[1] * sy),
                    Image.Resampling.BILINEAR, fillcolor="#20242b")
            else:
                view = source.resize(size, Image.Resampling.LANCZOS)
            self.photos[i] = ImageTk.PhotoImage(view)
            canvas.delete("all")
            canvas.create_image(*(offset if i == 0 else (0, 0)), anchor="nw", image=self.photos[i])
            if i == 0:
                sx, sy = size[0] / self.image.width, size[1] / self.image.height
                for index, (box, positive) in enumerate(self.boxes, 1):
                    x0, y0, x1, y1 = box
                    xy = (offset[0] + x0 * sx, offset[1] + y0 * sy,
                          offset[0] + x1 * sx, offset[1] + y1 * sy)
                    color = "#40ff80" if positive else "#ff5050"
                    canvas.create_rectangle(*xy, outline=color, width=2)
                    canvas.create_text(xy[0] + 3, xy[1] - 3, anchor="sw", fill=color,
                                       text=f'B{index}', tags="box_label")
                for index, ((x, y), positive) in enumerate(self.points, 1):
                    cx, cy = offset[0] + x * sx, offset[1] + y * sy
                    color = "#40ff80" if positive else "#ff5050"
                    canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5,
                                       fill=color, outline="white", width=2, tags="prompt_point")
                    canvas.create_text(cx + 9, cy, anchor="w", fill=color, text=f'P{index}')

    def reset_view(self):
        self.zoom, self.pan = 1.0, (0.0, 0.0)
        self.pan_start = None
        self.zoom_label.set("结果缩放 100%")
        self.redraw(only_result=True)

    def wheel(self, event):
        if self.image is None:
            return "break"
        direction = (1 if event.num == 4 else -1) if getattr(event, "num", None) in (4, 5) else (1 if event.delta > 0 else -1)
        canvas = self.canvases[1]
        old_offset, old_size = self.layout(canvas)
        self.zoom = max(0.25, min(16.0, self.zoom * 1.2 ** direction))
        new_offset, new_size = self.layout(canvas)
        wanted = zoom_offset(old_offset, old_size, new_size, (event.x, event.y))
        self.pan = tuple(p + target - actual for p, target, actual in zip(self.pan, wanted, new_offset))
        self.zoom_label.set(f"结果缩放 {self.zoom:.0%}")
        self.redraw(only_result=True)
        return "break"

    def pan_press(self, event):
        if self.image is not None:
            self.pan_start = (event.x, event.y)

    def pan_drag(self, event):
        if self.pan_start is not None:
            self.pan = (self.pan[0] + event.x - self.pan_start[0],
                        self.pan[1] + event.y - self.pan_start[1])
            self.pan_start = (event.x, event.y)
            self.redraw(only_result=True)

    def pan_release(self, event):
        self.pan_start = None

    def point(self, event):
        offset, size = self.layout(self.canvases[0])
        return canvas_point(event.x, event.y, offset, size, self.image.size)

    def press(self, event):
        if self.busy or self.image is None:
            return
        offset, size = self.layout(self.canvases[0])
        if all(o <= v <= o + s for v, o, s in zip((event.x, event.y), offset, size)):
            if self.tool.get() == "point":
                self.start = None
                self.points.append((self.point(event), self.positive.get()))
                self.history.append("point")
                self.invalidate()
                self.redraw()
            else:
                self.start = self.point(event)

    def drag(self, event):
        if self.start is None or self.busy:
            return
        canvas = self.canvases[0]
        offset, size = self.layout(canvas)
        end = self.point(event)
        xy = [offset[i] + point[i] * size[i] / self.image.size[i]
              for point in (self.start, end) for i in range(2)]
        canvas.delete("draft")
        canvas.create_rectangle(*xy, outline="#40ff80" if self.positive.get() else "#ff5050",
                                width=2, dash=(5, 3), tags="draft")

    def release(self, event):
        if self.start is None or self.busy:
            return
        end = self.point(event)
        x0, x1 = sorted((self.start[0], end[0]))
        y0, y1 = sorted((self.start[1], end[1]))
        self.start = None
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            self.boxes.append(((x0, y0, x1, y1), self.positive.get()))
            self.history.append("box")
            self.invalidate()
        self.redraw()

    def invalidate(self):
        self.result = self.overlay = self.snapshot = None
        self.save_button.configure(state="disabled")
        self.status.set(f"已设置 {len(self.boxes)} 个框、{len(self.points)} 个点。点击运行以更新结果。")

    def undo(self):
        if not self.busy and self.history:
            (self.points if self.history.pop() == "point" else self.boxes).pop()
            self.invalidate()
            self.redraw()

    def clear(self):
        if not self.busy:
            self.boxes.clear()
            self.points.clear()
            self.history.clear()
            self.start = None
            self.invalidate()
            self.redraw()

    def run(self):
        from tkinter import messagebox
        if self.busy or self.image is None:
            return
        text = self.text.get().strip()
        try:
            threshold = float(self.threshold.get())
            if not 0 <= threshold <= 1:
                raise ValueError("置信度必须在 0 到 1 之间。")
            if not text and not self.boxes and not self.points:
                raise ValueError("请输入文本或添加至少一个提示框/提示点。")
        except ValueError as error:
            messagebox.showerror("提示参数错误", str(error))
            return
        self.invalidate()
        self.redraw()
        self.busy = True
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.status.set("加载模型 / 执行推理中……首次运行可能需要下载权重。")
        snapshot = {"text": text, "threshold": threshold, "prompts": list(self.boxes),
                    "points": list(self.points), "point_format": "xy_original_pixels",
                    "image": str(self.image_path)}
        image = self.image

        def worker():
            try:
                result = self.predictor.predict(image, text, snapshot["prompts"], threshold,
                                                snapshot["points"])
                self.events.put((result, snapshot, None))
            except Exception:
                self.events.put((None, None, traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        from tkinter import messagebox
        try:
            result, snapshot, error = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            self.run_button.configure(state="normal")
            self.open_button.configure(state="normal")
            if error:
                print(error, file=sys.stderr)
                self.status.set("推理失败，详情见弹窗和终端。")
                messagebox.showerror("SAM3 推理失败", error)
            else:
                try:
                    self.result, self.snapshot = result, snapshot
                    self.overlay = self.make_overlay()
                    self.redraw()
                    self.save_button.configure(state="normal")
                    self.status.set(f'完成：{len(result["scores"])} 个实例；'
                                    f'文本：{snapshot["text"] or "(仅视觉提示)"}；'
                                    f'阈值：{snapshot["threshold"]}。修改提示后请重新运行。')
                except Exception as error:
                    self.invalidate()
                    messagebox.showerror("结果显示失败", str(error))
        self.root.after(100, self.poll)

    def make_overlay(self):
        import numpy as np
        from PIL import Image, ImageDraw
        pixels = np.array(self.image, dtype=np.float32)
        for i, mask in enumerate(self.result["masks"]):
            mask = mask.reshape(self.image.height, self.image.width).astype(bool)
            pixels[mask] = pixels[mask] * 0.55 + np.array(self.COLORS[i % len(self.COLORS)]) * 0.45
        overlay = Image.fromarray(pixels.astype(np.uint8))
        draw = ImageDraw.Draw(overlay)
        for i, (box, score) in enumerate(zip(self.result["boxes"], self.result["scores"])):
            color = self.COLORS[i % len(self.COLORS)]
            draw.rectangle(tuple(box), outline=color, width=3)
            draw.text((float(box[0]) + 3, float(box[1]) + 3), f"#{i + 1} {score:.3f}", fill=color)
        return overlay

    def save(self):
        from tkinter import messagebox
        import numpy as np
        from PIL import Image
        if self.result is None or self.busy:
            return
        try:
            folder = Path(self.args.output) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            folder.mkdir(parents=True, exist_ok=False)
            self.overlay.save(folder / "overlay.png")
            masks = self.result["masks"].reshape(-1, self.image.height, self.image.width)
            for i, mask in enumerate(masks, 1):
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"mask_{i:03d}.png")
            np.savez_compressed(folder / "instances.npz", masks=masks,
                                boxes=self.result["boxes"], scores=self.result["scores"])
            metadata = {"image": str(self.image_path),
                        "input_conversion": self.image_metadata,
                        "image_size": list(self.image.size), "exif_transposed": True,
                        "checkpoint": self.args.checkpoint or "facebook/sam3",
                        **self.snapshot, "box_format": "xyxy_original_pixels",
                        "boxes": self.result["boxes"].tolist(),
                        "scores": self.result["scores"].tolist()}
            (folder / "result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.set(f"结果已保存：{folder.resolve()}")
        except Exception as error:
            messagebox.showerror("保存失败", str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional initial image; otherwise use Open Image in the UI")
    parser.add_argument("--checkpoint", help="Local SAM3 .pt checkpoint; otherwise download from HF")
    parser.add_argument("--text", default="", help="Initial text prompt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/pcs", help="Result output directory")
    args = parser.parse_args()
    for name in ("image", "checkpoint"):
        value = getattr(args, name)
        if value and not Path(value).is_file():
            parser.error(f"{name} file does not exist: {value}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    # Allow direct invocation from any working directory without importing an older installation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import tkinter as tk
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
