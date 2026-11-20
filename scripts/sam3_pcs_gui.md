# SAM3 图像 PCS 交互测试

## 仓库已有内容

- `examples/sam3_image_interactive.ipynb`：Jupyter 交互界面，支持图片、文本和正负框。
- `examples/sam3_image_predictor_example.ipynb`：文本提示、单框和多正负框推理示例。
- `sam3/visualization_utils.py`：分割结果和框的绘制工具。
- `scripts/qualitative_test.py`：视频文本检测、传播和结果输出测试，不是本任务所需的单图交互界面。

新增 `scripts/sam3_pcs_gui.py` 使用 Tkinter 桌面窗口，不需要 Jupyter 或 Web 服务。使用 `Sam3Processor` 的 PCS 接口，默认加载 SAM3 图像模型，而非 SAM 1/2 的单实例交互接口。

## 环境和权重

请在已按仓库根目录 README 安装好 SAM3 的 CUDA 环境运行。仓库 README 要求 Python 3.12+、PyTorch 2.7+ 和兼容的 CUDA GPU；本脚本使用 CUDA，不提供 CPU 回退。

在仓库根目录执行以下命令补充依赖（PyTorch/CUDA 请先按 README 安装）：

```powershell
python -m pip install -e .
python -m pip install pillow "numpy>=1.26,<2" einops tifffile
python -m tkinter
```

最后一条用于确认 Tk 窗口可打开，关闭该测试窗口后再启动脚本。Linux 系统可能需要安装系统的 `python3-tk`；WSL 需要可用的 GUI 显示环境，例如 WSLg。

本地兼容修复：模型构建器直接从包目录读取词表，不再依赖 `pkg_resources`；Triton 距离变换内核延迟到误差中心采样函数内导入，因此本脚本的图像 PCS 不需要安装该内核。其他训练/视频功能仍可能需要 Triton。脚本在推理线程中启用 CUDA BF16 自动混合精度。

使用本地已下载的 **SAM3 图像权重 `sam3.pt`**（不是 SAM3.1 视频权重）：

```powershell
python scripts/sam3_pcs_gui.py --image "assets/images/test_image.jpg" --checkpoint "E:/models/sam3.pt" --text "person"
```

也可以省略 `--checkpoint`，由仓库的构建函数下载官方 `facebook/sam3` 权重。按照仓库 README，需先申请该 Hugging Face 仓库的访问权限并登录：

```powershell
hf auth login
python scripts/sam3_pcs_gui.py --image "assets/images/test_image.jpg" --text "person" --threshold 0.5 --output "outputs/pcs"
```

图片和权重路径均可替换为自己的绝对路径。模型在首次点击运行时加载，随后复用模型和当前图像特征。点击“打开图像”或按 `Ctrl+O` 可切换图片；模型继续复用，新图片在下次运行时重新提取图像特征。

也可以不传 `--image`，先打开窗口再选择图片：

```powershell
python scripts/sam3_pcs_gui.py --checkpoint "sam3.pt"
```

切换图片会清空文本、框、点、旧分割结果和撤销记录，并重置缩放/平移；置信度阈值保留。取消选择或文件读取失败会保留当前内容。推理期间暂时禁止切换图片，完成或失败后恢复。保存结果中的路径和尺寸对应当前检测的图片。

### TIFF / TIF 输入

“打开图像”和 `--image` 均支持 TIFF，包括大小写扩展名。按文件头识别 TIFF，使用 `tifffile` 读取，普通 PNG/JPEG 的读取方式不变。

- 支持常见 8/16 位灰度、16 位 RGB、浮点、平面分离 RGB、常见多波段和 BigTIFF 容器。
- 多页文件弹窗选择页码（从 1 开始），只读取所选页；取消选择保留当前图像。当前不提供三维体数据分割。
- 8 位数据保留原亮度。高位深/浮点图像按有效像素的 1%～99% 分位区间映射为 8 位，显示和 SAM3 使用同一份 RGB 图像；RGB 通道共享映射范围。NaN、Inf 和 GDAL NoData 不参与范围计算。恒定值图像显示为均匀黑色；没有任何有效像素时报告错误。
- 灰度复制为 RGB；多于 3 个通道时取前 3 个波段，灰度加 alpha 取灰度通道。原始 TIFF 不会被修改。多光谱前三波段不一定代表自然色 RGB，可事先导出所需波段组合。
- 方向标签会应用到显示图像；提示坐标与输出掩码对应方向校正后的所选页。保存的 `result.json` 中 `input_conversion` 记录原始数据类型、页码（`page_index` 从 0 开始）、尺寸及亮度映射信息。输出 PNG/NPZ 不保留 GeoTIFF 的地理坐标参考。
- 常见灰度 LZW 压缩可使用 Pillow 解码回退。若特定压缩方式提示缺少编解码器，在当前环境运行 `python -m pip install imagecodecs` 后重启。不是所有专业 TIFF 色彩编码或高维布局都受支持；无法读取时弹窗报告具体原因，保留原有图片。

TIFF 回归测试：`python -m unittest discover -s test -p test_pcs_tiff.py -v`。已验证灰度/RGB 高位深、浮点无效值、平面 RGB、多页、LZW、方向标签、多波段及普通 PNG 输入。

## 操作

1. 左侧显示预设图片。选择“提示框”，按住鼠标左键拖动；选择“提示点”，单击图片添加点。分别选择“正提示（目标）”或“负提示（排除）”，绿色代表正提示，红色代表负提示。支持多个框和点、反向拖框、按添加顺序撤销和清空全部提示。
2. 输入目标概念文本，例如 `person`、`car`；推荐先使用简短英文概念进行测试。支持纯文本、纯框、纯点及文本/框/点组合。提示框以 `B1` 等框外标签标识，点以 `P1` 等圆点标识；画框不会自动添加点。旧版框角的 `1 +` 是框标签，并非提示点。
3. 点击“运行 PCS”。框按原图坐标转换为归一化 `cxcywh`，与文本共同传给 SAM3；每次运行清除上次提示，避免累积。
4. 右侧展示彩色实例掩码、检测框、实例编号和置信度。无检测时显示原图和 0 个实例。可调低阈值后重新运行。在右侧滚动滚轮可围绕鼠标位置缩放（适应窗口尺寸的 25%～1600%），按住左键拖动可平移，双击或点击“重置结果视图”可恢复适应窗口。缩放只影响显示，保存仍使用原图分辨率。
5. 点击“保存结果”，在输出目录下创建带时间戳的子目录。

**框是用于说明目标概念的视觉示例，不是裁剪区域，也不保证只分割框内对象。PCS 会在整张图像中检测并分割匹配实例。** 负框用于提供应排除的示例。

点通过几何编码器的 `append_points` 传入同一 PCS 分支，使用归一化 `xy` 坐标和正负标签；可与文本、框联合使用。这是 PCS 概念提示，不是切换到单实例交互分割分支，不保证所有输出掩码都包含所点击的位置。

推理期间提示框锁定；文本编辑不会改变正在执行的请求。结果和保存元数据始终对应点击运行时的提示快照，修改文本/阈值后需再次运行。

## 输出

- `overlay.png`：原图叠加实例掩码、检测框和分数。
- `mask_001.png` 等：每个实例的原图分辨率二值掩码，前景 255，背景 0；实例编号与数组顺序一致。
- `instances.npz`：`masks` 为 `N×H×W` 布尔数组，`boxes` 为 `N×4` 原图像素 `xyxy`，`scores` 为 `N`。
- `result.json`：图片/权重来源、文本、正负提示框、阈值、输出框和分数。提示框格式为 `[xyxy, positive]`；`points` 保存 `[[x, y], positive]`，坐标为原图像素。

带 EXIF 方向信息的图片会先旋转为正常显示方向，所有坐标、图像尺寸和掩码都对应旋转后的图像。各实例掩码分别保存，保留重叠部分。

## 验证

```powershell
python -m unittest discover -s test -p test_pcs_gui.py -v
python scripts/sam3_pcs_gui.py --help
```

上述测试不依赖模型，覆盖缩放/留白坐标映射、边界裁切、框归一化、文本和正负框组合，以及跨次推理清除提示。完整分割效果仍需在安装依赖并配置权重的 CUDA 环境中验证。

点提示/缩放更新验证：6 项逻辑测试通过；Tk 交互检查覆盖只画框时没有点、正负点添加、撤销/清空、鼠标位置缩放、平移和重置。使用真实权重完成纯正负点、文本加框加正负点，以及清除点后文本推理，并检查模型接收的点数量和正负标签与界面输入一致。

本次本机检查：4 项逻辑测试通过；在 `sam3` Conda 环境中完成 Tk 界面创建、叠加图生成，以及 0/2 个模拟实例的 PNG、NPZ、JSON 保存检查。修复导入依赖和混合精度后，使用本机 RTX 4060 Ti、PyTorch `2.10.0+cu128`、仓库根目录 `sam3.pt` 和 `assets/images/test_image.jpg` 完成真实推理：文本 `person` 在 0.5 阈值下返回 0 个实例；该文本加图像中央正框返回 7 个实例及原图分辨率掩码。这验证了推理链路，结果质量仍取决于具体图片和提示。

本机可直接运行：

```powershell
conda activate sam3
python scripts/sam3_pcs_gui.py --image "assets/images/test_image.jpg" --checkpoint "sam3.pt"
```
