# 通用 COCO 单框提示 SAM3 PCS baseline

对于与 `E:/Code/Datasets/ic/coco_pcs` 相同的目录组织，最简单的用法是直接传入根目录：

```powershell
conda activate sam3
python scripts/eval_pcs_box_baseline.py "E:/Code/Datasets/ic/coco_pcs"
```

脚本自动定位根目录下的 `annotations/instances_test2017.json` 和 `images/test2017/`，默认使用本代码仓库中的 `sam3.pt`，无需再输入 `--split`、标注或图片目录。启动时打印实际使用的路径；缺失测试集时报错，不会回退到训练集或验证集。

同样结构的多个数据集可以直接连续传入：

```powershell
python scripts/eval_pcs_box_baseline.py "E:/Datasets/dataset_a" "E:/Datasets/dataset_b"
```

仅检查自动定位和数据有效性、不运行模型时，在命令末尾加 `--prepare-only`。完整运行默认输出到 `outputs/pcs_baseline/<时间戳>/`。

运行位置为 SAM3 仓库根目录，使用已配置 CUDA 的 `sam3` 环境：

```powershell
conda activate sam3
python scripts/eval_pcs_box_baseline.py --dataset "E:/Code/Datasets/ic/coco_pcs" --checkpoint "sam3.pt" --seed 42
```

依赖为当前 SAM3 推理依赖以及 `pycocotools`、Pillow、NumPy。本机环境已具备。脚本不依赖 GUI，不修改数据集，不下载权重。

## 指定一个或多个数据集

脚本不内置数据集路径，直接传入根目录或使用以下命名参数指定数据来源。输入格式为 COCO 实例分割标注（polygon 或 RLE），不是任意数据格式的自动转换工具。

### 标准目录结构

`--dataset ROOT --split SPLIT` 读取 `ROOT/annotations/instances_SPLIT.json` 和 `ROOT/images/SPLIT/`；`--split` 默认 `test2017`。

```powershell
# 单数据集，指定划分
python scripts/eval_pcs_box_baseline.py --dataset "E:/Datasets/dataset_a" --split test --checkpoint "sam3.pt"

# 多数据集，同一划分；重复 --dataset
python scripts/eval_pcs_box_baseline.py --dataset "E:/Datasets/dataset_a" --dataset "E:/Datasets/dataset_b" --split test --checkpoint "sam3.pt"
```

### 自定义目录结构

```powershell
python scripts/eval_pcs_box_baseline.py --annotations "E:/Datasets/custom/test.json" --image-root "E:/Datasets/custom/tiles" --split test --checkpoint "sam3.pt"
```

`--image-root` 是标注中 `images[].file_name` 的相对路径基准。`--annotations` 和 `--image-root` 必须一起使用，不能与 `--dataset` 混用。

### 不同路径/划分的批量配置

创建 JSON 文件，例如 `datasets.json`：

```json
[
  {"name": "dataset_a", "root": "E:/Datasets/dataset_a", "split": "test2017"},
  {"name": "dataset_b", "annotations": "E:/Datasets/dataset_b/test.json", "image_root": "E:/Datasets/dataset_b/tiles", "split": "test"}
]
```

```powershell
python scripts/eval_pcs_box_baseline.py --datasets-config datasets.json --checkpoint sam3.pt --output outputs/pcs_baseline/batch_run
```

配置中的相对路径以 JSON 文件所在目录为基准。每项填写 `root` 或成对的 `annotations` / `image_root`；缺省 `split` 使用命令行的 `--split`。多个数据集共用模型权重和 seed/threshold/limit 参数；模型只加载一次，依次评估，每个数据集的图片状态独立。

多数据集结果写入 `01_<name>/`、`02_<name>/` 等独立子目录，根目录 `datasets_summary.json` 汇总已完成的数据集指标，不混合不同数据集的 AP。单数据集仍直接在输出目录写入结果。发生错误时停止后续评估，已完成的数据集文件会保留。`--limit` 对每个数据集分别生效。

## 评估协议

- 数据集和划分由参数指定。此前 IC 数据集的单次结果位于 `outputs/pcs_baseline/seed42_test/`，升级脚本不会覆盖这些结果。
- 每张小图从所有非空实例中均匀抽取一个；以 `seed:image_id` 构造独立伪随机序列，按 annotation ID 排序抽样，避免遍历顺序或 `--limit` 改变同一张图的提示。
- 从该实例的真实二值掩码计算紧致、轴对齐外接矩形。像素范围使用 `[xmin, ymin, xmax+1, ymax+1]`，再转换为归一化 `cxcywh`。不是旋转矩形，也不将矩形内所有目标视为提示实例。
- 使用原始 SAM3 开源权重，单个正框，**不加文本/点提示**。每张图重新创建图像状态，避免跨图累积提示。推理使用 CUDA BF16 autocast，不进行训练。
- PCS 检测范围是整张小图；对照该图**抽中类别的全部实例**评估，包含提供提示框的实例。单类别数据等价于评价图中全部实例；多类别时不评价当前图中未被抽中的类别，预测类别由提示实例决定。`evaluation_gt.json` 保存实际参与评估的标注。该结果属于 GT 框提示的条件分割 baseline，不是全类别自动检测评估。
- 支持单类别和多类别数据，当前仍要求无 crowd/ignore 标注；发现不支持的数据会报错。没有可抽样目标的图片会记录为跳过，不能静默当作已评估。无目标图片不参与该提示式协议。
- 保留边界截断目标和微小目标，不额外筛选。COCO 标注坐标对应原始存储像素，因此不应用 EXIF 旋转。

## 指标

1. COCO `segm` 和 `bbox` AP、AP50、AP75、AP_small/medium/large、AR；采用标准 IoU 0.50:0.05:0.95、maxDets=[1,10,100]。`null` 表示该尺寸分组无有效 GT 等不可定义情形。分割评估按预测掩码面积划分尺寸组。
2. 前景联合掩码 IoU/Dice：将分数大于 `--union-threshold` 的预测实例合并，与当前图抽中类别的全部 GT 的并集比较。报告逐图平均 macro 和累积像素 micro，不能替代实例 AP。

`--score-threshold` 默认为 0.0，尽量保留完整分数排序用于 AP。`--union-threshold` 默认为 0.5，且不能小于前者。概率掩码二值化阈值为 0.5。固定种子保证提示采样可复现，但不保证不同硬件、CUDA/PyTorch 版本上的输出逐位一致。

## 常用选项

```powershell
# 只检查数据并生成抽样清单，不加载模型
python scripts/eval_pcs_box_baseline.py --dataset "E:/Datasets/dataset_a" --prepare-only

# 前 2 张图的小规模验证（不代表完整测试集指标）
python scripts/eval_pcs_box_baseline.py --dataset "E:/Datasets/dataset_a" --limit 2

# 指定新的结果目录，已有目录会拒绝覆盖
python scripts/eval_pcs_box_baseline.py --dataset "E:/Datasets/dataset_a" --seed 42 --output "outputs/pcs_baseline/my_run"
```

不指定 `--output` 时使用 `outputs/pcs_baseline/<时间戳>/`。

- `config.json`：权重路径、数据集标注 SHA256、参数、GPU/软件版本、评估协议及运行状态；只有成功计算完整指标后才标为 `complete`。
- `prompts.json`：每张图抽中的 annotation ID、提示框像素和归一化坐标。
- `predictions.json`：COCO 格式预测，包括 RLE 掩码、检测框和分数。
- `per_image.jsonl`：每张图实例数、前景 IoU/Dice、像素统计、推理耗时；逐张写入便于查看进度。
- `metrics.json`：整体评估指标。
- `coco_segm_curves.npz`、`coco_bbox_curves.npz`：COCO precision、recall、score 数组，保留 IoU、类别、面积和 maxDets 维度。

首次加载权重、图片读取和评估计算不计入逐图推理时间；逐图时间包括图像特征计算、提示推理和预测结果回传 CPU。建议后续用不同种子重复运行，报告均值和标准差；不要根据测试集指标选择最优种子或调阈值。
