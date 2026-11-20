# COCO 单框视觉提示评估（SAM 3 PCS / AP⁺）

本脚本使用现有 `sam3.pt` 和 CUDA BF16 推理环境，无需训练或重新下载权重。原有 `eval_pcs_box_baseline.py` 保留不变。

## 运行

在仓库根目录、已激活的 sam3 环境运行：

```powershell
python scripts/eval_coco_single_box.py "E:\Code\Datasets\COCO"
```

默认读取 `annotations/instances_val2017.json`，自动寻找 `images/val2017` 或 `val2017`。若两个位置都完整则要求显式指定，避免选错。默认评估全部标注图像，同时输出 bbox 和 segm AP⁺。

先检查全部图片尺寸、类别查询和抽框清单，不加载模型：

```powershell
python scripts/eval_coco_single_box.py "E:\Code\Datasets\COCO" --prepare-only
```

小规模推理检查（不是正式完整评估）：

```powershell
python scripts/eval_coco_single_box.py "E:\Code\Datasets\COCO" --limit 3
```

只关注论文 COCO bbox 指标：

```powershell
python scripts/eval_coco_single_box.py "E:\Code\Datasets\COCO" --iou-types bbox
```

显式指定任意 COCO 格式数据与权重：

```powershell
python scripts/eval_coco_single_box.py --annotations "E:\Code\Datasets\COCO\annotations\instances_val2017.json" --image-root "E:\Code\Datasets\COCO\images\val2017" --checkpoint "E:\Code\sam3\sam3.pt"
```

重用已保存的抽框清单（路径换成实际结果目录，所选图像范围必须一致）：

```powershell
python scripts/eval_coco_single_box.py "E:\Code\Datasets\COCO" --prompts "E:\Code\sam3\outputs\coco_single_box\YOUR_RUN\prompts.json"
```

`--output` 可以指定新的输出目录，已存在的目录不会覆盖。默认输出到 `outputs/coco_single_box/<时间戳>/`。

## 协议

- 对每个有标注的 `(image_id, category_id)` 查询一次，从该对的实例中均匀抽一个 GT bbox。每次只有一个正框，不提供类别文本、点或其他框；内部使用 `visual` 占位文本。
- 使用原始浮点 bbox；按官方 `Sam3ImageDataset` 裁剪到图像边界再归一化，不从 mask 重新计算框。
- 采样包括 crowd，与公开 `COCO_FROM_JSON` + `TextQueryToVisual` 路径一致。GT 不移除 crowd，评估器按 COCO 规则处理。所有同类别 GT 都参加评估，包括提供提示的实例。
- 每张图只编码一次，每次类别查询前 `reset_all_prompts`，防止提示累计。
- 默认 seed 42；每个图像—类别对独立播种，与遍历顺序无关。它不等同于作者未公开的确切抽框序列。`--prompts` 校验类别覆盖、annotation ID 和原始框后复用提示。
- 固定 1008 输入、官方 processor 的 0.5 mean/std、CUDA BF16、eval 模式，score 使用 detector probability × presence probability。置信度阈值 0，不按 0.5 提前过滤检测；mask 二值阈值仍为 0.5。
- 所有正类别查询结果合并，默认每张图跨类别保留最高分 100 个候选，参考公开 `odinw_visual_only.yaml` 的 `PredictionDumper`。可通过 `--max-dets-per-image` 调整，调整值会记录且影响可比性。COCOeval 的 maxDets 固定为 [1,10,100]。
- 直接调用仓库 `sam3.eval.coco_eval_offline.COCOevalCustom(..., dt_only_positive=True)`。不裁剪成每图一个类别的 GT。segm 加载预测时移除 bbox，保证面积分组按 mask 面积计算。
- 无标注图像保留在评估图像列表中，不进行单框查询。只推理正图像—类别对；这并不模拟公开 ODinW 配置中负类别查询参与候选合并的流程。

论文 Table 3 的 COCO I 模式报告 bbox AP⁺=76.8（本脚本 AP 输出范围 0–1，应乘 100 比较）。segm 是额外评估，不与 76.8 对比。本脚本实现公开协议，不承诺完整数值复现：COCO 专用配置、确切随机提示和作者环境仍需核实，不能把 ODinW 配置当成已确认的 COCO 配置。

完整 COCO val2017 应有 5,000 张图像；脚本按标注文件决定评估范围，数量不符时提示警告。仅复制完整图像而保留子集 JSON，仍只评估子集。

参考：https://arxiv.org/html/2511.16719v1#S6 ，以及仓库 `coco_eval_offline.py`、`filter_query_transforms.py`、`odinw_visual_only.yaml`。

## 输出

- `config.json`：协议、路径、标注/权重/提示 SHA256、代码 commit、环境与完成状态。
- `prompts.json`：每个查询的类别、原始框、归一化框、提示 annotation ID。
- `per_query.jsonl`：逐查询记录；`predictions.jsonl`：逐图合并后的预测，运行中及时写盘。
- `predictions.json`：完整 COCO 结果，可用于离线复核。
- `metrics.json`：`coco.bbox.AP`、`coco.segm.AP` 等 12 项指标，均为 AP⁺ 协议；没有可评估 GT 的分组显示 null。
- `coco_bbox_curves.npz` / `coco_segm_curves.npz`：官方评估精度、召回率和分数数组。

异常时 `config.json` 标记 failed，并保留已写入记录；当前不支持中断续跑。旧的 IC foreground_union 不纳入此脚本。
