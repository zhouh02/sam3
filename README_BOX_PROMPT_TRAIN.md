# 实验 B：原生 box 提示前端微调

入口：`scripts/train_box_prompt.py`。正式训练在 Linux + CUDA 服务器运行，
默认 BF16；本地 Windows 只做 `--dry-run` 和单元验证。
本实现复用 SAM3 原有 Trainer、匹配器、检测与分割损失，不修改原生模型结构。

## 训练范围与比较条件

只更新以下模块，共 7 个参数张量、3,279,616 个可训练参数：

```text
geometry_encoder.boxes_direct_project.{weight,bias}
geometry_encoder.boxes_pool_project.{weight,bias}
geometry_encoder.boxes_pos_enc_project.{weight,bias}
geometry_encoder.label_embed.weight
```

三层 `geometry_encoder.encode`、公共投影/归一化、CLS embedding、
图像/文本 backbone、检测和分割网络均冻结。公共融合层仍正常前向、传递输入梯度，
仅 backbone 使用 `no_grad`。保持与 mask Stage 1 相同的 `train()`/`eval()` 行为：
冻结参数不意味着关闭训练时的 dropout。没有 LoRA，没有身份降权。
优化器白名单和首次更新检查会验证这四个模块的梯度及更新；
运行目录记录完整可训练参数列表。

默认协议对齐 `sam3-mask-pcs-stage1/mask_prompt/configs/stage1.yaml`：

| 项目 | 设置 |
| --- | --- |
| 输入 | 同图一个正 box + 类别文本 |
| 提示实例 | 训练随机；验证按 mask 面积中位数确定，面积相同按源标注顺序 |
| box 来源 | 所选实例 mask 的紧外接框；目标检测框仍使用原 COCO bbox |
| 监督 | 所选类别全部实例，包含提示实例 |
| 类别筛选 | 每图该类别至少有 2 个带 segmentation 的实例 |
| 类别分组 | 每个数据项至多 2 个类别查询；不生成负类别查询 |
| 图像变换 | 1008×1008 resize、归一化；无新增 box jitter |
| 优化器 | AdamW，固定学习率 1e-4，weight decay 0.1，bias 不衰减 |
| 预算 | 默认 3 epoch，每 GPU batch size 1，梯度裁剪 0.1 |
| 损失 | 原生分类/presence、box/GIoU、mask/Dice，保留 O2O/O2M 与辅助层配置 |
| 验证 | 每轮 bbox AP、segm AP；GT 限于实际被查询的图像/类别 |

验证指标包含提示实例，尚未实现排除提示实例的指标。不要把它解释成未提示实例 AP。
如 E 使用身份降权，则 B/E 比较仍同时包含提示方案和损失策略的变化；
要只比较编码器，应使用 B/D（mask 侧身份相对权重全为 1）。
`--text-mode visual` 将类别名替换成原生视觉提示占位词 `visual`；
只有 mask 对照也采用同样文本条件时才使用。默认保持类别文本以对应现有 Stage 1。
学习率可通过 `--lr` 调整；应在相同调参预算下选模，而非使用测试集选模。

## 上传和安装

上传此项目的源码和 `sam3.pt` 到独立目录，例如 `/workspace/sam3-box-b`，
不要覆盖 mask 实验目录。若使用交付的源码 zip，先解压到该目录，再单独上传权重。
源码包不包含训练集、权重、历史结果或 Git 元数据。

服务器应先安装与其 CUDA 环境匹配的 PyTorch/torchvision；推荐 Python 3.12，
使用已经能运行 mask Stage 1 的环境时可复用依赖，但应确保导入的是本项目：

```bash
cd /workspace/sam3-box-b
python -m pip install -e '.[train]'
python -m pip install pycocotools opencv-python einops
python -c "import sam3, torch; print(sam3.__file__); print(torch.cuda.is_available())"
python -c "import triton; from sam3.train.loss.sam3_loss import Sam3LossWrapper; print('training dependencies OK')"
```

原生 focal loss 依赖 Triton。使用与服务器 PyTorch 相匹配的 Triton 版本
（Linux CUDA 版 PyTorch 通常会安装匹配依赖），不要照搬 Windows 的依赖环境。

脚本优先从当前源码目录导入 SAM3。BPE 文件支持
`sam3/assets/bpe_simple_vocab_16e6.txt.gz` 或 `assets/bpe_simple_vocab_16e6.txt.gz`，
也可以用 `--bpe-path` 指定。

## 数据与启动

默认数据目录：

```text
/data/ic/coco_pcs/
  images/train2017/...
  images/val2017/...
  annotations/instances_train2017.json
  annotations/instances_val2017.json
```

需要标准 COCO 实例分割标注。启动时检查图像存在、目标 mask 非空和 bbox 有效，
生成与查询范围一致的验证 GT，记录标注 SHA256。请使用与 mask 实验相同的数据版本。

先预览最终配置（不加载权重、不启动训练、不写输出）：

```bash
python scripts/train_box_prompt.py train \
  --data-root /data/ic/coco_pcs \
  --checkpoint /workspace/sam3-box-b/sam3.pt \
  --output outputs/box_b_ic --dry-run
```

正式单卡训练，只需这一条命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_box_prompt.py train \
  --data-root /data/ic/coco_pcs \
  --checkpoint /workspace/sam3-box-b/sam3.pt \
  --output outputs/box_b_ic \
  --epochs 3 --lr 1e-4 --batch-size 1 --gpus 1
```

新实验要求输出目录为空，防止误覆盖或混用旧权重。服务器上如需先做一次小验证：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_box_prompt.py train \
  --data-root /data/ic/coco_pcs --checkpoint sam3.pt \
  --output outputs/box_b_server_smoke --smoke-steps 1
```

`--smoke-steps` 只限制训练数据项及更新步数，仍会进行完整验证；它使用独立输出目录，
不应当作为正式结果。默认 workers=0 与 mask 配置一致，可用 `--workers` 调整。
多卡用 `CUDA_VISIBLE_DEVICES=0,1 ... --gpus 2`，脚本自行启动进程，不要再套一层 torchrun。
有效 batch size 是每 GPU batch size × GPU 数；对比实验要保持一致。

若数据不是默认目录结构，可显式传入：

```bash
python scripts/train_box_prompt.py train --checkpoint sam3.pt \
  --train-images /data/ic/train --train-annotations /data/ic/train.json \
  --val-images /data/ic/val --val-annotations /data/ic/val.json \
  --output outputs/box_b_ic
```

续训使用保存的配置和最新 checkpoint，只允许延长总 epoch 数：

```bash
python scripts/train_box_prompt.py resume --output outputs/box_b_ic --epochs 6
```

恢复模型、优化器、scaler、进度、随机数状态；要求 GPU 数及数据不变。
不承诺中断在 epoch 中途后的逐 batch 精确重放。每轮保留 checkpoint，可按验证 AP 选模。

单独验证某轮权重：

```bash
python scripts/train_box_prompt.py eval --data-root /data/ic/coco_pcs \
  --checkpoint outputs/box_b_ic/checkpoints/checkpoint_3.pt \
  --output outputs/box_b_ic_eval
```

训练 checkpoint 需用本脚本 `eval` 或 `sam3.train.box_prompt.model.build_model(initialization="trained", ...)`
加载；原生 builder 的官方权重读取器只提取 `detector.*`，不能直接用于这种 Trainer checkpoint。

## 输出与本地验证

```text
outputs/box_b_ic/
  config.yaml / config_resolved.yaml
  data_audit.json
  trainable_parameters.json
  pcs_val_ground_truth.json
  checkpoints/checkpoint.pt
  checkpoints/checkpoint_1.pt ...
  logs/log.txt / logs/train_stats.json / logs/val_stats.json
  tensorboard/
  dumps/bbox/ / dumps/segm/
```

`checkpoint.pt` 是最新一轮，未自动标为最优；按相同的验证指标和选模规则比较 B/D/E。
源码附有无需加载完整 SAM3 权重的本地测试：

```bash
python -m unittest discover -s test -p test_box_prompt_training.py -v
```

测试覆盖单 box 采样、全部同类 GT、非连续类别 ID、多查询拼接、验证 GT 筛选、
冻结融合层传递梯度、仅四个前端模块更新、权重完整性检查及续训配置。
本次交付仅完成这些本地验证和配置检查，未完成服务器端到端训练验证。
