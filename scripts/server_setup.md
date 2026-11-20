# 将本地 PCS 评估版本同步到服务器

目标仓库：`https://github.com/zhouh02/sam3`

本次版本分支：`codex/pcs-evaluation-20261120`。该分支基于本地已验证代码，不覆盖远端 main 的独立修改。

## 1. 登录服务器并获取代码

在本地终端登录（密码在 SSH 终端中输入，不写入命令、文件或 Git）：

```bash
ssh zh2025@10.12.54.21
```

在服务器终端执行：

```bash
cd /ssd-data3/zh2025
git clone --branch codex/pcs-evaluation-20261120 --single-branch https://github.com/zhouh02/sam3.git sam3-pcs-eval
cd /ssd-data3/zh2025/sam3-pcs-eval
git log -1 --oneline
```

使用独立目录 `sam3-pcs-eval`，避免覆盖已有 `sam3` 工作目录。如果目录已存在，先检查它：

```bash
git -C /ssd-data3/zh2025/sam3-pcs-eval remote -v
git -C /ssd-data3/zh2025/sam3-pcs-eval status --short
git -C /ssd-data3/zh2025/sam3-pcs-eval branch --show-current
```

仅在确认是目标仓库、位于本次分支、且工作区没有未提交修改后更新：

```bash
git -C /ssd-data3/zh2025/sam3-pcs-eval pull --ff-only
```

私有仓库需要服务器自己的 GitHub 授权。若服务器无法连接 GitHub，先解决网络访问；本地 SSH 可连接不等于服务器能访问 GitHub。

## 2. 推理环境

若服务器已有可用的 SAM3 环境，激活该环境，在仓库目录运行 `python -m pip install -e .` 即可更新代码引用；不要无故覆盖已有 PyTorch。

需要新环境时，可参考本地版本创建 Python 3.12 环境：

```bash
conda create -n sam3-pcs python=3.12 -y
conda activate sam3-pcs
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-pcs-eval.txt
python -m pip install -e .
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported())"
```

上述 CUDA 12.8 wheel 来自本地已用版本；服务器 GPU 和驱动兼容性尚未验证。依赖清单记录本地主要包版本，不是跨平台完整锁文件。无图形桌面的服务器运行命令行评估即可，Tk GUI 需要图形显示与 tkinter。

## 3. 权重与数据

本次 Git 提交只包含代码、说明、测试和评估指标，不包含 `sam3.pt`、数据集、完整预测和 Excel 实验表。

使用服务器已有的 SAM3 原始权重和 COCO 数据。若需要单独传输本地权重，可在本地 PowerShell 执行：

```powershell
scp E:\Code\sam3\sam3.pt zh2025@10.12.54.21:/ssd-data3/zh2025/sam3-pcs-eval/sam3.pt
```

这是可选命令，当前没有执行。评估记录中的权重 SHA256 为：

```text
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

服务器可用 `sha256sum /实际路径/sam3.pt` 核对。完整 COCO val2017 需原始 5,000 张图及对应 `instances_val2017.json`；本地现有 JSON 为 328 张子集。

## 4. 检查后运行评估

以下两个路径变量必须替换为服务器实际路径：

```bash
cd /ssd-data3/zh2025/sam3-pcs-eval
COCO_ROOT=/实际路径/COCO
SAM3_CHECKPOINT=/实际路径/sam3.pt
python scripts/eval_coco_single_box.py "$COCO_ROOT" --prepare-only
python scripts/eval_coco_single_box.py "$COCO_ROOT" --checkpoint "$SAM3_CHECKPOINT" --limit 3
```

确认小规模测试速度和显存正常后，移除 `--limit 3` 运行完整标注范围。需要保持 SSH 断开后继续运行时，可以进入已有的 tmux 会话，或使用：

```bash
mkdir -p outputs
RUN_TAG=$(date +%Y%m%d_%H%M%S)
nohup python -u scripts/eval_coco_single_box.py "$COCO_ROOT" --checkpoint "$SAM3_CHECKPOINT" > "outputs/coco_${RUN_TAG}.log" 2>&1 &
tail -f "outputs/coco_${RUN_TAG}.log"
```

结果在 `outputs/coco_single_box/<时间戳>/metrics.json`，默认同时包含 bbox 和 segm AP⁺。这条命令只会启动评估，不安装额外模型或上传结果。

完整参数见 `scripts/eval_coco_single_box.md`；多数据集 IC baseline 见 `scripts/eval_pcs_box_baseline.md`。
