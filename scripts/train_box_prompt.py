"""Launch experiment B: python scripts/train_box_prompt.py train --data-root ..."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "resume", "eval"), nargs="?", default="train")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/box_prompt_b")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bpe-path", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--text-mode", choices=("category", "visual"))
    parser.add_argument("--smoke-steps", type=int, help="One epoch with this many updates per GPU; full validation")
    parser.add_argument("--dry-run", action="store_true", help="Print config without loading model or writing output")
    for split in ("train", "val"):
        parser.add_argument(f"--{split}-images", type=Path)
        parser.add_argument(f"--{split}-annotations", type=Path)
    args = parser.parse_args(argv)
    for key in ("epochs", "batch_size", "gpus", "smoke_steps"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.lr is not None and (not 0 < args.lr < float("inf")):
        parser.error("--lr must be positive and finite")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.mode == "resume":
        changes = ("data_root", "checkpoint", "bpe_path", "lr", "batch_size", "gpus", "workers", "seed", "text_mode", "smoke_steps",
                   "train_images", "train_annotations", "val_images", "val_annotations")
        if any(getattr(args, k) is not None for k in changes):
            parser.error("resume uses saved config; only --output, --epochs and --dry-run may be set")
        if args.epochs is None:
            parser.error("resume requires a cumulative --epochs budget")
    elif args.data_root is None and any(getattr(args, k) is None for k in (
            ("val_images", "val_annotations") if args.mode == "eval" else
            ("train_images", "train_annotations", "val_images", "val_annotations"))):
        parser.error("Supply --data-root or explicit image/annotation paths")
    if args.mode == "eval" and args.checkpoint is None:
        parser.error("eval requires --checkpoint from experiment B")
    if args.smoke_steps is not None and args.mode != "train":
        parser.error("--smoke-steps is only available for fresh training")
    return args


def configuration(args):
    from omegaconf import OmegaConf

    output = args.output.resolve()
    if args.mode == "resume":
        cfg = OmegaConf.load(output / "config.yaml")
        if Path(cfg.paths.experiment_log_dir) != output:
            raise ValueError("Resume from the original output directory")
        cfg.scratch.max_epochs = args.epochs
        cfg.trainer.model.initialization = "resume"
        cfg.trainer.model.checkpoint_path = str(output / "checkpoints/checkpoint.pt")
        return cfg
    cfg = OmegaConf.load(ROOT / "sam3/train/box_prompt/config.yaml")
    cfg.paths.experiment_log_dir = str(output)
    cfg.paths.coco_root = str((args.data_root or ROOT / "data/coco").resolve())
    cfg.paths.sam3_checkpoint = str((args.checkpoint or ROOT / "sam3.pt").resolve())
    vocabulary = next((p for p in (ROOT / "sam3/assets/bpe_simple_vocab_16e6.txt.gz", ROOT / "assets/bpe_simple_vocab_16e6.txt.gz") if p.is_file()), ROOT / "assets/bpe_simple_vocab_16e6.txt.gz")
    cfg.paths.bpe_path = str((args.bpe_path or vocabulary).resolve())
    for arg, key in (("epochs", "max_epochs"), ("lr", "stage1_lr"), ("batch_size", "train_batch_size"), ("workers", "num_train_workers")):
        if getattr(args, arg) is not None:
            cfg.scratch[key] = getattr(args, arg)
    cfg.scratch.num_val_workers = cfg.scratch.num_train_workers
    cfg.launcher.gpus_per_node = args.gpus or 1
    if args.seed is not None:
        cfg.trainer.seed_value = args.seed
    for split in ("train", "val"):
        ds = cfg.trainer.data[split].dataset
        for arg, key in ((f"{split}_images", "img_folder"), (f"{split}_annotations", "ann_file")):
            if getattr(args, arg) is not None:
                ds[key] = str(getattr(args, arg).resolve())
        ds.coco_json_loader.text_mode = args.text_mode or "category"
    for meter in cfg.trainer.meters.val.coco.values():
        meter.pred_file_evaluators[0].gt_path = str(output / "pcs_val_ground_truth.json")
    if args.smoke_steps:
        cfg.scratch.max_epochs = 1
        cfg.trainer.data.train.dataset.limit_ids = args.smoke_steps * cfg.launcher.gpus_per_node * cfg.scratch.train_batch_size
    if args.mode == "eval":
        cfg.trainer.mode = "val"
        cfg.trainer.model.initialization = "trained"
        cfg.trainer.model.eval_mode = True
        cfg.trainer.data.train = None
        cfg.trainer.skip_saving_ckpts = True
    cfg.trainer.distributed.backend = "nccl"
    # Keep interpolations: resume can then extend max_epochs without stale copies.
    return cfg


def run_worker(rank, port, config):
    import torch
    from hydra.utils import instantiate
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE=str(config.launcher.gpus_per_node))
    register_omegaconf_resolvers()
    try:
        instantiate(config.trainer, _recursive_=False).run()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main(argv=None):
    args = arguments(argv)
    cfg = configuration(args)
    from omegaconf import OmegaConf

    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if sys.platform == "win32":
        raise RuntimeError("Run training/evaluation on a Linux CUDA server. Local --dry-run and unit tests work on Windows.")
    import torch
    from sam3.train.box_prompt.data import audit_and_write_gt
    # Fail on missing server training dependencies before creating a run directory.
    from sam3.train.loss.sam3_loss import Sam3LossWrapper  # noqa: F401

    root = Path(cfg.paths.experiment_log_dir)
    if args.mode != "resume" and root.exists() and any(root.iterdir()):
        raise ValueError("Use an empty --output directory for fresh training/evaluation")
    if cfg.launcher.gpus_per_node > torch.cuda.device_count() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Experiment B requires CUDA with BF16 support")
    for path in (cfg.trainer.model.checkpoint_path, cfg.paths.bpe_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    reports = {}
    for split, spec in cfg.trainer.data.items():
        if spec is None:
            continue
        ds = spec.dataset
        reports[split] = audit_and_write_gt(ds.ann_file, ds.img_folder, None)
        if split == "train":
            available = reports[split]["datapoints"]
            limit = ds.get("limit_ids", available)
            if available < limit or limit < cfg.launcher.gpus_per_node * cfg.scratch.train_batch_size:
                raise ValueError("Not enough eligible training datapoints for the batch size/smoke budget")
    if args.mode == "resume":
        previous = json.loads((root / "data_audit.json").read_text(encoding="utf-8"))
        if reports != previous:
            raise ValueError("Resume requires unchanged annotations and dataset eligibility")
    root.mkdir(parents=True, exist_ok=True)
    val = cfg.trainer.data.val.dataset
    audit_and_write_gt(val.ann_file, val.img_folder, root / "pcs_val_ground_truth.json")
    OmegaConf.save(cfg, root / ("resume_config.yaml" if args.mode == "resume" else "config.yaml"))
    (root / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    (root / "data_audit.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    if cfg.launcher.gpus_per_node == 1:
        run_worker(0, port, cfg)
    else:
        torch.multiprocessing.spawn(run_worker, args=(port, cfg), nprocs=cfg.launcher.gpus_per_node)


if __name__ == "__main__":
    main()
