"""
WSSS training script (Weakly Supervised Semantic Segmentation).
"""

import argparse
import json
import os

import mlflow
import pytorch_lightning as pl
import torch

from callbacks import MemoryLogCallback, SaveEveryEpochCallback
from datamodule import WSSDataModule
from dataset import DatasetConfig
from model import WSSSegmentationModel

config = None


def main():
    """Main training function."""
    global config
    parser = argparse.ArgumentParser(description="WSSS training")
    parser.add_argument("--config", default="configs/default.json", help="Path to JSON config file")
    parser.add_argument("--dataset-root", default=None, help="Dataset root directory (overrides config)")
    parser.add_argument("--mlflow-creds", default=None, help="Path to MLflow credentials (JSON)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    if args.dataset_root is not None:
        config["dataset_root"] = args.dataset_root
    if args.mlflow_creds is not None:
        config["mlflow_creds_path"] = args.mlflow_creds

    pl.seed_everything(config.get("seed", 42))
    torch.set_float32_matmul_precision("high")

    print(f"\n{'='*60}")
    print(f"WSSS TRAINING")
    print(f"{'='*60}\n")

    dataset_root = config.get("dataset_root")
    if not dataset_root or not os.path.isdir(dataset_root):
        raise ValueError(
            "dataset_root is required and must be a directory with dataset.yaml and structure as described in plans/dataset.md. "
            "Use --dataset-root DIR or set dataset_root in config."
        )
    dsc = DatasetConfig(dataset_root)
    num_classes = dsc.num_classes
    if config.get("num_classes") is not None and config["num_classes"] != num_classes:
        print(f"Warning: config num_classes ({config['num_classes']}) differs from dataset.yaml ({num_classes}). Using dataset.yaml.")

    checkpoint_path = config.get("pretrain_checkpoint_path", None)
    freeze_backbone = config.get("freeze_backbone", False)

    if checkpoint_path and not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"   Training from scratch...")
        checkpoint_path = None

    model = WSSSegmentationModel(
        num_classes=num_classes,
        lr=config.get("lr", 2e-5),
        image_size=config.get("image_size", 448),
        freeze_backbone=freeze_backbone,
        config=config,
    )

    bg_weight = config.get("background_weight", 0.1)
    foreground_weight = config.get("foreground_weight", 1.0)
    num_fg_classes = num_classes - 1
    print(f"Segmentation class weights:")
    print(f"   Background (class 0): {bg_weight:.2f}")
    print(f"   Foreground classes (1-{num_fg_classes}): {foreground_weight:.2f}")
    print(f"   Total classes: {num_classes} (1 background + {num_fg_classes} foreground classes)")
    print(f"   Foreground/Background ratio: {foreground_weight/bg_weight:.1f}x\n")

    creds_path = config.get("mlflow_creds_path", "creds/mlflow.json")
    try:
        creds = json.load(open(creds_path))
        os.environ["MLFLOW_TRACKING_URI"] = creds["MLFLOW_TRACKING_URI"]
        os.environ["MLFLOW_TRACKING_USERNAME"] = creds["MLFLOW_TRACKING_USERNAME"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = creds["MLFLOW_TRACKING_PASSWORD"]
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    except FileNotFoundError:
        print(f"MLflow credentials file not found ({creds_path})")
        print("   MLflow logger will be disabled even if online=True")
        config["online"] = False
    except Exception as e:
        print(f"Error loading MLflow credentials: {e}")
        print("   MLflow logger will be disabled even if online=True")
        config["online"] = False

    from lightning.pytorch.loggers import CSVLogger, MLFlowLogger

    run_name = config.get("run_name", "wsss")
    if freeze_backbone:
        run_name += "_frozen"

    logger = CSVLogger("logs", name=run_name)

    if config.get("online", False):
        mlflow_logger = MLFlowLogger(
            experiment_name=config.get("experiment_name", "WSSS"),
            run_name=run_name,
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
            synchronous=True,
        )
        logger = [logger, mlflow_logger]
    else:
        logger = [logger]

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    callbacks_list = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        SaveEveryEpochCallback(dirpath=checkpoint_dir),
    ]
    callbacks_list.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{val/miou/mIoU:.4f}",
            monitor="val/miou/mIoU",
            mode="max",
            save_last=True,
        )
    )

    debug = config.get("debug", False)
    enable_profiling = config.get("enable_profiling", False)
    profiler_dir = config.get("profiler_dir", "profiler_logs/wsss")
    if enable_profiling:
        callbacks_list.append(
            MemoryLogCallback(log_path=os.path.join(profiler_dir, "memory_log.txt"))
        )
        profiler = None
    if enable_profiling:
        # Schedule to cover validation at epoch 4: with limit_*_batches=2, ~4 steps/epoch; wait=14, warmup=2, active=6
        profiler = pl.profilers.PyTorchProfiler(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=14, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            dirpath=profiler_dir,
            filename="memory_summary",
            sort_by_key="self_cuda_memory_usage",
        )

    epochs = config.get("epochs", 50)
    eval_last_epoch_only = config.get("eval_last_epoch_only", False)
    check_val_every_n_epoch = epochs if eval_last_epoch_only else 1

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices="auto",
        inference_mode=False,
        limit_train_batches=2 if debug else None,
        limit_val_batches=2 if debug else None,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        logger=logger,
        use_distributed_sampler=False,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        callbacks=callbacks_list,
        enable_progress_bar=True,
        enable_model_summary=False,
        profiler=profiler,
    )

    datamodule = WSSDataModule(
        dataset_root=dataset_root,
        config=config,
        batch_size=config.get("batch_size", 8),
        image_size=config.get("image_size", 448),
        use_balanced_sampler=config.get("use_balanced_sampler", True),
        sampler_num_samples=config.get("sampler", {}).get("num_samples", 3000),
        fold=config.get("fold"),
        n_folds=config.get("n_folds"),
        num_workers=config.get("num_workers", 4),
        persistent_workers=config.get("persistent_workers", True),
    )

    print(f"Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {config.get('batch_size', 8)}")
    print(f"   Debug mode: {debug}")
    print(f"   Frozen backbone: {freeze_backbone}")
    if checkpoint_path:
        print(f"   Initial checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    memory_snapshot_path = None
    if enable_profiling and torch.cuda.is_available():
        try:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            memory_snapshot_path = os.path.join(profiler_dir, "oom_snapshot.pickle")
        except Exception as e:
            print(f"Warning: could not enable memory history: {e}")

    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        if memory_snapshot_path and torch.cuda.is_available():
            try:
                torch.cuda.memory._dump_snapshot(memory_snapshot_path)
                print(f"Memory snapshot saved at: {memory_snapshot_path}")
            except Exception as dump_err:
                print(f"Warning: could not save snapshot: {dump_err}")
        raise
    finally:
        if enable_profiling and torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"TRAINING FINISHED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
