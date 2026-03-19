"""Callbacks for WSSS training."""

import os
import resource
import pytorch_lightning as pl
import torch


class MemoryLogCallback(pl.Callback):
    """Logs RSS (RAM) and VRAM at each batch for OOM analysis. RSS unit: KB (Linux)."""

    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path
        self._header_written = False

    def _log_line(self, trainer, phase: str, batch_idx: int, rss_kb: float, vram_mb: float):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "a") as f:
            if not self._header_written:
                f.write("epoch,global_step,phase,batch_idx,rss_kb,vram_mb\n")
                self._header_written = True
            epoch = trainer.current_epoch
            step = trainer.global_step
            f.write(f"{epoch},{step},{phase},{batch_idx},{rss_kb:.0f},{vram_mb:.2f}\n")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        self._log_line(trainer, "train", batch_idx, rss_kb, vram_mb)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        self._log_line(trainer, "val", batch_idx, rss_kb, vram_mb)


class SaveEveryEpochCallback(pl.Callback):
    """Saves a checkpoint at every epoch."""
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = dirpath

    def on_train_epoch_end(self, trainer, pl_module):
        os.makedirs(self.dirpath, exist_ok=True)
        epoch = trainer.current_epoch
        filepath = os.path.join(self.dirpath, f"epoch_{epoch:02d}.ckpt")
        trainer.save_checkpoint(filepath)
        print(f"Checkpoint saved: {filepath}")
