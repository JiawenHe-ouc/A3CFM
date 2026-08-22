from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard is optional for smoke tests / minimal installs
    SummaryWriter = None  # type: ignore

from .distributed import is_main_process, unwrap_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def amp_dtype_from_name(name: str | None):
    if name is None or str(name).lower() in ("none", "fp32", "float32"):
        return None
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unknown amp dtype: {name}")


def make_writer(out_dir: str | os.PathLike) -> Optional[SummaryWriter]:
    if not is_main_process() or SummaryWriter is None:
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(out_dir))


def log_config(writer: Optional[SummaryWriter], cfg: Dict[str, Any]) -> None:
    if writer is not None:
        text = "```yaml\n" + json.dumps(cfg, indent=2, ensure_ascii=False) + "\n```"
        writer.add_text("run/config", text, 0)


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    if not is_main_process():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
    }
    torch.save(obj, path)


def load_model_weights(model: torch.nn.Module, checkpoint: str | os.PathLike, strict: bool = False, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=map_location)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {"missing": missing, "unexpected": unexpected, "checkpoint": str(checkpoint)}


def append_jsonl(path: str | os.PathLike, row: Dict[str, Any]) -> None:
    if not is_main_process():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
