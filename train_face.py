#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from mme_rag.config import add_common_config_args, get_by_path, load_config_with_overrides
from mme_rag.constants import LABELS
from mme_rag.datasets import FaceExpressionDataset, split_face_manifest
from mme_rag.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from mme_rag.metrics import classification_metrics
from mme_rag.models import MultimodalEmotionModel
from mme_rag.train_utils import append_jsonl, amp_dtype_from_name, configure_torch, log_config, make_writer, save_checkpoint, seed_everything
from mme_rag.models import AffectLoss


def _safe_loss_value(loss: torch.Tensor, fallback: float = 0.0) -> float:
    if not torch.isfinite(loss).all():
        return float(fallback)
    return float(loss.detach().cpu())


def evaluate(model, loader, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["face"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and device.type == "cuda")):
                out = model(face=x)
                # loss = F.cross_entropy(out["face_logits"], y) + 0.2 * F.cross_entropy(out["face_proto_logits"], y)
                loss = F.cross_entropy(out["face_logits"], y) # V2
            total_loss += float(loss.detach().cpu()) * y.numel()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(out["face_logits"].argmax(dim=-1).cpu().tolist())
    m = classification_metrics(y_true, y_pred)
    m["loss"] = total_loss / max(1, len(y_true))
    return m


def main():
    parser = argparse.ArgumentParser()
    add_common_config_args(parser, "configs/face_pretrain.yaml")
    args = parser.parse_args()
    cfg = load_config_with_overrides(args)

    dist = init_distributed()
    configure_torch()
    seed_everything(int(get_by_path(cfg, "train.seed", 42)) + dist.rank)

    data_cfg, train_cfg, model_cfg, loss_cfg = cfg["data"], cfg["train"], cfg["model"], cfg.get("loss", {})
    manifest = data_cfg.get("face_manifest", "data/face_manifest.csv")
    if bool(data_cfg.get("auto_split", False)):
        if is_main_process():
            split_face_manifest(manifest, data_cfg.get("split_manifest", manifest), seed=int(train_cfg.get("seed", 42)))
        barrier()
        manifest = data_cfg.get("split_manifest", manifest)
    if is_main_process():
        print(f"Using face manifest: {manifest}")

    train_ds = FaceExpressionDataset(
        manifest,
        image_size=int(data_cfg.get("image_size", 112)),
        split="train",
        random_horizontal_flip=True,
        corruption=data_cfg.get("train_corruption", {}),
    )
    val_ds = FaceExpressionDataset(manifest, image_size=int(data_cfg.get("image_size", 112)), split="val", random_horizontal_flip=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.distributed else None
    val_sampler = None
    loader_kwargs = dict(
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=int(train_cfg.get("num_workers", 8)),
        pin_memory=True,
        persistent_workers=int(train_cfg.get("num_workers", 8)) > 0,
    )
    if int(train_cfg.get("num_workers", 8)) > 0:
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, drop_last=False, **loader_kwargs)

# 模型结构
    model = MultimodalEmotionModel(
        num_classes=len(LABELS),
        physio_channels=int(model_cfg.get("physio_channels_placeholder", 8)),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        temperature=float(loss_cfg.get("temperature", 0.07)),
    ).to(dist.device)

    affect_loss_fn = AffectLoss(
    temperature=float(loss_cfg.get("temperature", 0.07)),
    ce_w=float(loss_cfg.get("ce_weight", 1.0)),
    supcon_w=float(loss_cfg.get("supcon_weight", 0.5)),
    hier_w=float(loss_cfg.get("hier_weight", 0.1)),
)
    
    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if dist.distributed:
        model = DDP(model, device_ids=[dist.local_rank] if dist.device.type == "cuda" else None, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-2)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg.get("epochs", 50)))
    amp_dtype = amp_dtype_from_name(train_cfg.get("amp_dtype", "none"))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and dist.device.type == "cuda"))

    out_dir = Path(train_cfg.get("out", "runs/face_pretrain"))
    writer = make_writer(out_dir)
    log_config(writer, cfg)
    best = -1.0

    for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train(); start = time.time(); total_loss = 0.0; total_samples = 0
        iterator = train_loader if not is_main_process() else tqdm(train_loader, desc=f"face epoch {epoch}")
        for batch in iterator:
            x = batch["face"].to(dist.device, non_blocking=True)
            y = batch["label"].to(dist.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and dist.device.type == "cuda")):
                out = model(face=x)
                raw_model = model.module if dist.distributed else model
                ce = F.cross_entropy(out["face_logits"], y)
                if float(loss_cfg.get("hier_weight", 0.0)) > 0.0:
                    loss, loss_detail = affect_loss_fn(
                        out, y,
                        label_names=LABELS,
                        manifold=raw_model.prototype_head,
                    )
                else:
                    proto_ce = F.cross_entropy(out["face_proto_logits"], y) if "face_proto_logits" in out else torch.zeros_like(ce)
                    loss = ce + 0.2 * proto_ce
                    loss_detail = {"ce": ce, "proto_ce": proto_ce, "supcon": torch.zeros_like(ce), "hier": torch.zeros_like(ce)}

            if not torch.isfinite(loss).all():
                print(f"[warn] non-finite loss at epoch {epoch}; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            total_loss += _safe_loss_value(loss, 0.0) * y.numel(); total_samples += y.numel()
        scheduler.step()
        val = evaluate(model, val_loader, dist.device, amp_dtype)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_samples), **{f"val_{k}": v for k, v in val.items()}, "samples_per_second_global": (total_samples * dist.world_size) / max(1e-6, time.time() - start)}
        append_jsonl(out_dir / "metrics.jsonl", row)
        if writer is not None:
            writer.add_scalar("train/loss", row["train_loss"], epoch); writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("val/loss", val["loss"], epoch); writer.add_scalar("val/accuracy", val["accuracy"], epoch); writer.add_scalar("val/macro_f1", val["macro_f1"], epoch)
            writer.add_scalar("perf/samples_per_second_global", row["samples_per_second_global"], epoch)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, row, cfg)
        if val["macro_f1"] > best:
            best = val["macro_f1"]; save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, row, cfg)
        if is_main_process(): print(row)

    if writer is not None: writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
