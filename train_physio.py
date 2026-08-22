#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from tqdm import tqdm

from mme_rag.config import add_common_config_args, get_by_path, load_config_with_overrides
from mme_rag.constants import LABELS
from mme_rag.datasets import WESADWindows, build_wesad_tensor_cache
from mme_rag.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from mme_rag.metrics import classification_metrics
from sklearn.metrics import classification_report, confusion_matrix
from mme_rag.losses import masked_cross_entropy, masked_argmax, class_balanced_weight_from_labels, focal_loss
from mme_rag.models import MultimodalEmotionModel
from mme_rag.train_utils import (
    append_jsonl,
    amp_dtype_from_name,
    configure_torch,
    load_model_weights,
    log_config,
    make_writer, 
    save_checkpoint,
    seed_everything,
)


def evaluate(model, loader, device, amp_dtype, active_classes=None, return_preds: bool = False):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_conf = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["physio"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and device.type == "cuda")):
                out = model(physio=x)
                logits = out["physio_logits"]
                proto_logits = out["physio_proto_logits"]
                probs = torch.softmax(logits, dim=-1)
                loss = masked_cross_entropy(logits, y, active_classes) + 0.2 * masked_cross_entropy(proto_logits, y, active_classes)
            total_loss += float(loss.detach().cpu()) * y.numel()
            y_true.extend(y.cpu().tolist())
            preds = masked_argmax(logits, active_classes)
            y_pred.extend(preds.cpu().tolist())
            y_conf.extend(probs.max(dim=1).values.cpu().tolist())
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(y_true))
    if return_preds:
        return metrics, y_true, y_pred, y_conf
    return metrics


def main():
    parser = argparse.ArgumentParser()
    add_common_config_args(parser, "configs/wesad_physio.yaml")
    args = parser.parse_args()
    cfg = load_config_with_overrides(args)

    dist = init_distributed()
    configure_torch()
    seed_everything(int(get_by_path(cfg, "train.seed", 42)) + dist.rank)

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg.get("loss", {})
    simple_loss_mode = bool(loss_cfg.get("simple", False) or loss_cfg.get("simple_loss", False))

    if is_main_process():
        build_wesad_tensor_cache(
            wesad_root=data_cfg["wesad_root"],
            cache_dir=data_cfg.get("cache_dir", "cache/wesad_tensors"),
            channels=data_cfg.get("channels"),
            window_seconds=float(data_cfg.get("window_seconds", 30)),
            stride_seconds=float(data_cfg.get("stride_seconds", 15)),
            source_hz=int(data_cfg.get("source_hz", 700)),
            target_hz=int(data_cfg.get("target_hz", 32)),
            min_label_fraction=float(data_cfg.get("min_label_fraction", 0.8)),
            train_subjects=data_cfg.get("train_subjects"),
            val_subjects=data_cfg.get("val_subjects"),
            test_subjects=data_cfg.get("test_subjects"),
            rebuild=bool(data_cfg.get("rebuild_cache", False)),
        )
    barrier()

    train_ds = WESADWindows(data_cfg.get("cache_dir", "cache/wesad_tensors"), "train")
    val_ds = WESADWindows(data_cfg.get("cache_dir", "cache/wesad_tensors"), "val")
    physio_channels = int(train_ds.x.shape[1])
    active_classes = sorted(set(int(v) for v in train_ds.y.tolist()))
    if is_main_process():
        active_names = [LABELS[i] for i in active_classes]
        print(f"WESAD active classes: {active_classes} -> {active_names}")
        print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    use_cb_sampler = bool(train_cfg.get("use_class_balanced_sampler", False))
    train_sampler = None
    val_sampler = None
    loader_kwargs = dict(
        batch_size=int(train_cfg.get("batch_size", 256)),
        num_workers=int(train_cfg.get("num_workers", 8)),
        pin_memory=True,
        persistent_workers=bool(train_cfg.get("persistent_workers", False)),
    )
    if int(train_cfg.get("num_workers", 8)) > 0:
        loader_kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    if use_cb_sampler and not dist.distributed:
        counts = torch.bincount(train_ds.y, minlength=len(LABELS)).float()
        inv = torch.zeros_like(counts)
        nz = counts > 0
        inv[nz] = 1.0 / counts[nz]
        sample_weights = inv[train_ds.y].tolist()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, shuffle=False, drop_last=True, **loader_kwargs)
        if is_main_process():
            print("Using WeightedRandomSampler for class-balanced WESAD training")
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.distributed else None
        train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, drop_last=False, **loader_kwargs)
    if len(train_loader) == 0:
        raise ValueError(
            "WESAD train_loader has 0 batches. Reduce train.batch_size or world_size, "
            "or relax data.min_label_fraction/window settings."
        )

    model = MultimodalEmotionModel(
        num_classes=len(LABELS),
        physio_channels=physio_channels,
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        temperature=float(loss_cfg.get("temperature", 0.07)),
    ).to(dist.device)

    if bool(train_cfg.get("init_classifier_bias", False)):
        try:
            counts = torch.bincount(train_ds.y, minlength=len(LABELS)).float()
            counts = counts + 1e-6
            priors = counts / counts.sum()
            log_prior = torch.log(priors)
            if is_main_process():
                print("Initializing classifier.bias with log-prior of WESAD training labels")
            with torch.no_grad():
                if hasattr(model, "classifier") and model.classifier.bias is not None:
                    model.classifier.bias.data.copy_(log_prior.to(model.classifier.bias.device))
        except Exception:
            if is_main_process():
                print("Failed to initialize classifier bias; continuing with default init")

    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if dist.distributed:
        model = DDP(model, device_ids=[dist.local_rank] if dist.device.type == "cuda" else None, find_unused_parameters=True)

    class_weights = class_balanced_weight_from_labels(
        train_ds.y,
        num_classes=len(LABELS),
        active_classes=active_classes,
        max_weight=float(loss_cfg.get("max_class_weight", 5.0)),
    ).to(dist.device)

    if use_cb_sampler and not dist.distributed:
        class_weights = None
        if is_main_process():
            print("Disabled class_weights because using WeightedRandomSampler")
    if is_main_process() and class_weights is not None:
        print('WESAD class_weights:', class_weights.tolist())

    if simple_loss_mode:
        lr = float(train_cfg.get("simple_lr", 1e-3))
    else:
        lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    if simple_loss_mode:
        weight_decay = float(train_cfg.get("simple_weight_decay", weight_decay))

    classifier_lr_mult = float(train_cfg.get("classifier_lr_multiplier", 10.0))
    classifier_wd = float(train_cfg.get("classifier_weight_decay", 0.0))
    other_params = []
    classifier_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in n:
            classifier_params.append(p)
        else:
            other_params.append(p)
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay, "name": "backbone"})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": lr * classifier_lr_mult, "weight_decay": classifier_wd, "name": "classifier"})
    optimizer = torch.optim.AdamW(param_groups)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 5.0))
    if is_main_process():
        print(f"Using learning rate: {lr} weight_decay: {weight_decay} classifier_lr_mult: {classifier_lr_mult} max_grad_norm: {max_grad_norm}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg.get("epochs", 60)))
    warmup_epochs = int(train_cfg.get("warmup_classifier_epochs", 0))
    total_epochs = int(train_cfg.get("epochs", 60))
    backbone_lr = lr
    if warmup_epochs > 0:
        if other_params:
            for p in other_params:
                p.requires_grad = False
        for g in optimizer.param_groups:
            if g.get("name") == "backbone":
                g["orig_lr"] = g.get("lr", backbone_lr)
                g["lr"] = 0.0
        if is_main_process():
            print(f"Warmup enabled: freezing backbone for {warmup_epochs} epochs")
    amp_dtype = amp_dtype_from_name(train_cfg.get("amp_dtype", "bf16"))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and dist.device.type == "cuda"))

    if amp_dtype == torch.bfloat16 and dist.device.type == "cuda":
        supports_bf16 = False
        try:
            supports_bf16 = torch.cuda.is_bf16_supported()
        except Exception:
            try:
                dev = int(dist.local_rank) if hasattr(dist, "local_rank") else 0
                cc = torch.cuda.get_device_capability(dev)
                supports_bf16 = cc[0] >= 8
            except Exception:
                supports_bf16 = False
        if not supports_bf16:
            if is_main_process():
                print("Warning: bf16 not supported on this CUDA device; falling back to fp32/autocast disabled.")
            amp_dtype = None
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    ce_w = float(loss_cfg.get("ce_weight", 1.0))
    proto_w = float(loss_cfg.get("prototype_weight", 0.2))
    if simple_loss_mode:
        proto_w = float(loss_cfg.get("simple_proto_weight", 0.0))


    out_dir = Path(train_cfg.get("out", "runs/wesad_physio"))
    writer = make_writer(out_dir)
    log_config(writer, cfg)
    best_metric = -1.0
    debug_verbose = bool(train_cfg.get("debug_verbose_metrics", False) or simple_loss_mode)

    for epoch in range(1, int(train_cfg.get("epochs", 60)) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if warmup_epochs > 0 and epoch == (warmup_epochs + 1):
            if other_params:
                for p in other_params:
                    p.requires_grad = True
            for g in optimizer.param_groups:
                if g.get("name") == "backbone":
                    g["lr"] = g.pop("orig_lr", backbone_lr)
            remaining = total_epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, remaining))
            if is_main_process():
                print(f"Unfreezing backbone at epoch {epoch}; remaining epochs={remaining}")
        model.train()
        start = time.time()
        total_loss = 0.0
        total_samples = 0
        iterator = train_loader if not is_main_process() else tqdm(train_loader, desc=f"epoch {epoch}")
        for batch_idx, batch in enumerate(iterator):
            x = batch["physio"].to(dist.device, non_blocking=True)
            y = batch["label"].to(dist.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and dist.device.type == "cuda")):
                out = model(physio=x)
                try:
                    emb = out.get("physio_embedding")
                    logits_debug = out.get("physio_logits")
                except Exception:
                    emb = None
                    logits_debug = None
                if bool(loss_cfg.get("use_focal", False)):
                    gamma = float(loss_cfg.get("focal_gamma", 2.0))
                    loss = ce_w * focal_loss(out["physio_logits"], y, active_classes, gamma=gamma, weight=class_weights, label_smoothing=float(loss_cfg.get("label_smoothing", 0.03)))
                else:
                    loss = ce_w * masked_cross_entropy(out["physio_logits"], y, active_classes, weight=class_weights, label_smoothing=float(loss_cfg.get("label_smoothing", 0.03)))
                if proto_w != 0.0:
                    loss = loss + proto_w * masked_cross_entropy(out["physio_proto_logits"], y, active_classes, weight=class_weights, label_smoothing=float(loss_cfg.get("label_smoothing", 0.03)))
            if batch_idx == 0 and epoch == 1 and debug_verbose and is_main_process():
                try:
                    preds_sample = masked_argmax(out["physio_logits"], active_classes).cpu().tolist()
                    labels_sample = y.cpu().tolist()
                    print("[debug] first-batch labels (first 32):", labels_sample[:32])
                    print("[debug] first-batch preds  (first 32):", preds_sample[:32])
                    from collections import Counter
                    print("[debug] pred distribution:", Counter(preds_sample))
                    if emb is not None:
                        try:
                            emb_mean = float(emb.mean().detach().cpu())
                            emb_std = float(emb.std().detach().cpu())
                            emb_norm_mean = float(emb.norm(dim=1).mean().detach().cpu())
                            print(f"[debug] physio embedding mean={emb_mean:.6f} std={emb_std:.6f} norm_mean={emb_norm_mean:.6f}")
                        except Exception:
                            pass
                    if logits_debug is not None:
                        try:
                            probs_dbg = torch.softmax(logits_debug, dim=-1)
                            mean_prob = probs_dbg.mean(dim=0).detach().cpu().tolist()
                            print(f"[debug] mean softmax per-class (first 10): {mean_prob[:10]}")
                        except Exception:
                            pass
                except Exception:
                    pass
            scaler.scale(loss).backward()
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
            if batch_idx == 0 and epoch == 1 and debug_verbose and is_main_process():
                try:
                    target_model = model.module if hasattr(model, "module") else model
                    if hasattr(target_model, "classifier") and target_model.classifier.weight.grad is not None:
                        gnorm = float(target_model.classifier.weight.grad.norm().detach().cpu())
                        print(f"[debug] classifier.weight.grad norm (post-unscale): {gnorm:.6f}")
                except Exception:
                    pass
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            except Exception:
                pass
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * y.numel()
            total_samples += y.numel()
        scheduler.step()

        debug_verbose = bool(train_cfg.get("debug_verbose_metrics", False) or simple_loss_mode)
        if len(val_ds):
            if debug_verbose:
                val, y_true, y_pred = evaluate(model, val_loader, dist.device, amp_dtype, active_classes, return_preds=True)
                if is_main_process():
                    try:
                        labels_for_report = list(active_classes)
                        names = [LABELS[i] for i in labels_for_report]
                    except Exception:
                        labels_for_report = None
                        names = None
                    print("Validation classification report:")
                    try:
                        print(classification_report(y_true, y_pred, labels=labels_for_report, target_names=names, zero_division=0))
                    except Exception:
                        pass
                    try:
                        print("Confusion matrix:")
                        print(confusion_matrix(y_true, y_pred, labels=labels_for_report))
                    except Exception:
                        pass
            else:
                val = evaluate(model, val_loader, dist.device, amp_dtype, active_classes)
        else:
            val = {"loss": 0, "accuracy": 0, "macro_f1": 0, "weighted_f1": 0}
        train_loss = total_loss / max(1, total_samples)
        samples_per_sec = (total_samples * dist.world_size) / max(1e-6, time.time() - start)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}, "samples_per_second_global": samples_per_sec}
        append_jsonl(out_dir / "metrics.jsonl", row)
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            writer.add_scalar("val/loss", val["loss"], epoch)
            writer.add_scalar("val/accuracy", val["accuracy"], epoch)
            writer.add_scalar("val/macro_f1", val["macro_f1"], epoch)
            writer.add_scalar("val/weighted_f1", val["weighted_f1"], epoch)
            writer.add_scalar("perf/samples_per_second_global", samples_per_sec, epoch)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, row, cfg)
        metric = val.get(str(train_cfg.get("save_best_metric", "macro_f1")), val.get("macro_f1", 0.0))
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, row, cfg)
        if is_main_process():
            print(row)

    if writer is not None:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
