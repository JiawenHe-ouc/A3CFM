#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import math
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from mme_rag.config import add_common_config_args, get_by_path, load_config_with_overrides
from mme_rag.constants import LABELS, LABEL_TO_ID
from mme_rag.datasets import (
    AudioEmotionDataset,
    FaceExpressionDataset,
    normalize_label,
    WESADWindows,
    build_wesad_tensor_cache,
    ensure_audio_manifest_from_config,
)
from mme_rag.distributed import barrier, cleanup_distributed, init_distributed, is_main_process
from mme_rag.losses import (
    pairwise_multiview_mmd_loss,
    pairwise_multiview_ot_loss,
    reliability_consistency_loss,
    supervised_contrastive_loss,
    temporal_smoothness_loss,
    masked_cross_entropy,
    masked_argmax,
    class_balanced_weight_from_labels,
)
from mme_rag.metrics import classification_metrics
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


def freeze_batchnorm_running_stats(module: torch.nn.Module) -> None:
    
    bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)
    target = module.module if hasattr(module, "module") else module
    for m in target.modules():
        if isinstance(m, bn_types):
            m.eval()



def cycle_loader(loader):
    while True:
        for b in loader:
            yield b


def _strip_checkpoint_wrappers(key: str) -> str:
    prefixes = ("module.", "model.", "network.", "net.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for pref in prefixes:
            if key.startswith(pref):
                key = key[len(pref):]
                changed = True
    return key


def _unique_suffix_match(
    source_key: str,
    source_value: torch.Tensor,
    target_state: Dict[str, torch.Tensor],
    allowed_prefixes: Optional[List[str]],
) -> Optional[str]:
    
    candidates = []
    src_parts = source_key.split(".")
    for n_parts in (4, 3, 2):
        if len(src_parts) < n_parts:
            continue
        suffix = ".".join(src_parts[-n_parts:])
        candidates = [
            k for k, v in target_state.items()
            if (not allowed_prefixes or any(k.startswith(pref) for pref in allowed_prefixes))
            and k.endswith(suffix)
            and tuple(v.shape) == tuple(source_value.shape)
        ]
        if len(candidates) == 1:
            return candidates[0]
    return None


def load_partial(
    model,
    ckpt_path: Optional[str],
    label: str,
    prefixes: Optional[List[str]] = None,
    remap_prefixes: Optional[Dict[str, str]] = None,
):
    
    if not ckpt_path or not Path(ckpt_path).exists():
        if is_main_process():
            print(f"Skip {label}: checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if not isinstance(raw_state, dict):
        raise TypeError(f"Unsupported checkpoint format for {label}: {type(raw_state)!r}")

    target = model.module if hasattr(model, "module") else model
    target_state = target.state_dict()
    remap_prefixes = remap_prefixes or {}
    state: Dict[str, torch.Tensor] = {}
    skipped_shape = []
    unmatched = []

    modality = None
    for m in ("audio", "physio", "face"):
        if prefixes and any(pref.startswith(m + "_") for pref in prefixes):
            modality = m
            break

    for raw_key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        key = _strip_checkpoint_wrappers(str(raw_key))

        mapped = key
        for old_pref, new_pref in remap_prefixes.items():
            if mapped.startswith(old_pref):
                mapped = new_pref + mapped[len(old_pref):]
                break

        aliases = []
        if modality is not None:
            aliases.extend([
                ("encoder.", f"{modality}_encoder."),
                ("backbone.", f"{modality}_encoder."),
                ("feature_extractor.", f"{modality}_encoder."),
                ("projector.", f"{modality}_affect_projector."),
                ("affect_projector.", f"{modality}_affect_projector."),
                ("hierarchical_affect_fusion.", f"{modality}_hierarchical_affect_fusion."),
            ])
        mapped_candidates = [mapped]
        for old_pref, new_pref in aliases:
            if key.startswith(old_pref):
                mapped_candidates.append(new_pref + key[len(old_pref):])

        chosen = None
        for cand in mapped_candidates:
            if prefixes and not any(cand.startswith(pref) for pref in prefixes):
                continue
            if cand in target_state:
                if tuple(target_state[cand].shape) == tuple(value.shape):
                    chosen = cand
                    break
                skipped_shape.append((key, cand, tuple(value.shape), tuple(target_state[cand].shape)))

        if chosen is None:
            chosen = _unique_suffix_match(key, value, target_state, prefixes)

        if chosen is None:
            unmatched.append(key)
            continue
        state[chosen] = value

    missing, unexpected = target.load_state_dict(state, strict=False)
    if is_main_process():
        print(
            f"Loaded {label}: {ckpt_path}; keys={len(state)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"unmatched={len(unmatched)}, shape_skipped={len(skipped_shape)}"
        )
        if len(state) == 0:
            sample = list(raw_state.keys())[:12]
            print(f"WARNING: no parameters were loaded from {label}. Sample checkpoint keys: {sample}")
        elif unmatched:
            print(f"  unmatched key examples: {unmatched[:8]}")
        if skipped_shape:
            print(f"  shape mismatch examples: {skipped_shape[:4]}")


def install_safe_face_region_patch(model: torch.nn.Module) -> int:
    
    target = model.module if hasattr(model, "module") else model
    patched = 0

    boxes = {
        "brow": (0.05, 0.38, 0.12, 0.88),
        "brows": (0.05, 0.38, 0.12, 0.88),
        "eyebrow": (0.05, 0.38, 0.12, 0.88),
        "eyebrows": (0.05, 0.38, 0.12, 0.88),
        "eye": (0.18, 0.52, 0.08, 0.92),
        "eyes": (0.18, 0.52, 0.08, 0.92),
        "left_eye": (0.18, 0.52, 0.05, 0.52),
        "right_eye": (0.18, 0.52, 0.48, 0.95),
        "nose": (0.30, 0.75, 0.30, 0.70),
        "mouth": (0.58, 0.98, 0.18, 0.82),
        "lip": (0.58, 0.98, 0.18, 0.82),
        "lips": (0.58, 0.98, 0.18, 0.82),
        "cheek": (0.38, 0.82, 0.02, 0.98),
        "cheeks": (0.38, 0.82, 0.02, 0.98),
        "jaw": (0.62, 1.00, 0.05, 0.95),
        "lower_face": (0.48, 1.00, 0.05, 0.95),
        "upper_face": (0.00, 0.58, 0.05, 0.95),
        "global": (0.00, 1.00, 0.00, 1.00),
        "face": (0.00, 1.00, 0.00, 1.00),
    }

    for module in target.modules():
        original = getattr(module, "_get_region_patches", None)
        if original is None or not callable(original):
            continue
        if getattr(module, "_safe_region_patch_installed", False):
            continue

        def safe_get_region_patches(self, patch_tokens, region_name):
            if patch_tokens.dim() != 3:
                raise ValueError(f"Expected face patch tokens [B,N,D], got {tuple(patch_tokens.shape)}")
            b, n, d = patch_tokens.shape
            if n <= 0:
                raise ValueError("Face encoder returned zero patch tokens")

            side_minus = int(math.isqrt(max(0, n - 1)))
            if n > 1 and side_minus * side_minus == n - 1:
                tokens = patch_tokens[:, 1:, :]
                side = side_minus
            else:
                side = int(math.isqrt(n))
                if side * side == n:
                    tokens = patch_tokens
                else:
                    side = int(math.ceil(math.sqrt(n)))
                    target_n = side * side
                    pad_n = target_n - n
                    pad = patch_tokens.mean(dim=1, keepdim=True).expand(b, pad_n, d)
                    tokens = torch.cat([patch_tokens, pad], dim=1)

            grid = tokens.contiguous().reshape(b, side, side, d)
            key = str(region_name).strip().lower().replace("-", "_").replace(" ", "_")
            r0f, r1f, c0f, c1f = boxes.get(key, boxes["global"])

            r0 = max(0, min(side - 1, int(math.floor(r0f * side))))
            r1 = max(r0 + 1, min(side, int(math.ceil(r1f * side))))
            c0 = max(0, min(side - 1, int(math.floor(c0f * side))))
            c1 = max(c0 + 1, min(side, int(math.ceil(c1f * side))))

            region = grid[:, r0:r1, c0:c1, :].contiguous().reshape(b, -1, d)
            if region.shape[1] == 0:
                region = grid.contiguous().reshape(b, side * side, d)
            return region

        module._get_region_patches = types.MethodType(safe_get_region_patches, module)
        module._safe_region_patch_installed = True
        patched += 1
    return patched

def same_label_indices(base_y: torch.Tensor, other_y: torch.Tensor) -> Tuple[List[int], List[int]]:
    base_idx, other_idx = [], []
    for i, y in enumerate(base_y.tolist()):
        matches = (other_y == y).nonzero(as_tuple=False).view(-1)
        if matches.numel() > 0:
            j = matches[torch.randint(0, matches.numel(), (1,), device=other_y.device)].item()
            base_idx.append(i); other_idx.append(j)
    return base_idx, other_idx


def make_pseudo_multiview_batch(batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:

    keys = [k for k in ["audio", "physio", "face"] if k in batch]
    if len(keys) < 2:
        return {}, None, None

    anchor = keys[0]
    _, y_anchor = batch[anchor]
    device = y_anchor.device

    label_lookup: Dict[str, Dict[int, List[int]]] = {}
    for k in keys[1:]:
        _, yk = batch[k]
        lookup: Dict[int, List[int]] = {}
        for idx, label in enumerate(yk.tolist()):
            label = int(label)
            lookup.setdefault(label, []).append(idx)
        label_lookup[k] = lookup

    selected: Dict[str, List[int]] = {k: [] for k in keys}
    pseudo_labels: List[torch.Tensor] = []

    for i in range(y_anchor.numel()):
        y = int(y_anchor[i].item())
        chosen = {anchor: i}
        ok = True

        for k in keys[1:]:
            matches = label_lookup.get(k, {}).get(y)
            if not matches:
                ok = False
                break
            chosen[k] = int(matches[0])

        if not ok:
            continue

        for k in keys:
            selected[k].append(int(chosen[k]))
        pseudo_labels.append(y_anchor[i].to(device=device))

    if not pseudo_labels:
        return {}, None, None

    out: Dict[str, torch.Tensor] = {}
    idxs: Dict[str, torch.Tensor] = {}
    for k in keys:
        x, _ = batch[k]
        idx = torch.tensor(selected[k], device=x.device, dtype=torch.long)
        out[k] = x.index_select(0, idx)
        idxs[k] = idx

    py = torch.stack(pseudo_labels).to(device=device, dtype=y_anchor.dtype)
    return out, py, idxs


def make_dummy_multiview_batch(batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], keys: Tuple[str, ...]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:

    pseudo: Dict[str, torch.Tensor] = {}
    idxs: Dict[str, torch.Tensor] = {}
    for k in keys:
        x, _ = batch[k]
        pseudo[k] = x[:1]
        idxs[k] = torch.tensor([0], device=x.device, dtype=torch.long)
    _, y0 = batch[keys[0]]
    return pseudo, y0[:1], idxs


def fused_output_from_embeddings(
    model,
    embeddings: Dict[str, torch.Tensor],
    reliabilities: Dict[str, Optional[torch.Tensor]],
    prompt_ids: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    target = model.module if hasattr(model, "module") else model
    z_fused, modality_weights, aux = target.fusion(
        embeddings.get("audio"),
        embeddings.get("physio"),
        embeddings.get("face"),
        reliabilities.get("audio"),
        reliabilities.get("physio"),
        reliabilities.get("face"),
    )
    z_fused = target.prompt_conditioner(z_fused, prompt_ids)
    fused_out = target.classify_embedding(z_fused)
    out = {f"fused_{k}": v for k, v in fused_out.items()}
    out["fusion_modality_weights"] = modality_weights
    out["fusion_modality_mask"] = aux.get("modality_mask")
    out["fusion_gate_audio"] = modality_weights[:, 0:1]
    out["fusion_gate_physio"] = modality_weights[:, 1:2]
    out["fusion_gate_face"] = modality_weights[:, 2:3]
    out["fusion_router_weights"] = aux["router_weights"]
    if "cross_attention" in aux:
        out["fusion_cross_attention"] = aux["cross_attention"]
    return out


def evaluate(model, loaders: Dict[str, DataLoader], device, amp_dtype, active_by_mod: Optional[Dict[str, List[int]]] = None, prompt_ids: Optional[torch.Tensor] = None):
    model.eval()
    y_true: Dict[str, List[int]] = {"audio": [], "physio": [], "face": [], "fusion": []}
    y_pred: Dict[str, List[int]] = {"audio": [], "physio": [], "face": [], "fusion": []}
    with torch.no_grad():
        # Evaluate unimodal heads.
        for mod, loader in loaders.items():
            for batch in loader:
                x = batch[mod].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                kwargs = {mod: x}
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and device.type == "cuda")):
                    out = model(**kwargs, prompt_ids=prompt_ids)
                y_true[mod].extend(y.cpu().tolist())
                y_pred[mod].extend(masked_argmax(out[f"{mod}_logits"], (active_by_mod or {}).get(mod)).cpu().tolist())
                if len(y_true[mod]) > 1500:
                    break
        val_keys = [k for k in ["audio", "physio", "face"] if k in loaders]
        if len(val_keys) >= 2:
            cyc = {k: cycle_loader(loaders[k]) for k in val_keys}
            steps = min(len(loaders[k]) for k in val_keys)
            active_sets = [set((active_by_mod or {}).get(k, [])) for k in val_keys]
            fusion_active = sorted(set.intersection(*active_sets)) if active_sets and all(active_sets) else []
            if not fusion_active:
                fusion_active = sorted(set().union(*active_sets)) if active_sets else None
            for _ in range(min(steps, 50)):
                raw = {k: next(cyc[k]) for k in val_keys}
                modal_batch = {}
                for k, b in raw.items():
                    modal_batch[k] = (b[k].to(device, non_blocking=True), b["label"].to(device, non_blocking=True))
                pseudo, py, _ = make_pseudo_multiview_batch(modal_batch)
                if py is None:
                    continue
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and device.type == "cuda")):
                    out = model(**pseudo, prompt_ids=prompt_ids)
                y_true["fusion"].extend(py.cpu().tolist())
                y_pred["fusion"].extend(masked_argmax(out["fused_logits"], fusion_active).cpu().tolist())
    row = {}
    for k in y_true:
        if y_true[k]:
            m = classification_metrics(y_true[k], y_pred[k])
            row[f"{k}_accuracy"] = m["accuracy"]; row[f"{k}_macro_f1"] = m["macro_f1"]
    return row


def main():
    parser = argparse.ArgumentParser()
    add_common_config_args(parser, "configs/triview_alignment.yaml")
    args = parser.parse_args()
    cfg = load_config_with_overrides(args)

    dist = init_distributed()
    configure_torch()
    seed_everything(int(get_by_path(cfg, "train.seed", 42)) + dist.rank)
    if is_main_process():
        print(
            f"Distributed init: distributed={dist.distributed} rank={dist.rank} "
            f"world_size={dist.world_size} local_rank={dist.local_rank} device={dist.device}"
        )

    data_cfg, train_cfg, model_cfg, loss_cfg = cfg["data"], cfg["train"], cfg["model"], cfg.get("loss", {})

    simple_loss_mode = bool(loss_cfg.get("simple", False) or loss_cfg.get("simple_loss", False))
    use_distribution_losses = (not simple_loss_mode) and bool(loss_cfg.get("use_distribution_losses", True))
    eval_every = max(1, int(train_cfg.get("eval_every", 2)))

    # Physiological view
    if is_main_process():
        build_wesad_tensor_cache(
            wesad_root=data_cfg["wesad_root"],
            cache_dir=data_cfg.get("wesad_cache_dir", "cache/wesad_tensors"),
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
    physio_train = WESADWindows(data_cfg.get("wesad_cache_dir", "cache/wesad_tensors"), "train")
    physio_val = WESADWindows(data_cfg.get("wesad_cache_dir", "cache/wesad_tensors"), "val")

    # Audio view
    if str(data_cfg.get("dataset_format", "manifest")).lower() == "iemocap":
        if is_main_process():
            audio_manifest = ensure_audio_manifest_from_config(data_cfg)
        barrier()
        if not is_main_process():
            data_cfg_worker = dict(data_cfg); data_cfg_worker["prepare_manifest"] = False; data_cfg_worker["force_rebuild_manifest"] = False
            audio_manifest = ensure_audio_manifest_from_config(data_cfg_worker)
    else:
        audio_manifest = ensure_audio_manifest_from_config(data_cfg)
    audio_include_labels = data_cfg.get("audio_include_labels") or data_cfg.get("include_labels")
    audio_exclude_labels = data_cfg.get("audio_exclude_labels") or data_cfg.get("exclude_labels")
    audio_train = AudioEmotionDataset(
        audio_manifest,
        data_cfg.get("sample_rate", 16000),
        data_cfg.get("clip_seconds", 6.0),
        split="train",
        random_crop=True,
        include_labels=audio_include_labels,
        exclude_labels=audio_exclude_labels,
    )
    audio_val = AudioEmotionDataset(
        audio_manifest,
        data_cfg.get("sample_rate", 16000),
        data_cfg.get("clip_seconds", 6.0),
        split="val",
        random_crop=False,
        include_labels=audio_include_labels,
        exclude_labels=audio_exclude_labels,
    )

    physio_label_ids = physio_train.y.detach().cpu().long()
    audio_label_ids = torch.tensor([LABEL_TO_ID[normalize_label(r["label"])] for r in audio_train.rows], dtype=torch.long)
    active_by_mod: Dict[str, List[int]] = {
        "physio": sorted(set(int(v) for v in physio_label_ids.tolist())),
        "audio": sorted(set(int(v) for v in audio_label_ids.tolist())),
    }
    label_ids_by_mod: Dict[str, torch.Tensor] = {"physio": physio_label_ids, "audio": audio_label_ids}

    enable_face = bool(data_cfg.get("enable_face", True))
    face_train = face_val = None
    if enable_face:
        face_manifest = data_cfg.get("face_manifest", "data/face_manifest.csv")
        if not Path(face_manifest).exists():
            raise FileNotFoundError(f"Face manifest not found: {face_manifest}. Set data.enable_face=false for two-view training.")
        face_train = FaceExpressionDataset(face_manifest, int(data_cfg.get("image_size", 112)), split="train", random_horizontal_flip=True, corruption=data_cfg.get("face_train_corruption", {}))
        face_val = FaceExpressionDataset(face_manifest, int(data_cfg.get("image_size", 112)), split="val", random_horizontal_flip=False)
        face_label_ids = torch.tensor([LABEL_TO_ID[str(r["label"])] for r in face_train.rows], dtype=torch.long)
        active_by_mod["face"] = sorted(set(int(v) for v in face_label_ids.tolist()))
        label_ids_by_mod["face"] = face_label_ids

    if is_main_process():
        print(f"Audio manifest: {audio_manifest}")
        print(f"Face enabled: {enable_face}")
        for mod, ids in active_by_mod.items():
            counts = {LABELS[i]: int((label_ids_by_mod[mod] == i).sum()) for i in ids}
            print(f"{mod} active classes: {ids} -> {[LABELS[i] for i in ids]}; train counts={counts}")

    def make_loader(ds, shuffle=True, drop_last=True):
        sampler = DistributedSampler(ds, shuffle=shuffle) if dist.distributed else None
        num_workers = int(train_cfg.get("num_workers", 8))
        persistent = bool(train_cfg.get("persistent_workers", False))
        kwargs = dict(batch_size=int(train_cfg.get("batch_size", 64)), num_workers=num_workers, pin_memory=bool(train_cfg.get("pin_memory", dist.device.type == "cuda")), persistent_workers=(persistent and num_workers > 0))
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        return DataLoader(ds, sampler=sampler, shuffle=(sampler is None and shuffle), drop_last=drop_last, **kwargs), sampler

    audio_loader, sampler_a = make_loader(audio_train)
    physio_loader, sampler_p = make_loader(physio_train)
    face_loader = sampler_v = None
    if face_train is not None:
        face_loader, sampler_v = make_loader(face_train)
    if len(audio_loader) == 0 or len(physio_loader) == 0 or (face_loader is not None and len(face_loader) == 0):
        lengths = {
            "audio": len(audio_loader),
            "physio": len(physio_loader),
            "face": len(face_loader) if face_loader is not None else None,
        }
        raise ValueError(
            f"At least one Stage2 train loader has 0 batches: {lengths}. "
            "Reduce train.batch_size/world_size or relax label filtering."
        )
    val_loaders = {
        "audio": DataLoader(audio_val, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)), pin_memory=bool(train_cfg.get("pin_memory", dist.device.type == "cuda"))),
        "physio": DataLoader(physio_val, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)), pin_memory=bool(train_cfg.get("pin_memory", dist.device.type == "cuda"))),
    }
    if face_val is not None:
        val_loaders["face"] = DataLoader(face_val, batch_size=int(train_cfg.get("batch_size", 64)), shuffle=False, num_workers=int(train_cfg.get("num_workers", 8)), pin_memory=True)

    model = MultimodalEmotionModel(
        num_classes=len(LABELS),
        physio_channels=int(physio_train.x.shape[1]),
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        temperature=float(loss_cfg.get("temperature", 0.07)),
        fusion_heads=int(model_cfg.get("fusion_heads", 4)),
    ).to(dist.device)
    patched_region_modules = install_safe_face_region_patch(model)
    if is_main_process() and patched_region_modules:
        print(f"Installed safe face patch-grid handling on {patched_region_modules} local-region module(s).")
    init = cfg.get("init", {})
    load_partial(
        model,
        init.get("physio_checkpoint"),
        "physio checkpoint",
        prefixes=["physio_encoder.", "physio_affect_projector.", "physio_hierarchical_affect_fusion."],
        remap_prefixes={"affect_projector.": "physio_affect_projector.", "hierarchical_affect_fusion.": "physio_hierarchical_affect_fusion."},
    )
    load_partial(
        model,
        init.get("audio_checkpoint"),
        "audio checkpoint",
        prefixes=["audio_encoder.", "audio_affect_projector.", "audio_hierarchical_affect_fusion."],
        remap_prefixes={
            "wavlm.": "audio_encoder.wavlm.",
            "audio_projection.": "audio_encoder.temporal_proj.",
            "temporal_proj.": "audio_encoder.temporal_proj.",
            "pool_score.": "audio_encoder.pool_score.",
            "pooled_proj.": "audio_encoder.pooled_proj.",
            "affect_projector.": "audio_affect_projector.",
            "hierarchical_affect_fusion.": "audio_hierarchical_affect_fusion.",
        },
    )
    load_partial(
        model,
        init.get("face_checkpoint"),
        "face checkpoint",
        prefixes=["face_encoder.", "face_affect_projector.", "face_hierarchical_affect_fusion."],
        remap_prefixes={"affect_projector.": "face_affect_projector.", "hierarchical_affect_fusion.": "face_hierarchical_affect_fusion."},
    )

    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    if dist.distributed:
        find_unused = bool(train_cfg.get("find_unused_parameters", True))
        ddp_kwargs = {
            "find_unused_parameters": find_unused,
            "broadcast_buffers": False,
        }
        if dist.device.type == "cuda":
            ddp_kwargs["device_ids"] = [dist.local_rank]
            ddp_kwargs["output_device"] = dist.local_rank
        model = DDP(model, **ddp_kwargs)

    class_weights_by_mod = {
        mod: class_balanced_weight_from_labels(
            ids,
            num_classes=len(LABELS),
            active_classes=active_by_mod[mod],
            max_weight=float(loss_cfg.get("max_class_weight", 5.0)),
        ).to(dist.device)
        for mod, ids in label_ids_by_mod.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-2)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_cfg.get("epochs", 60)))
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

    ce_w = float(loss_cfg.get("ce_weight", 0.5))
    proto_w = float(loss_cfg.get("prototype_weight", 0.3))
    contrast_w = float(loss_cfg.get("contrastive_weight", 0.2))
    ot_w = float(loss_cfg.get("ot_weight", 0.05))
    mmd_w = float(loss_cfg.get("mmd_weight", 0.05))
    dyn_w = float(loss_cfg.get("dynamics_weight", 0.01))
    fusion_ce_w = float(loss_cfg.get("fusion_ce_weight", 1.0))
    rel_w = float(loss_cfg.get("reliability_weight", 0.01))
    if simple_loss_mode:
        contrast_w = 0.0
        ot_w = 0.0
        mmd_w = 0.0
        dyn_w = 0.0
        rel_w = 0.0
        ce_w = float(loss_cfg.get("simple_ce_weight", 1.0))
        proto_w = float(loss_cfg.get("simple_proto_weight", 0.2))

    out_dir = Path(train_cfg.get("out", "runs/triview_alignment"))
    writer = make_writer(out_dir); log_config(writer, cfg)
    best = -1.0
    prompt_text = (
        "Instruction: assess seafarer affect state from available modalities. "
        "Focus on stress, fatigue, risk, valence, arousal and safety evidence."
    )
    train_prompt_ids = MultimodalEmotionModel.prompt_ids_from_texts(prompt_text, device=dist.device)

    for epoch in range(1, int(train_cfg.get("epochs", 60)) + 1):
        for sampler in [sampler_a, sampler_p, sampler_v]:
            if sampler is not None: sampler.set_epoch(epoch)
        face_cycle = cycle_loader(face_loader) if face_loader is not None else None
        model.train(); freeze_batchnorm_running_stats(model); start = time.time(); total_samples = 0
        debug_timing = bool(train_cfg.get("debug_timing", False))
        perf_times = {"batch_total": 0.0, "ot_mmd": 0.0, "forward_raw": 0.0}
        losses = {"total":0.0,"ce":0.0,"proto":0.0,"supcon":0.0,"ot":0.0,"mmd":0.0,"dyn":0.0,"fusion":0.0,"rel":0.0}
        iterator = zip(audio_loader, physio_loader)
        steps = min(len(audio_loader), len(physio_loader))
        if is_main_process(): iterator = tqdm(iterator, total=steps, desc=f"triview epoch {epoch}")
        for a_batch, p_batch in iterator:
            if debug_timing:
                if dist.device.type == "cuda":
                    torch.cuda.synchronize()
                batch_t0 = time.time()
            batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
                "audio": (a_batch["audio"].to(dist.device, non_blocking=True), a_batch["label"].to(dist.device, non_blocking=True)),
                "physio": (p_batch["physio"].to(dist.device, non_blocking=True), p_batch["label"].to(dist.device, non_blocking=True)),
            }
            if face_cycle is not None:
                v_batch = next(face_cycle)
                batch["face"] = (v_batch["face"].to(dist.device, non_blocking=True), v_batch["label"].to(dist.device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and dist.device.type == "cuda")):
                prompt_ids = train_prompt_ids

                ce = torch.zeros((), device=dist.device)
                proto = torch.zeros((), device=dist.device)
                dyn = torch.zeros((), device=dist.device)
                rel = torch.zeros((), device=dist.device)
                supcon = torch.zeros((), device=dist.device)
                ot = torch.zeros((), device=dist.device)
                mmd = torch.zeros((), device=dist.device)
                fusion = torch.zeros((), device=dist.device)

                embeddings: Dict[str, torch.Tensor] = {}
                zlist: List[torch.Tensor] = []
                labels_for_contrast: List[torch.Tensor] = []
                present_mods = [m for m in ["audio", "physio", "face"] if m in batch]

               
                raw_inputs = {mod: batch[mod][0] for mod in present_mods}
                if debug_timing:
                    if dist.device.type == "cuda":
                        torch.cuda.synchronize()
                    fwd_t0 = time.time()
                out_raw = model(**raw_inputs, prompt_ids=prompt_ids)
                if debug_timing:
                    if dist.device.type == "cuda":
                        torch.cuda.synchronize()
                    perf_times["forward_raw"] += time.time() - fwd_t0
                raw_embeddings = {}
                raw_reliabilities: Dict[str, torch.Tensor] = {}
                for mod in present_mods:
                    _, y_mod = batch[mod]
                    ce = ce + masked_cross_entropy(
                        out_raw[f"{mod}_logits"],
                        y_mod,
                        active_by_mod.get(mod),
                        weight=class_weights_by_mod.get(mod),
                        label_smoothing=float(loss_cfg.get("label_smoothing", 0.03)),
                    )
                    proto = proto + masked_cross_entropy(
                        out_raw[f"{mod}_proto_logits"],
                        y_mod,
                        active_by_mod.get(mod),
                        weight=class_weights_by_mod.get(mod),
                        label_smoothing=float(loss_cfg.get("label_smoothing", 0.03)),
                    )
                    dyn = dyn + temporal_smoothness_loss(out_raw.get(f"{mod}_affect_seq"))
                    rel = rel + reliability_consistency_loss({mod: out_raw.get(f"{mod}_reliability")})
                    embeddings[mod] = out_raw[f"{mod}_embedding"]
                    raw_embeddings[mod] = out_raw[f"{mod}_embedding"]
                    raw_reliabilities[mod] = out_raw[f"{mod}_reliability"]
                    zlist.append(out_raw[f"{mod}_embedding"])
                    labels_for_contrast.append(y_mod)

                if use_distribution_losses and len(zlist) >= 2:
                    if debug_timing:
                        if dist.device.type == "cuda":
                            torch.cuda.synchronize()
                        ot_t0 = time.time()
                    supcon = supervised_contrastive_loss(
                        torch.cat(zlist, 0),
                        torch.cat(labels_for_contrast, 0),
                        temperature=float(loss_cfg.get("temperature", 0.07)),
                    )
                    ot = pairwise_multiview_ot_loss(
                        embeddings,
                        epsilon=float(loss_cfg.get("ot_epsilon", 0.05)),
                        n_iters=int(loss_cfg.get("ot_iters", 40)),
                    )
                    mmd = pairwise_multiview_mmd_loss(embeddings)
                    if debug_timing:
                        if dist.device.type == "cuda":
                            torch.cuda.synchronize()
                        perf_times["ot_mmd"] += time.time() - ot_t0

                
                fusion_terms: List[torch.Tensor] = []
                fusion_real_count = 0
                if not simple_loss_mode:
                    for pair in itertools.combinations(present_mods, 2):
                        _, py, idxs = make_pseudo_multiview_batch({k: batch[k] for k in pair})
                        has_real_pseudo = py is not None
                        if has_real_pseudo:
                            fusion_real_count += 1
                        if py is None:
                            _, py, idxs = make_dummy_multiview_batch(batch, pair)
                        selected_embeddings = {k: raw_embeddings[k].index_select(0, idxs[k]) for k in pair}
                        selected_reliabilities = {k: raw_reliabilities[k].index_select(0, idxs[k]) for k in pair}
                        out_pair = fused_output_from_embeddings(model, selected_embeddings, selected_reliabilities, prompt_ids)
                        if has_real_pseudo:
                            pair_active = sorted(set(active_by_mod.get(pair[0], [])) & set(active_by_mod.get(pair[1], [])))
                        else:
                            pair_active = []
                        if not pair_active:
                            pair_active = sorted(set(int(v) for v in py.detach().cpu().tolist()))
                        fusion_loss = masked_cross_entropy(out_pair["fused_logits"], py, pair_active)
                        if not has_real_pseudo:
                            fusion_loss = fusion_loss * 0.0
                        fusion_terms.append(fusion_loss)

                    if len(present_mods) >= 3:
                        _, py, idxs = make_pseudo_multiview_batch(batch)
                        has_real_pseudo = py is not None
                        if has_real_pseudo:
                            fusion_real_count += 1
                        if py is None:
                            _, py, idxs = make_dummy_multiview_batch(batch, tuple(present_mods))
                        selected_embeddings = {k: raw_embeddings[k].index_select(0, idxs[k]) for k in present_mods}
                        selected_reliabilities = {k: raw_reliabilities[k].index_select(0, idxs[k]) for k in present_mods}
                        out_tri = fused_output_from_embeddings(model, selected_embeddings, selected_reliabilities, prompt_ids)
                        if has_real_pseudo:
                            tri_active = sorted(set.intersection(*(set(active_by_mod.get(m, [])) for m in present_mods)))
                        else:
                            tri_active = []
                        if not tri_active:
                            tri_active = sorted(set(int(v) for v in py.detach().cpu().tolist()))
                        fusion_loss = masked_cross_entropy(out_tri["fused_logits"], py, tri_active)
                        if not has_real_pseudo:
                            fusion_loss = fusion_loss * 0.0
                        fusion_terms.append(fusion_loss)
                else:
                    fusion_terms = []

                if fusion_terms:
                    fusion = torch.stack(fusion_terms).sum() / max(1, fusion_real_count)

                total = (
                    ce_w * ce +
                    proto_w * proto +
                    contrast_w * supcon +
                    ot_w * ot +
                    mmd_w * mmd +
                    dyn_w * dyn +
                    fusion_ce_w * fusion +
                    rel_w * rel
                )
            scaler.scale(total).backward(); scaler.step(optimizer); scaler.update()
            bs = sum(int(batch[m][1].numel()) for m in batch); total_samples += bs
            if debug_timing:
                if dist.device.type == "cuda":
                    torch.cuda.synchronize()
                perf_times["batch_total"] += time.time() - batch_t0
            for k, v in [("total",total),("ce",ce),("proto",proto),("supcon",supcon),("ot",ot),("mmd",mmd),("dyn",dyn),("fusion",fusion),("rel",rel)]:
                losses[k] += float(v.detach().cpu() if torch.is_tensor(v) else v) * bs
        scheduler.step()
        if epoch == 1 or epoch % eval_every == 0:
            val = evaluate(model, val_loaders, dist.device, amp_dtype, active_by_mod, train_prompt_ids)
        else:
            val = {}
        if debug_timing and is_main_process():
            nsteps = max(1, steps)
            print({k: perf_times[k] / nsteps for k in perf_times})
        row = {"epoch": epoch, **{f"loss_{k}": v/max(1,total_samples) for k,v in losses.items()}, **{f"val_{k}": v for k,v in val.items()}, "samples_per_second_global": (total_samples * dist.world_size)/max(1e-6,time.time()-start)}
        append_jsonl(out_dir/"metrics.jsonl", row)
        if writer is not None:
            for k,v in row.items():
                if k != "epoch": writer.add_scalar(k.replace("loss_","loss/").replace("val_","val/"), float(v), epoch)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        save_checkpoint(out_dir/"last.pt", model, optimizer, epoch, row, cfg)
        metric = row.get("val_fusion_macro_f1", row.get("val_audio_macro_f1", 0.0))
        if metric > best:
            best = metric; save_checkpoint(out_dir/"best.pt", model, optimizer, epoch, row, cfg)
        if is_main_process(): print(row)

    if writer is not None: writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
