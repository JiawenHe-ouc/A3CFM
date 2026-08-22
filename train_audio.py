#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from mme_rag.config import add_common_config_args, load_config_with_overrides
from mme_rag.datasets import AudioEmotionDataset, ensure_audio_manifest_from_config, normalize_label
from mme_rag.constants import LABELS, LABEL_TO_ID
from mme_rag.losses import class_balanced_weight_from_labels
from mme_rag.train_utils import (
    append_jsonl,
    make_writer,
    log_config,
    save_checkpoint,
    seed_everything,
    configure_torch,
)


class WaveformAugment:

    def __init__(self, noise_prob=0.5, noise_level=0.005):
        self.noise_prob = noise_prob
        self.noise_level = noise_level

    def __call__(self, wav):
        if random.random() < self.noise_prob:
            noise = torch.randn_like(wav) * self.noise_level
            wav = wav + noise
        return wav.clamp(-1, 1)


class WavLMEmotionModel(nn.Module):

    def __init__(
        self,
        num_classes,
        model_name="microsoft/wavlm-base",
        dropout=0.3,
        freeze_feature=False,
    ):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained(model_name)

        hidden = self.wavlm.config.hidden_size

        if freeze_feature:
            self.wavlm.feature_extractor._freeze_parameters()

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, wav):

        outputs = self.wavlm(
            input_values=wav
        )

        hidden = outputs.last_hidden_state

        # temporal pooling
        emb = hidden.mean(dim=1)

        logits = self.classifier(emb)

        return {
            "logits": logits,
            "embedding": emb
        }


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            wav = batch["audio"].to(device)
            y = batch["label"].to(device)

            out = model(wav)
            pred = out["logits"].argmax(dim=-1)
            loss = F.cross_entropy(out["logits"], y)

            correct += (pred == y).sum().item()
            total += y.numel()
            total_loss += float(loss.detach().cpu()) * y.numel()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    acc = correct / max(total, 1)
    metrics = {
        "accuracy": acc,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
    }
    if y_true:
        from mme_rag.metrics import classification_metrics
        metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def main():

    parser = argparse.ArgumentParser()
    add_common_config_args(parser, "configs/audio_pretrain.yaml")
    args = parser.parse_args()

    cfg = load_config_with_overrides(args)

    configure_torch()

    seed_everything(
        int(cfg["train"].get("seed", 42))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]


    manifest = ensure_audio_manifest_from_config(data_cfg)


    train_ds = AudioEmotionDataset(
        manifest,
        data_cfg.get("sample_rate",16000),
        data_cfg.get("clip_seconds",4),
        split="train",
        random_crop=True,
        include_labels=data_cfg.get("audio_include_labels")
    )


    val_ds = AudioEmotionDataset(
        manifest,
        data_cfg.get("sample_rate",16000),
        data_cfg.get("clip_seconds",4),
        split="val",
        random_crop=False,
        include_labels=data_cfg.get("audio_include_labels")
    )


    labels = torch.tensor(
        [
            LABEL_TO_ID[
                normalize_label(x["label"])
            ]
            for x in train_ds.rows
        ]
    )


    active_classes = sorted(
        set(labels.tolist())
    )


    print(
        "classes:",
        [LABELS[x] for x in active_classes]
    )


    batch_size = int(train_cfg.get("batch_size",32))


    if train_cfg.get("use_class_balanced_sampler",True):

        counts=torch.bincount(
            labels,
            minlength=len(LABELS)
        ).float()

        weights=1/(counts+1e-6)

        sample_weights=weights[labels]

        sampler=WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True
        )

        train_loader=DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4
        )

    else:

        train_loader=DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )


    val_loader=DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )


    model=WavLMEmotionModel(
        num_classes=len(LABELS),
        model_name=model_cfg.get(
            "wavlm_name",
            "microsoft/wavlm-base"
        ),
        dropout=model_cfg.get(
            "dropout",
            0.3
        )
    ).to(device)


    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=float(
            train_cfg.get(
                "lr",
                1e-5
            )
        ),
        weight_decay=0.01
    )


    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg.get("epochs",50))
    )


    criterion=nn.CrossEntropyLoss(
        label_smoothing=float(
            cfg.get("loss",{}).get(
                "label_smoothing",
                0.1
            )
        )
    )


    aug=WaveformAugment()


    out_dir=Path(
        train_cfg.get(
            "out",
            "runs/wavlm_ser"
        )
    )

    writer=make_writer(out_dir)

    best = 0
    epochs = int(train_cfg.get("epochs", 50))

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0
        start = time.time()

        for batch in tqdm(
            train_loader,
            desc=f"epoch {epoch}"
        ):

            wav = batch["audio"].to(device)
            y = batch["label"].to(device)

            wav = aug(wav)
            out = model(wav)
            loss = criterion(out["logits"], y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            total_loss += float(loss.item()) * y.numel()
            total_samples += y.numel()

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        train_loss = total_loss / max(1, total_samples)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "samples_per_second_global": float(total_samples / max(1e-6, time.time() - start)),
        }

        print(row)

        append_jsonl(
            out_dir / "metrics.jsonl",
            row
        )

        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            epoch,
            row,
            cfg
        )

        if val_metrics["accuracy"] > best:
            best = val_metrics["accuracy"]

            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                epoch,
                row,
                cfg
            )


if __name__=="__main__":
    main()
