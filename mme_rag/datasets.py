from __future__ import annotations

import csv
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import AUDIO_LABEL_ALIASES, FACE_LABEL_ALIASES, DEFAULT_PHYSIO_CHANNELS, IEMOCAP_IGNORE_LABELS, FACE_IGNORE_LABELS, LABEL_TO_ID, WESAD_LABEL_MAP


def normalize_label(label: str) -> str:
    key = str(label).strip().lower().replace(" ", "_")
    if key not in AUDIO_LABEL_ALIASES:
        raise ValueError(f"Unknown audio label {label!r}. Add it to AUDIO_LABEL_ALIASES or map it in the manifest.")
    mapped = AUDIO_LABEL_ALIASES[key]
    if mapped not in LABEL_TO_ID:
        raise ValueError(f"Mapped label {mapped!r} is not in unified LABELS.")
    return mapped


def normalize_face_label(label: str) -> str:
    key = str(label).strip().lower().replace(" ", "_")
    if key not in FACE_LABEL_ALIASES:
        raise ValueError(f"Unknown face label {label!r}. Add it to FACE_LABEL_ALIASES or map it in the manifest.")
    mapped = FACE_LABEL_ALIASES[key]
    if mapped not in LABEL_TO_ID:
        raise ValueError(f"Mapped face label {mapped!r} is not in unified LABELS.")
    return mapped

# ---------------------------------------------------------------------------
# IEMOCAP manifest preparation
# ---------------------------------------------------------------------------

_IEMOCAP_EVAL_RE = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s+"
    r"(?P<utt>\S+)\s+(?P<label>\S+)\s+"
    r"\[(?P<vad>[^\]]+)\]"
)


def _session_sort_key(path: Path) -> int:
    m = re.search(r"session\s*([0-9]+)", path.name, flags=re.I)
    return int(m.group(1)) if m else 999


IEMOCAP_REDUCED_LABEL_MAP = {
    "neu": "neu",
    "hap": "amu",
    "exc": "str",
    "ang": "str",
    "sad": "str",
    "fru": "str",
}


def _find_iemocap_sessions(root: str | os.PathLike) -> List[Path]:
    """Find IEMOCAP Session1..Session5 directories, case-insensitively."""
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"IEMOCAP root does not exist: {root}")
    sessions = [p for p in root.iterdir() if p.is_dir() and re.match(r"session\s*[0-9]+$", p.name, flags=re.I)]
    if not sessions:
        sessions = [p for p in root.rglob("*") if p.is_dir() and re.match(r"session\s*[0-9]+$", p.name, flags=re.I)]
    sessions = sorted(set(sessions), key=_session_sort_key)
    if not sessions:
        raise FileNotFoundError(f"No Session*/session* directories found under IEMOCAP root: {root}")
    return sessions


def _find_child_dir(parent: Path, *names: str) -> Optional[Path]:
    """Find child directory with one of the given names, case-insensitively."""
    if not parent.exists():
        return None
    wanted = {n.lower() for n in names}
    for p in parent.iterdir():
        if p.is_dir() and p.name.lower() in wanted:
            return p
    return None


def _dialog_id_from_utterance(utt_id: str) -> str:
    """Ses01F_impro01_F000 -> Ses01F_impro01."""
    m = re.match(r"(.+?)_[FM][0-9]+$", utt_id)
    return m.group(1) if m else "_".join(utt_id.split("_")[:-1])


def _parse_iemocap_transcriptions(session_dir: Path) -> Dict[str, str]:
    """Parse Session*/dialog/Transcriptions/*.txt into utterance_id -> text."""
    dialog_dir = _find_child_dir(session_dir, "dialog")
    if dialog_dir is None:
        return {}
    trans_dir = _find_child_dir(dialog_dir, "Transcriptions", "transcriptions")
    if trans_dir is None:
        return {}
    out: Dict[str, str] = {}
    # Example line: Ses01F_impro01_F000 [6.2901-8.2357]: text...
    pat = re.compile(r"^(?P<utt>\S+)\s+\[[^\]]+\]\s*:\s*(?P<text>.*)$")
    for txt in sorted(trans_dir.glob("*.txt")):
        try:
            lines = txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        except UnicodeDecodeError:
            lines = txt.read_text(encoding="latin1", errors="ignore").splitlines()
        for line in lines:
            m = pat.match(line.strip())
            if m:
                out[m.group("utt")] = m.group("text").strip()
    return out


def _build_iemocap_wav_index(session_dir: Path) -> Dict[str, Path]:
    """Index sentence-level wavs by utterance id.

    Standard path is:
      SessionX/sentences/wav/<dialog_id>/<utterance_id>.wav
    but this index also supports mirrored/case-varied directory names.
    """
    index: Dict[str, Path] = {}
    candidate_roots: List[Path] = []
    sent_dir = _find_child_dir(session_dir, "sentences", "Sentences")
    if sent_dir is not None:
        wav_dir = _find_child_dir(sent_dir, "wav", "WAV", "audio", "Audio")
        if wav_dir is not None:
            candidate_roots.append(wav_dir)
    # fallback: recursive search below the whole session.
    candidate_roots.append(session_dir)
    for root in candidate_roots:
        for wav in root.rglob("*.wav"):
            index.setdefault(wav.stem, wav.resolve())
    return index


def _parse_iemocap_eval_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding="latin1", errors="ignore").splitlines()
    for line in lines:
        line = line.strip()
        if not line.startswith("["):
            continue
        m = _IEMOCAP_EVAL_RE.match(line)
        if not m:
            continue
        vad_raw = [x.strip() for x in m.group("vad").split(",")]
        vad = []
        for x in vad_raw[:3]:
            try:
                vad.append(float(x))
            except ValueError:
                vad.append(float("nan"))
        while len(vad) < 3:
            vad.append(float("nan"))
        rows.append({
            "start": float(m.group("start")),
            "end": float(m.group("end")),
            "utterance_id": m.group("utt"),
            "iemocap_label": m.group("label").lower(),
            "valence": vad[0],
            "activation": vad[1],
            "dominance": vad[2],
            "eval_file": str(path),
        })
    return rows


def _iemocap_split(session_name: str, strategy: str = "session5_test") -> str:
    """Default session-independent split.

    session5_test: Session1-3 train, Session4 val, Session5 test.
    all_train: every utterance is train, useful for quick debugging.
    """
    m = re.search(r"([0-9]+)", session_name)
    sid = int(m.group(1)) if m else 0
    strategy = str(strategy).lower()
    if strategy in {"all_train", "train"}:
        return "train"
    if strategy in {"session5_test", "speaker_independent", "default"}:
        if sid == 5:
            return "test"
        if sid == 4:
            return "val"
        return "train"
    raise ValueError(f"Unknown IEMOCAP split strategy: {strategy}")


def prepare_iemocap_manifest(
    iemocap_root: str | os.PathLike,
    out_csv: str | os.PathLike,
    split_strategy: str = "session5_test",
    include_original_labels: Optional[Sequence[str]] = None,
    skip_missing_wav: bool = True,
    force: bool = False,
) -> Path:
 
    out_csv = Path(out_csv)
    if out_csv.exists() and not force:
        return out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    include_set = {x.lower() for x in include_original_labels} if include_original_labels else None
    rows: List[Dict[str, Any]] = []
    missing_wav = 0
    skipped_label = 0

    for session_dir in _find_iemocap_sessions(iemocap_root):
        session_name = session_dir.name
        dialog_dir = _find_child_dir(session_dir, "dialog")
        if dialog_dir is None:
            raise FileNotFoundError(f"Missing dialog directory in {session_dir}")
        emo_dir = _find_child_dir(dialog_dir, "EmoEvaluation", "emoevaluation", "emotion", "EmoEval")
        if emo_dir is None:
            raise FileNotFoundError(f"Missing dialog/EmoEvaluation directory in {session_dir}")
        transcripts = _parse_iemocap_transcriptions(session_dir)
        wav_index = _build_iemocap_wav_index(session_dir)
        split = _iemocap_split(session_name, split_strategy)

        for eval_file in sorted(emo_dir.glob("*.txt")):
            for r in _parse_iemocap_eval_file(eval_file):
                raw = str(r["iemocap_label"]).lower()
                if raw in IEMOCAP_IGNORE_LABELS:
                    skipped_label += 1
                    continue
                if include_set is not None and raw not in include_set:
                    skipped_label += 1
                    continue
                if raw in IEMOCAP_REDUCED_LABEL_MAP:
                    label = IEMOCAP_REDUCED_LABEL_MAP[raw]
                else:
                    try:
                        label = normalize_label(raw)
                    except ValueError:
                        skipped_label += 1
                        continue
                utt = r["utterance_id"]
                wav_path = wav_index.get(utt)
                if wav_path is None:
                    missing_wav += 1
                    if skip_missing_wav:
                        continue
                dialog_id = _dialog_id_from_utterance(utt)
                speaker = ""
                m_spk = re.search(r"_([FM])[0-9]+$", utt)
                if m_spk:
                    speaker = f"{session_name}_{m_spk.group(1)}"
                rows.append({
                    "path": str(wav_path) if wav_path is not None else "",
                    "label": label,
                    "split": split,
                    "speaker_id": speaker,
                    "dataset": "IEMOCAP",
                    "session": session_name,
                    "dialog_id": dialog_id,
                    "utterance_id": utt,
                    "iemocap_label": raw,
                    "start": r["start"],
                    "end": r["end"],
                    "duration": max(0.0, float(r["end"]) - float(r["start"])),
                    "valence": r["valence"],
                    "activation": r["activation"],
                    "dominance": r["dominance"],
                    "transcription": transcripts.get(utt, ""),
                    "source_eval_file": r["eval_file"],
                })

    if not rows:
        raise RuntimeError(
            f"No usable IEMOCAP utterances found under {iemocap_root}. "
            f"Check directory layout and include_original_labels. missing_wav={missing_wav}, skipped_label={skipped_label}"
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    summary = df.groupby(["split", "label"]).size().reset_index(name="count")
    summary.to_csv(out_csv.with_suffix(".summary.csv"), index=False)
    return out_csv


def ensure_audio_manifest_from_config(data_cfg: Dict[str, Any]) -> str:

    dataset_format = str(data_cfg.get("dataset_format", "manifest")).lower()
    manifest = str(data_cfg.get("audio_manifest", "data/audio_manifest.csv"))
    if dataset_format in {"manifest", "csv"}:
        return manifest
    if dataset_format == "iemocap":
        root = data_cfg.get("iemocap_root")
        if not root:
            raise ValueError("data.iemocap_root is required when data.dataset_format=iemocap")
        prepare = bool(data_cfg.get("prepare_manifest", True))
        if prepare or not Path(manifest).exists():
            prepare_iemocap_manifest(
                iemocap_root=root,
                out_csv=manifest,
                split_strategy=str(data_cfg.get("iemocap_split_strategy", "session5_test")),
                include_original_labels=data_cfg.get("iemocap_include_original_labels"),
                skip_missing_wav=bool(data_cfg.get("skip_missing_wav", True)),
                force=bool(data_cfg.get("force_rebuild_manifest", False)),
            )
        return manifest
    raise ValueError(f"Unsupported data.dataset_format: {dataset_format}")


def _read_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except TypeError:
            f.seek(0)
            return pickle.load(f)


def _find_key_case_insensitive(d: Dict[str, Any], candidates: Sequence[str]) -> str:
    for c in candidates:
        if c in d:
            return c
    low = {str(k).lower(): k for k in d.keys()}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"Missing channel {candidates}; available keys: {sorted(map(str, d.keys()))}")


def _extract_wesad_chest(data: Dict[str, Any], channels: Sequence[str]) -> np.ndarray:
    signal = data.get("signal") or data.get(b"signal")
    if signal is None:
        raise KeyError(f"WESAD pickle missing 'signal'. Available keys: {list(data.keys())}")
    chest = signal.get("chest") if isinstance(signal, dict) else None
    if chest is None and isinstance(signal, dict):
        chest = signal.get(b"chest")
    if chest is None:
        raise KeyError(f"WESAD pickle missing signal['chest']. Available signal keys: {list(signal.keys())}")

    arrs: List[np.ndarray] = []
    for ch in channels:
        if ch.startswith("ACC_"):
            acc_key = _find_key_case_insensitive(chest, ["ACC", "acc"])
            acc = np.asarray(chest[acc_key], dtype=np.float32)
            if acc.ndim == 1:
                acc = acc[:, None]
            axis = {"ACC_x": 0, "ACC_y": 1, "ACC_z": 2}[ch]
            arrs.append(acc[:, axis])
        else:
            key = _find_key_case_insensitive(chest, [ch, ch.upper(), ch.lower(), "Resp" if ch.lower() == "resp" else ch])
            x = np.asarray(chest[key], dtype=np.float32)
            if x.ndim > 1:
                x = x.reshape(x.shape[0], -1)[:, 0]
            arrs.append(x)

    min_len = min(len(x) for x in arrs)
    return np.stack([x[:min_len] for x in arrs], axis=0)  # [C, T]


def _extract_wesad_labels(data: Dict[str, Any], target_len: int) -> np.ndarray:
    labels = data.get("label")
    if labels is None:
        labels = data.get(b"label")
    if labels is None:
        raise KeyError(f"WESAD pickle missing label. Available keys: {list(data.keys())}")
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if len(labels) < target_len:
        target_len = len(labels)
    return labels[:target_len]


def _resample_window_np(x: np.ndarray, target_steps: int) -> np.ndarray:
    """Linear interpolation from [C, T] to [C, target_steps]."""
    c, t = x.shape
    if t == target_steps:
        return x.astype(np.float32)
    old_idx = np.linspace(0.0, 1.0, num=t, dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_steps, dtype=np.float32)
    out = np.empty((c, target_steps), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(new_idx, old_idx, x[i]).astype(np.float32)
    return out


def discover_wesad_pickles(root: str | os.PathLike) -> List[Path]:
    root = Path(root)
    paths = sorted(root.glob("S*/S*.pkl"))
    if not paths:
        paths = sorted(root.rglob("S*.pkl"))
    return paths


def build_wesad_tensor_cache(
    wesad_root: str | os.PathLike,
    cache_dir: str | os.PathLike,
    channels: Sequence[str] = DEFAULT_PHYSIO_CHANNELS,
    window_seconds: float = 30.0,
    stride_seconds: float = 15.0,
    source_hz: int = 700,
    target_hz: int = 32,
    min_label_fraction: float = 0.8,
    train_subjects: Optional[Sequence[str]] = None,
    val_subjects: Optional[Sequence[str]] = None,
    test_subjects: Optional[Sequence[str]] = None,
    rebuild: bool = False,
) -> None:
    """Create train/val/test .pt files with precomputed WESAD windows.

    Modify this function when replacing WESAD with another physiological dataset.
    Expected cached sample tensors are [N, C, T] and labels [N].
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    done_file = cache_dir / "_DONE"
    if done_file.exists() and not rebuild:
        return

    pkl_paths = discover_wesad_pickles(wesad_root)
    if not pkl_paths:
        raise FileNotFoundError(f"No WESAD S*/S*.pkl files found under: {wesad_root}")

    subject_ids = [p.parent.name for p in pkl_paths]
    unique_subjects = sorted(set(subject_ids), key=lambda s: int(s[1:]) if s[1:].isdigit() else s)
    if train_subjects is None or val_subjects is None or test_subjects is None:
        # Subject-independent default split.
        random.Random(42).shuffle(unique_subjects)
        n = len(unique_subjects)
        n_val = max(1, int(round(n * 0.15)))
        n_test = max(1, int(round(n * 0.15)))
        val_subjects = unique_subjects[:n_val]
        test_subjects = unique_subjects[n_val : n_val + n_test]
        train_subjects = unique_subjects[n_val + n_test :]

    split_of = {}
    for s in train_subjects:
        split_of[str(s)] = "train"
    for s in val_subjects:
        split_of[str(s)] = "val"
    for s in test_subjects:
        split_of[str(s)] = "test"

    windows: Dict[str, List[torch.Tensor]] = defaultdict(list)
    labels_out: Dict[str, List[int]] = defaultdict(list)
    subjects_out: Dict[str, List[str]] = defaultdict(list)

    src_window = int(round(window_seconds * source_hz))
    src_stride = int(round(stride_seconds * source_hz))
    target_steps = int(round(window_seconds * target_hz))

    for pkl_path in tqdm(pkl_paths, desc="Building WESAD cache"):
        subject = pkl_path.parent.name
        split = split_of.get(subject)
        if split is None:
            continue
        data = _read_pickle(pkl_path)
        mat = _extract_wesad_chest(data, channels)
        labs = _extract_wesad_labels(data, mat.shape[1])
        tmax = min(mat.shape[1], labs.shape[0])
        mat = mat[:, :tmax]
        labs = labs[:tmax]

        for start in range(0, tmax - src_window + 1, src_stride):
            y_win = labs[start : start + src_window]
            counts = Counter(int(v) for v in y_win.tolist())
            raw_label, count = counts.most_common(1)[0]
            if raw_label not in WESAD_LABEL_MAP:
                continue
            if count / max(1, len(y_win)) < min_label_fraction:
                continue
            label_name = WESAD_LABEL_MAP[raw_label]
            label_id = LABEL_TO_ID[label_name]
            x_win = mat[:, start : start + src_window]
            x_resampled = _resample_window_np(x_win, target_steps)
            windows[split].append(torch.from_numpy(x_resampled))
            labels_out[split].append(label_id)
            subjects_out[split].append(subject)

    # Normalize using train split statistics only.
    if not windows["train"]:
        raise RuntimeError("No train WESAD windows generated. Check labels/window settings.")
    train_x = torch.stack(windows["train"])
    mean = train_x.mean(dim=(0, 2), keepdim=True)
    std = train_x.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)

    meta = {
        "channels": list(channels),
        "window_seconds": window_seconds,
        "stride_seconds": stride_seconds,
        "source_hz": source_hz,
        "target_hz": target_hz,
        "target_steps": target_steps,
        "train_subjects": list(train_subjects),
        "val_subjects": list(val_subjects),
        "test_subjects": list(test_subjects),
        "mean": mean.squeeze(0).tolist(),
        "std": std.squeeze(0).tolist(),
    }

    for split in ["train", "val", "test"]:
        if windows[split]:
            x = torch.stack(windows[split]).float()
            x = (x - mean) / std
            y = torch.tensor(labels_out[split], dtype=torch.long)
            torch.save({"x": x, "y": y, "subjects": subjects_out[split], "meta": meta}, cache_dir / f"{split}.pt")
        else:
            torch.save({"x": torch.empty(0, len(channels), target_steps), "y": torch.empty(0, dtype=torch.long), "subjects": [], "meta": meta}, cache_dir / f"{split}.pt")

    torch.save(meta, cache_dir / "meta.pt")
    done_file.write_text("ok", encoding="utf-8")


class WESADWindows(Dataset):
    def __init__(self, cache_dir: str | os.PathLike, split: str):
        path = Path(cache_dir) / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"WESAD cache missing: {path}. Run train_wesad_physio.py with rebuild_cache=true.")
        obj = torch.load(path, map_location="cpu")
        self.x = obj["x"].float()
        self.y = obj["y"].long()
        self.subjects = obj.get("subjects", [""] * len(self.y))
        self.meta = obj.get("meta", {})

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"physio": self.x[idx], "label": self.y[idx], "subject": self.subjects[idx]}


class AudioEmotionDataset(Dataset):
    """CSV manifest based audio dataset.

    Required columns: path,label
    Optional columns: speaker_id,dataset,language,duration

    To replace the audio dataset, create a new CSV with the same columns, or
    modify this class if your labels/paths are stored differently.
    """

    def __init__(
        self,
        manifest_csv: str | os.PathLike,
        sample_rate: int = 16000,
        clip_seconds: float = 6.0,
        split: str | None = None,
        split_column: str = "split",
        random_crop: bool = True,
        include_labels: Optional[Sequence[str]] = None,
        exclude_labels: Optional[Sequence[str]] = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Audio manifest not found: {self.manifest_csv}")
        df = pd.read_csv(self.manifest_csv)
        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError("Audio manifest must contain columns: path,label")
        if split and split_column in df.columns:
            df = df[df[split_column].astype(str).str.lower() == split.lower()].copy()

        include_set = {normalize_label(x) for x in include_labels} if include_labels else None
        exclude_set = {normalize_label(x) for x in exclude_labels} if exclude_labels else set()
        rows = []
        skipped = 0
        for r in df.to_dict("records"):
            label_name = normalize_label(r["label"])
            if include_set is not None and label_name not in include_set:
                skipped += 1
                continue
            if label_name in exclude_set:
                skipped += 1
                continue
            rr = dict(r)
            rr["label"] = label_name
            rows.append(rr)
        self.rows = rows
        if not self.rows:
            raise ValueError(
                f"No audio rows for split={split!r} in {manifest_csv}; "
                f"include_labels={include_labels}, exclude_labels={exclude_labels}, skipped={skipped}"
            )
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(round(float(clip_seconds) * self.sample_rate))
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_audio(self, path: str) -> torch.Tensor:
        try:
            import torchaudio  # type: ignore
        except Exception:
            torchaudio = None  # type: ignore

        p = Path(path).expanduser()
        if not p.exists():
            # Resolve relative paths against the CSV directory.
            p = (self.manifest_csv.parent / path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if torchaudio is not None:
            wav, sr = torchaudio.load(str(p))
            wav = wav.mean(dim=0)  # mono [T]
        else:
            import soundfile as sf
            arr, sr = sf.read(str(p), always_2d=False)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            wav = torch.from_numpy(arr)
        if sr != self.sample_rate:
            if torchaudio is not None:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            else:
                try:
                    from scipy.signal import resample_poly
                    import math
                    arr = wav.detach().cpu().numpy().astype(np.float32)
                    g = math.gcd(int(sr), int(self.sample_rate))
                    arr = resample_poly(arr, self.sample_rate // g, int(sr) // g).astype(np.float32)
                    wav = torch.from_numpy(arr)
                except Exception as e:
                    raise RuntimeError("Audio resampling requires torchaudio or scipy when sample rates differ") from e
        if wav.numel() >= self.clip_samples:
            if self.random_crop:
                start = random.randint(0, wav.numel() - self.clip_samples)
            else:
                start = (wav.numel() - self.clip_samples) // 2
            wav = wav[start : start + self.clip_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.clip_samples - wav.numel()))
        wav = wav.float()
        wav = wav / wav.abs().max().clamp_min(1e-5)
        return wav

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        label_name = normalize_label(row["label"])
        return {
            "audio": self._load_audio(str(row["path"])),
            "label": torch.tensor(LABEL_TO_ID[label_name], dtype=torch.long),
            "path": str(row["path"]),
            "speaker_id": str(row.get("speaker_id", "")),
        }


def split_audio_manifest(
    manifest_csv: str | os.PathLike,
    out_csv: str | os.PathLike,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    speaker_column: str = "speaker_id",
) -> None:
    """Create a manifest with a split column. Speaker-independent if speaker_id exists."""
    df = pd.read_csv(manifest_csv)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("Manifest must contain path,label")
    rng = random.Random(seed)
    if speaker_column in df.columns:
        keys = sorted(df[speaker_column].astype(str).unique())
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        val_keys = set(keys[:n_val])
        test_keys = set(keys[n_val : n_val + n_test])
        def split_row(x: Any) -> str:
            s = str(x)
            if s in val_keys:
                return "val"
            if s in test_keys:
                return "test"
            return "train"
        df["split"] = df[speaker_column].map(split_row)
    else:
        idx = list(range(len(df)))
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        split = ["train"] * n
        for i in idx[:n_val]:
            split[i] = "val"
        for i in idx[n_val : n_val + n_test]:
            split[i] = "test"
        df["split"] = split
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

# ---------------------------------------------------------------------------
# Facial-expression manifest dataset
# ---------------------------------------------------------------------------

def normalize_face_label(label: str) -> str:
    from .constants import FACE_LABEL_ALIASES
    key = str(label).strip().lower().replace(" ", "_")
    if key not in FACE_LABEL_ALIASES:
        raise ValueError(f"Unknown face label {label!r}. Add it to FACE_LABEL_ALIASES in constants.py.")
    mapped = FACE_LABEL_ALIASES[key]
    if mapped not in LABEL_TO_ID:
        raise ValueError(f"Mapped face label {mapped!r} is not in unified LABELS.")
    return mapped


class FaceEmotionDataset(Dataset):
    """Generic facial-expression dataset from a CSV manifest.

    Required CSV columns:
      path,label
    Recommended columns:
      path,label,split,subject_id,dataset

    Values are returned as image tensors [3,H,W] and unified label ids. This is
    the replacement point for AffectNet/RAF-DB/FERPlus or your own face data.
    """

    def __init__(
        self,
        manifest_csv: str | os.PathLike,
        image_size: int = 112,
        split: Optional[str] = None,
        random_flip: bool = False,
        low_quality: bool = False,
    ):
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Face manifest not found: {self.manifest_csv}")
        df = pd.read_csv(self.manifest_csv)
        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError("Face manifest must contain path,label columns")
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str).str.lower() == str(split).lower()]
        rows = []
        for _, r in df.iterrows():
            path = Path(str(r["path"])).expanduser()
            if not path.is_absolute():
                path = (self.manifest_csv.parent / path).resolve()
            if not path.exists():
                continue
            try:
                label = normalize_face_label(str(r["label"]))
            except ValueError:
                continue
            rows.append({"path": str(path), "label": label})
        if not rows:
            raise RuntimeError(f"No valid face samples found in {manifest_csv} split={split}")
        self.rows = rows
        self.image_size = int(image_size)
        self.random_flip = bool(random_flip)
        self.low_quality = bool(low_quality)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            from PIL import Image, ImageFilter
        except Exception as exc:
            raise ImportError("FaceEmotionDataset requires pillow. Install with: pip install pillow") from exc
        img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
        if self.random_flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.low_quality:
            # Controlled low-quality simulation for face view.
            if random.random() < 0.35:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.8)))
            if random.random() < 0.35:
                # Simulate low resolution.
                small = max(16, int(self.image_size * random.uniform(0.35, 0.75)))
                img = img.resize((small, small)).resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "face": self._load_image(r["path"]),
            "label": torch.tensor(LABEL_TO_ID[r["label"]], dtype=torch.long),
            "path": r["path"],
        }


def ensure_face_manifest_from_config(data_cfg: Dict[str, Any]) -> str:
    manifest = str(data_cfg.get("face_manifest", "data/face_manifest.csv"))
    if not Path(manifest).exists():
        raise FileNotFoundError(
            f"Face manifest not found: {manifest}. Provide a CSV with path,label,split columns, "
            "or set data.enable_face: false for two-view training."
        )
    return manifest


class FaceExpressionDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | os.PathLike,
        image_size: int = 112,
        split: str | None = None,
        split_column: str = "split",
        random_horizontal_flip: bool = True,
        corruption: Optional[Dict[str, Any]] = None,
    ):
        self.manifest_csv = Path(manifest_csv)
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Face manifest not found: {self.manifest_csv}")
        df = pd.read_csv(self.manifest_csv)
        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError("Face manifest must contain columns: path,label")
        if split and split_column in df.columns:
            df = df[df[split_column].astype(str).str.lower() == split.lower()].copy()
        rows = []
        skipped = 0
        for r in df.to_dict("records"):
            lab = str(r.get("label", "")).lower().strip()
            if lab in FACE_IGNORE_LABELS:
                skipped += 1
                continue
            try:
                normalize_face_label(lab)
            except ValueError:
                skipped += 1
                continue
            rows.append(r)
        if not rows:
            raise ValueError(f"No usable face rows for split={split!r} in {manifest_csv}; skipped={skipped}")
        self.rows = rows
        self.image_size = int(image_size)
        self.random_horizontal_flip = bool(random_horizontal_flip)
        self.corruption = corruption or {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.exists():
            p = (self.manifest_csv.parent / path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Face image not found: {path}")
        return p

    def _load_image(self, path: str) -> torch.Tensor:
        from PIL import Image, ImageFilter, ImageEnhance, ImageOps
        p = self._resolve_path(path)
        img = Image.open(p).convert("RGB")

        # RandomResizedCrop-like augmentation when in training mode (random_horizontal_flip used as train flag)
        if self.random_horizontal_flip:
            scale = float(self.corruption.get("random_resized_crop_scale", 0.0)) or float(getattr(self, "random_resized_crop_scale", 0.0))
            if not scale:
                scale = float(self.manifest_csv.parent.joinpath("..").exists()) if False else 0.9
            # choose crop size between scale * image_size and image_size
            s = random.uniform(scale, 1.0)
            crop_size = max(4, int(self.image_size * s))
            w, h = img.size
            if w > crop_size and h > crop_size:
                left = random.randint(0, max(0, w - crop_size))
                top = random.randint(0, max(0, h - crop_size))
                img = img.crop((left, top, left + crop_size, top + crop_size)).resize((self.image_size, self.image_size), Image.BILINEAR)
            else:
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        else:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Horizontal flip
        if self.random_horizontal_flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur corruption (kept from previous config)
        blur_prob = float(self.corruption.get("blur_prob", 0.0))
        if blur_prob > 0 and random.random() < blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(self.corruption.get("blur_radius", 1.5))))

        # Color jitter using PIL enchancers
        cj = self.corruption.get("color_jitter") or {}
        if cj:
            if random.random() < 0.8:
                if float(cj.get("brightness", 0.0)) > 0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(max(0.0, 1.0 - float(cj.get("brightness", 0.0))), 1.0 + float(cj.get("brightness", 0.0))))
                if float(cj.get("contrast", 0.0)) > 0:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(random.uniform(max(0.0, 1.0 - float(cj.get("contrast", 0.0))), 1.0 + float(cj.get("contrast", 0.0))))
                if float(cj.get("saturation", 0.0)) > 0:
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(random.uniform(max(0.0, 1.0 - float(cj.get("saturation", 0.0))), 1.0 + float(cj.get("saturation", 0.0))))

        arr = np.asarray(img).astype(np.float32) / 255.0

        # gaussian noise
        noise_std = float(self.corruption.get("gaussian_noise_std", 0.0))
        if noise_std > 0 and random.random() < 0.5:
            arr = np.clip(arr + np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32), 0.0, 1.0)

        # Random erasing
        re_prob = float(self.corruption.get("random_erasing_prob", 0.0))
        if re_prob > 0 and random.random() < re_prob:
            h, w, c = arr.shape
            erasing_area = random.uniform(0.02, 0.2) * h * w
            erasing_aspect = random.uniform(0.3, 3.3)
            eh = int(round((erasing_area * erasing_aspect) ** 0.5))
            ew = int(round((erasing_area / erasing_aspect) ** 0.5))
            if eh < h and ew < w:
                x1 = random.randint(0, h - eh)
                y1 = random.randint(0, w - ew)
                arr[x1:x1+eh, y1:y1+ew, :] = np.random.uniform(0.0, 1.0, size=(eh, ew, c)).astype(np.float32)

        # normalize to roughly ImageNet scale without hard dependency on torchvision
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        label_name = normalize_face_label(row["label"])
        return {
            "face": self._load_image(str(row["path"])),
            "label": torch.tensor(LABEL_TO_ID[label_name], dtype=torch.long),
            "path": str(row["path"]),
            "subject_id": str(row.get("subject_id", "")),
        }


def split_face_manifest(
    manifest_csv: str | os.PathLike,
    out_csv: str | os.PathLike,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    subject_column: str = "subject_id",
) -> None:
    """Create a face manifest with a split column; subject-independent if possible."""
    df = pd.read_csv(manifest_csv)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError("Face manifest must contain path,label")
    rng = random.Random(seed)
    if subject_column in df.columns:
        keys = sorted(df[subject_column].astype(str).unique())
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        val_keys = set(keys[:n_val]); test_keys = set(keys[n_val:n_val+n_test])
        df["split"] = df[subject_column].astype(str).map(lambda s: "val" if s in val_keys else "test" if s in test_keys else "train")
    else:
        idx = list(range(len(df)))
        rng.shuffle(idx)
        split = ["train"] * len(df)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        n_test = max(1, int(round(len(idx) * test_ratio)))
        for i in idx[:n_val]: split[i] = "val"
        for i in idx[n_val:n_val+n_test]: split[i] = "test"
        df["split"] = split
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
