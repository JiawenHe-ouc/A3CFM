from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
try:
    import torchaudio  # type: ignore
except Exception:
    torchaudio = None  # type: ignore

from .constants import ID_TO_LABEL, LABELS
from .models import MultimodalEmotionModel


def load_multimodal_model(checkpoint: str, device: str | torch.device = "auto") -> tuple[MultimodalEmotionModel, Dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    state = ckpt.get("model", ckpt)
    first_conv_key = next((k for k in state if k.endswith("physio_encoder.features.0.net.0.weight") or "physio_encoder.features.0.net.0.weight" in k), None)
    physio_channels = int(state[first_conv_key].shape[1]) if first_conv_key else len(data_cfg.get("channels", [])) or 8

    inferred_num_classes = None
    for key in ("classifier.weight", "prototype_head.prototypes"):
        tensor = state.get(key)
        if tensor is not None and tensor.ndim >= 2:
            inferred_num_classes = int(tensor.shape[0])
            break
    if inferred_num_classes is None:
        inferred_num_classes = int(model_cfg.get("num_classes", len(LABELS)))
    if inferred_num_classes <= 0:
        inferred_num_classes = len(LABELS)

    model = MultimodalEmotionModel(
        num_classes=inferred_num_classes,
        physio_channels=physio_channels,
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        fusion_heads=int(model_cfg.get("fusion_heads", 4)),
    )
    model.load_state_dict(state, strict=False)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, cfg


def preprocess_gradio_audio(audio_input, sample_rate: int = 16000, clip_seconds: float = 6.0) -> tuple[np.ndarray, torch.Tensor]:
    if audio_input is None:
        return np.zeros(0, dtype=np.float32), torch.empty(0)
    if isinstance(audio_input, tuple):
        sr, arr = audio_input
        arr = np.asarray(arr)
        if arr.ndim > 1:
            arr = arr.mean(axis=-1)
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)
        wav = torch.from_numpy(arr)
    else:
        if torchaudio is not None:
            wav, sr = torchaudio.load(str(audio_input)); wav = wav.mean(dim=0); arr = wav.numpy().astype(np.float32)
        else:
            import soundfile as sf
            arr, sr = sf.read(str(audio_input), always_2d=False)
            arr = np.asarray(arr)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            arr = arr.astype(np.float32)
            wav = torch.from_numpy(arr)
    if sr != sample_rate:
        if torchaudio is not None:
            wav = torchaudio.functional.resample(wav.float(), int(sr), sample_rate); arr = wav.numpy().astype(np.float32)
        else:
            try:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(int(sr), int(sample_rate))
                arr = resample_poly(arr, sample_rate // g, int(sr) // g).astype(np.float32)
                wav = torch.from_numpy(arr)
            except Exception as e:
                raise RuntimeError("Audio resampling requires torchaudio or scipy when sample rates differ") from e
    clip_samples = int(sample_rate * clip_seconds)
    if wav.numel() >= clip_samples:
        start = (wav.numel() - clip_samples) // 2; wav_fixed = wav[start:start+clip_samples]
    else:
        wav_fixed = torch.nn.functional.pad(wav, (0, clip_samples - wav.numel()))
    wav_fixed = wav_fixed.float(); wav_fixed = wav_fixed / wav_fixed.abs().max().clamp_min(1e-5)
    return arr.astype(np.float32), wav_fixed.unsqueeze(0)


def load_physio_file(path: str | None, expected_channels: int, expected_steps: int | None = None) -> Optional[torch.Tensor]:
    if path is None: return None
    p = Path(path)
    if not p.exists(): return None
    if p.suffix.lower() == ".npy": arr = np.load(p)
    elif p.suffix.lower() in (".csv", ".txt"): arr = np.loadtxt(p, delimiter="," if p.suffix.lower() == ".csv" else None)
    else: raise ValueError("Physio file must be .npy, .csv, or .txt")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2: raise ValueError(f"Physio array must be 2D [C,T] or [T,C], got {arr.shape}")
    if arr.shape[0] != expected_channels and arr.shape[1] == expected_channels: arr = arr.T
    if arr.shape[0] != expected_channels: raise ValueError(f"Expected {expected_channels} physio channels, got {arr.shape}")
    if expected_steps is not None and arr.shape[1] != expected_steps:
        old = np.linspace(0,1,arr.shape[1],dtype=np.float32); new = np.linspace(0,1,expected_steps,dtype=np.float32)
        arr = np.stack([np.interp(new, old, arr[i]).astype(np.float32) for i in range(arr.shape[0])], axis=0)
    arr = (arr - arr.mean(axis=1, keepdims=True)) / (arr.std(axis=1, keepdims=True) + 1e-6)
    return torch.from_numpy(arr).unsqueeze(0)


def load_face_file(path: str | None, image_size: int = 112) -> Optional[torch.Tensor]:
    if path is None: return None
    p = Path(path)
    if not p.exists(): return None
    from PIL import Image
    img = Image.open(p).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32); std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2,0,1)).float().unsqueeze(0)


def affect_latent_summary(z: torch.Tensor | None) -> Dict[str, Any]:
    if z is None: return {}
    z0 = z[0].detach().float().cpu() if z.dim() == 2 else z.detach().float().view(-1).cpu()
    topk = torch.topk(z0.abs(), k=min(8, z0.numel()))
    return {"l2_norm": round(float(z0.norm().item()),4), "mean": round(float(z0.mean().item()),4), "std": round(float(z0.std().item()),4), "top_abs_dims": [int(i) for i in topk.indices.tolist()], "top_abs_values": [round(float(v),4) for v in topk.values.tolist()]}


def prediction_to_struct(logits: torch.Tensor, z_affect: Optional[torch.Tensor] = None, audio_reliability=None, physio_reliability=None, face_reliability=None, router_weights=None, modality_weights=None, modality_mask=None) -> Dict[str, Any]:
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax().item()); emotion = ID_TO_LABEL.get(idx, str(idx)); conf = float(probs[idx].item())
    if emotion == "stress": stress_score = int(round(6 + 4 * conf))
    elif emotion in ("angry", "fearful", "sad", "disgusted"): stress_score = int(round(5 + 3 * conf))
    else: stress_score = int(round(2 + 3 * (1 - conf)))
    stress_score = max(1, min(10, stress_score)); risk = "high" if stress_score >= 8 else "medium" if stress_score >= 5 else "low"
    out = {"emotion": emotion, "stress_score": stress_score, "risk_level": risk, "confidence": round(conf,4), "probabilities": {ID_TO_LABEL[i]: round(float(p),4) for i,p in enumerate(probs.tolist())}, "affect_latent_summary": affect_latent_summary(z_affect)}
    if audio_reliability is not None: out["audio_reliability"] = round(float(audio_reliability[0,0].detach().cpu()),4)
    if physio_reliability is not None: out["physio_reliability"] = round(float(physio_reliability[0,0].detach().cpu()),4)
    if face_reliability is not None: out["face_reliability"] = round(float(face_reliability[0,0].detach().cpu()),4)
    if router_weights is not None:
        rw = router_weights[0].detach().float().cpu().tolist()
        out["fusion_router_weights"] = {"audio_expert": round(rw[0],4), "physio_expert": round(rw[1],4), "face_expert": round(rw[2],4), "joint_expert": round(rw[3],4)}

    # Explicit affect tokens for prompt-aware downstream LLM reporting.
    affect_tokens = []
    affect_tokens.append(f"<AFFECT_{emotion.upper()}>")
    if stress_score >= 8:
        affect_tokens.append("<STRESS_HIGH>")
    elif stress_score >= 5:
        affect_tokens.append("<STRESS_MEDIUM>")
    else:
        affect_tokens.append("<STRESS_LOW>")
    affect_tokens.append(f"<RISK_{risk.upper()}>")
    if conf >= 0.75:
        affect_tokens.append("<CONFIDENCE_HIGH>")
    elif conf >= 0.45:
        affect_tokens.append("<CONFIDENCE_MEDIUM>")
    else:
        affect_tokens.append("<CONFIDENCE_LOW>")
    for name, val in [("AUDIO", audio_reliability), ("PHYSIO", physio_reliability), ("FACE", face_reliability)]:
        if val is None:
            affect_tokens.append(f"<{name}_MISSING>")
        else:
            r = float(val[0,0].detach().cpu())
            affect_tokens.append(f"<{name}_{'RELIABLE' if r >= 0.7 else 'UNCERTAIN' if r >= 0.3 else 'WEAK'}>")
    out["affect_tokens"] = affect_tokens
    if modality_weights is not None:
        mw = modality_weights[0].detach().float().cpu().tolist()
        out["fusion_modality_weights"] = {"audio": round(mw[0],4), "physio": round(mw[1],4), "face": round(mw[2],4)}
    if modality_mask is not None:
        mm = modality_mask[0].detach().float().cpu().tolist()
        out["modality_mask"] = {"audio": int(mm[0]), "physio": int(mm[1]), "face": int(mm[2])}
    return out


def run_emotion_inference(model: MultimodalEmotionModel, audio_tensor: Optional[torch.Tensor], physio_tensor: Optional[torch.Tensor], face_tensor: Optional[torch.Tensor] = None, device: str | torch.device = "auto", prompt_text: str | None = None) -> Dict[str, Any]:
    if device == "auto": device = next(model.parameters()).device
    audio = audio_tensor.to(device) if audio_tensor is not None and audio_tensor.numel() else None
    physio = physio_tensor.to(device) if physio_tensor is not None else None
    face = face_tensor.to(device) if face_tensor is not None else None
    prompt_ids = None
    if prompt_text:
        prompt_ids = model.prompt_ids_from_texts(prompt_text, device=torch.device(device) if not isinstance(device, torch.device) else device)
    with torch.no_grad(): out = model(audio=audio, physio=physio, face=face, prompt_ids=prompt_ids)
    if "fused_logits" in out:
        return prediction_to_struct(out["fused_logits"].detach().cpu(), z_affect=out.get("z_affect", None).detach().cpu() if "z_affect" in out else None, audio_reliability=out.get("audio_reliability", None).detach().cpu() if "audio_reliability" in out else None, physio_reliability=out.get("physio_reliability", None).detach().cpu() if "physio_reliability" in out else None, face_reliability=out.get("face_reliability", None).detach().cpu() if "face_reliability" in out else None, router_weights=out.get("fusion_router_weights", None).detach().cpu() if "fusion_router_weights" in out else None, modality_weights=out.get("fusion_modality_weights", None).detach().cpu() if "fusion_modality_weights" in out else None, modality_mask=out.get("fusion_modality_mask", None).detach().cpu() if "fusion_modality_mask" in out and out.get("fusion_modality_mask") is not None else None)
    for mod in ["audio", "physio", "face"]:
        if f"{mod}_logits" in out:
            return prediction_to_struct(out[f"{mod}_logits"].detach().cpu(), z_affect=out.get(f"{mod}_embedding", None).detach().cpu() if f"{mod}_embedding" in out else None, **{f"{mod}_reliability": out.get(f"{mod}_reliability", None).detach().cpu() if f"{mod}_reliability" in out else None})
    raise ValueError("No valid modality was provided")
