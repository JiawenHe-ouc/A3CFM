from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def format_sensevoice_text(raw: str) -> str:
    # Lightweight cleanup. Keep this separate so you can paste your original
    # format_str_v3 implementation if you want exactly the same emoji behavior.
    if not raw:
        return ""
    for token in ["<|zh|>", "<|en|>", "<|yue|>", "<|ja|>", "<|ko|>", "<|nospeech|>", "<|Speech|>", "<|withitn|>", "<|woitn|>"]:
        raw = raw.replace(token, "")
    emo_map = {
        "<|HAPPY|>": " happy",
        "<|SAD|>": " sad",
        "<|ANGRY|>": " angry",
        "<|NEUTRAL|>": " neutral",
        "<|FEARFUL|>": " fearful",
        "<|DISGUSTED|>": " disgusted",
        "<|SURPRISED|>": " surprised",
    }
    for k, v in emo_map.items():
        raw = raw.replace(k, v)
    return " ".join(raw.split())


class SenseVoiceASR:
    def __init__(self, model_path: str, vad_model: str = "fsmn-vad", offline: bool = True):
        self.model_path = model_path
        self.model = None
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self._load(vad_model)

    def _load(self, vad_model: str):
        try:
            from funasr import AutoModel
        except Exception as e:
            raise ImportError("funasr is required for SenseVoice ASR. pip install funasr modelscope") from e
        path = self.model_path
        if not (Path(path).exists() or path == "iic/SenseVoiceSmall"):
            raise FileNotFoundError(
                f"SenseVoice path does not exist locally: {path}. "
                "Use a local directory such as ./iic/SenseVoiceSmall or install/cache iic/SenseVoiceSmall."
            )
        try:
            self.model = AutoModel(
                model=path,
                vad_model=vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=True,
            )
        except AssertionError:
            # Some FunASR versions do not register local paths. Try canonical id if cached.
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model=vad_model,
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=True,
            )

    def transcribe(self, audio: np.ndarray, language: str = "auto") -> Dict[str, str]:
        text = self.model.generate(
            input=audio,
            cache={},
            language=language or "auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
        )
        raw = text[0].get("text", "") if text else ""
        return {"raw": raw, "text": format_sensevoice_text(raw)}
