from __future__ import annotations

# Unified affect ontology. Keep the order stable after training checkpoints.
LABELS = [
    "neutral",
    "stress",
    "amusement",
    "surprise",
]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

WESAD_LABEL_MAP = {
    1: "neutral",   # baseline
    2: "stress",    # stress
    3: "amusement", # happy -> amusement (4-class target)
    4: "neutral",   # meditation; collapse to neutral for this task
}

# Shared label aliases for audio and facial expression datasets.
LABEL_ALIASES = {
    "neutral": "neutral", "neu": "neutral", "baseline": "neutral", "normal": "neutral",
    "calm": "neutral", "meditation": "neutral", "relaxed": "neutral", "relax": "neutral",
    "happy": "amusement", "happiness": "amusement", "hap": "amusement", "exc": "amusement", "excited": "amusement",
    "amusement": "amusement", "positive": "amusement",
    "stress": "stress", "stressed": "stress", "tense": "stress", "anxious": "stress", "anxiety": "stress",
    "sad": "stress", "sadness": "stress", "sa": "stress",
    "angry": "stress", "anger": "stress", "ang": "stress",
    "fear": "stress", "fearful": "stress", "fea": "stress",
    "surprise": "surprise", "surprised": "surprise", "sur": "surprise",
    "disgust": "stress", "disgusted": "stress", "dis": "stress",
    "fru": "stress", "frustrated": "stress", "frustration": "stress",
    "str": "stress", "amu": "amusement",
}
AUDIO_LABEL_ALIASES = LABEL_ALIASES
FACE_LABEL_ALIASES = LABEL_ALIASES

# Labels that should usually be skipped for supervised affect training.
IEMOCAP_IGNORE_LABELS = {"xxx", "oth", "other", "unknown"}
FACE_IGNORE_LABELS = {"unknown", "other", "contempt", "none"}

DEFAULT_PHYSIO_CHANNELS = ["ACC_x", "ACC_y", "ACC_z", "ECG", "EDA", "EMG", "Resp", "Temp"]
DEFAULT_OVERLAP_LABELS = ["neutral", "stress", "amusement", "surprise"]

FACE_LABEL_ALIASES = {
    "neutral": "neutral",
    "happy": "amusement",
    "happiness": "amusement",
    "amusement": "amusement",
    "sad": "stress",
    "sadness": "stress",
    "angry": "stress",
    "anger": "stress",
    "fear": "stress",
    "fearful": "stress",
    "surprise": "surprise",
    "surprised": "surprise",
    "disgust": "stress",
    "disgusted": "stress",
    "contempt": "stress",
    "calm": "neutral",
    "stress": "stress",
    # common integer encodings found in small FER manifests; adjust per dataset.
    "0": "neutral",
    "1": "amusement",
    "2": "stress",
    "3": "surprise",
    "4": "stress",
    "5": "stress",
    "6": "stress",
}
