#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from mme_rag.datasets import split_audio_manifest


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test split column for an audio manifest.")
    parser.add_argument("--input", required=True, help="Input CSV with path,label and optional speaker_id")
    parser.add_argument("--output", default="data/audio_manifest.csv", help="Output CSV with split column")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_audio_manifest(args.input, args.output, args.val_ratio, args.test_ratio, args.seed)
    print(f"Saved split manifest to {args.output}")


if __name__ == "__main__":
    main()
