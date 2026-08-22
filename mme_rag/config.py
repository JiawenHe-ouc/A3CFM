from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML dict: {path}")
    return data


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def get_by_path(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> Dict[str, Any]:
    """Parse CLI overrides in key=value form, e.g. train.batch_size=256."""
    updates: Dict[str, Any] = {}
    if not items:
        return updates
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        k, raw = item.split("=", 1)
        try:
            v = yaml.safe_load(raw)
        except Exception:
            v = raw
        set_by_path(updates, k, v)
    return updates


def add_common_config_args(parser: argparse.ArgumentParser, default_config: str) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=default_config, help="YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        help="Override YAML values, e.g. --set train.batch_size=512 train.epochs=80",
    )
    return parser


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml(args.config)
    updates = parse_overrides(args.overrides)
    return deep_update(cfg, updates)


def ensure_local_path(path: str | os.PathLike, name: str, must_exist: bool = True) -> Path:
    p = Path(path).expanduser()
    if must_exist and not p.exists():
        raise FileNotFoundError(
            f"{name} must be a local path and it does not exist: {p}\n"
            f"Do not use a Hugging Face repo id on an offline server."
        )
    return p
