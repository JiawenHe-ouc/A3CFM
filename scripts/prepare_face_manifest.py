#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 三套别名表将不同数据集的标签统一映射为相同的规范名
LABEL_ALIASES: Dict[str, str] = {
    "fear": "fearful",
    "fearful": "fearful",
    "surprise": "surprised",
    "surprised": "surprised",
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "happiness": "happy",
    "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "sad": "sad",
    "calm": "calm",
    "stress": "stress",
}

# RAF-DB / FER2013 integer class folder convention.
RAF_FER_INT_LABEL_ALIASES: Dict[str, str] = {
    "0": "angry",
    "1": "disgusted",
    "2": "fearful",
    "3": "happy",
    "4": "sad",
    "5": "surprised",
    "6": "neutral",
}


AFFECTNET_INT_LABEL_ALIASES: Dict[str, Optional[str]] = {
    "0": "neutral",
    "1": "amusement",
    "2": "stress",
    "3": "surprise",
    "4": "stress",
    "5": "stress",
    "6": "stress",
    "7": None,
}

AFFECTNET_NAME_ALIASES: Dict[str, Optional[str]] = {
    "neutral": "neutral",
    "happy": "amusement",
    "happiness": "amusement",
    "sad": "stress",
    "sadness": "stress",
    "surprise": "surprise",
    "surprised": "surprise",
    "fear": "stress",
    "fearful": "stress",
    "disgust": "stress",
    "disgusted": "stress",
    "anger": "stress",
    "angry": "stress",
    "contempt": None,
}

# Normalize split folder names.
SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "training": "train",
    "trainset": "train",
    "train_set": "train",
    "valid": "val",
    "validation": "val",
    "validset": "val",
    "valid_set": "val",
    "val": "val",
    "valset": "val",
    "val_set": "val",
    "test": "test",
    "testing": "test",
    "testset": "test",
    "test_set": "test",
}


def _norm_key(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _is_affectnet_name(name: str) -> bool:
    return "affectnet" in str(name).lower().replace("_", "").replace("-", "")


def normalize_label(
    name: str,
    keep_original_label: bool = False,
    dataset_name: str = "face_dataset",
    allow_unknown: bool = False,
) -> Optional[str]:

    label = name.strip()
    if keep_original_label:
        return label

    key = _norm_key(label)
    is_affectnet = _is_affectnet_name(dataset_name)

    if is_affectnet and key in AFFECTNET_INT_LABEL_ALIASES:
        return AFFECTNET_INT_LABEL_ALIASES[key]
    if is_affectnet and key in AFFECTNET_NAME_ALIASES:
        return AFFECTNET_NAME_ALIASES[key]
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    if key in RAF_FER_INT_LABEL_ALIASES:
        return RAF_FER_INT_LABEL_ALIASES[key]

    if allow_unknown:
        return None
    allowed = sorted(set(LABEL_ALIASES.keys()) | set(RAF_FER_INT_LABEL_ALIASES.keys()) | set(AFFECTNET_NAME_ALIASES.keys()) | set(AFFECTNET_INT_LABEL_ALIASES.keys()))
    raise ValueError(
        f"Unknown face label folder or annotation label '{name}'. "
        f"Known labels/aliases are: {', '.join(allowed)}. "
        f"For AffectNet, pass --dataset-name AffectNet or use the auto-detected root. "
        f"Use --keep-original-label only if your downstream code supports these labels."
    )


def normalize_split(name: str) -> str | None:
    raw = name.strip().lower()
    key = _norm_key(name)
    return SPLIT_ALIASES.get(raw) or SPLIT_ALIASES.get(key)


def iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _has_any_image(folder: Path) -> bool:
    try:
        next(iter_images(folder))
        return True
    except StopIteration:
        return False


def _contains_known_layout(root: Path) -> bool:
    return bool(find_split_dirs(root)) or bool(find_affectnet_split_dirs(root)) or _looks_like_affectnet_annotation_root(root)

# 嵌套目录处理
def maybe_unwrap_nested_root(root: Path) -> Path:
    current = root
    visited = set()
    while True:
        if current in visited:
            break
        visited.add(current)
        if _contains_known_layout(current):
            break
        children = [p for p in sorted(current.iterdir()) if p.is_dir() and not p.name.startswith(".")]
        if len(children) == 1:
            child = children[0]
            if child.name.lower() == current.name.lower() or _is_affectnet_name(child.name) or _contains_known_layout(child):
                print(f"Auto-descending into nested dataset folder: {child}")
                current = child
                continue
        break
    return current


def find_nested_dataset_root(root: Path, max_depth: int = 4) -> Optional[Path]:

    queue: List[Tuple[Path, int]] = [(root, 0)]
    seen = {root}
    while queue:
        cur, depth = queue.pop(0)
        if cur != root and _contains_known_layout(cur):
            return cur
        if cur != root:
            label_like = 0
            for child in cur.iterdir():
                if not child.is_dir():
                    continue
                try:
                    lab = normalize_label(child.name, dataset_name="AffectNet", allow_unknown=True)
                except Exception:
                    lab = None
                if lab is not None and _has_any_image(child):
                    label_like += 1
            if label_like >= 2:
                return cur
        if depth >= max_depth:
            continue
        for child in sorted(cur.iterdir()):
            if child.is_dir() and child not in seen and not child.name.startswith("."):
                seen.add(child)
                if _is_affectnet_name(child.name) or normalize_split(child.name) is not None or child.name.lower() in {"images", "annotations"}:
                    queue.append((child, depth + 1))
                elif depth < 1:
                    queue.append((child, depth + 1))
    return None


def find_split_dirs(root: Path) -> Dict[str, Path]:
    split_dirs: Dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        split = normalize_split(child.name)
        if split is not None:
            split_dirs[split] = child
    return split_dirs


def _find_child_dir(parent: Path, names: Iterable[str]) -> Optional[Path]:
    wanted = {_norm_key(x) for x in names}
    if not parent.exists():
        return None
    for p in sorted(parent.iterdir()):
        if p.is_dir() and _norm_key(p.name) in wanted:
            return p
    return None


def find_affectnet_split_dirs(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        split = normalize_split(child.name)
        if split is None:
            continue
        img_dir = _find_child_dir(child, ["images", "image", "imgs", "aligned", "faces"])
        ann_dir = _find_child_dir(child, ["annotations", "annotation", "annos", "anno"])
        if img_dir is not None and ann_dir is not None:
            out[split] = child
    return out


def _looks_like_affectnet_annotation_root(root: Path) -> bool:
    return bool(find_affectnet_split_dirs(root))


def build_from_rafdb_split_folders(
    root: Path,
    keep_original_label: bool = False,
    dataset_name: str = "RAF-DB",
    allow_unknown: bool = False,
) -> List[dict]:
    split_dirs = find_split_dirs(root)
    if not split_dirs:
        raise RuntimeError(
            f"No split folders found under {root}. Expected train/, valid/ or val/, and test/."
        )

    rows: List[dict] = []
    skipped_unknown = 0

    for split, split_dir in sorted(split_dirs.items()):
        if _find_child_dir(split_dir, ["images", "image", "imgs", "aligned", "faces"]) is not None and _find_child_dir(split_dir, ["annotations", "annotation", "annos", "anno"]) is not None:
            return build_from_affectnet_annotations(root, keep_original_label=keep_original_label, dataset_name=dataset_name, allow_unknown=allow_unknown)

        label_dirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
        if not label_dirs:
            raise RuntimeError(f"No label folders found under {split_dir}")

        for label_dir in label_dirs:
            label = normalize_label(
                label_dir.name,
                keep_original_label=keep_original_label,
                dataset_name=dataset_name,
                allow_unknown=allow_unknown,
            )
            if label is None:
                skipped_unknown += len(list(iter_images(label_dir)))
                continue
            for img_path in iter_images(label_dir):
                rows.append({"path": str(img_path.resolve()), "label": label, "split": split, "dataset": dataset_name})

    if not rows:
        raise RuntimeError(f"No images found under split-folder root: {root}; skipped_unknown={skipped_unknown}")
    if skipped_unknown:
        print(f"Skipped {skipped_unknown} images with ignored/unknown labels.")
    return rows

# 随机 Split 划分
def build_from_single_class_folder_root(
    root: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    keep_original_label: bool = False,
    dataset_name: str = "face_dataset",
    allow_unknown: bool = False,
) -> List[dict]:
    """Folders like root/happy/* or root/0/*. Randomly split each class."""
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be non-negative and sum to < 1.")

    rng = random.Random(seed)
    rows: List[dict] = []
    skipped_unknown = 0

    label_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not label_dirs:
        raise RuntimeError(f"No label folders found under {root}")

    for label_dir in label_dirs:
        if _is_affectnet_name(label_dir.name) or _contains_known_layout(label_dir):
            nested = maybe_unwrap_nested_root(label_dir)
            if nested != root:
                try:
                    if find_affectnet_split_dirs(nested):
                        rows.extend(build_from_affectnet_annotations(nested, keep_original_label=keep_original_label, dataset_name=dataset_name if _is_affectnet_name(dataset_name) else "AffectNet", allow_unknown=True))
                        continue
                    if find_split_dirs(nested):
                        rows.extend(build_from_rafdb_split_folders(nested, keep_original_label=keep_original_label, dataset_name=dataset_name, allow_unknown=allow_unknown))
                        continue
                except Exception:
                    pass

        label = normalize_label(
            label_dir.name,
            keep_original_label=keep_original_label,
            dataset_name=dataset_name,
            allow_unknown=allow_unknown,
        )
        files = list(iter_images(label_dir))
        if label is None:
            skipped_unknown += len(files)
            continue
        rng.shuffle(files)

        n = len(files)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        for i, img_path in enumerate(files):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            rows.append({"path": str(img_path.resolve()), "label": label, "split": split, "dataset": dataset_name})

    if not rows:
        raise RuntimeError(f"No images found under {root}; skipped_unknown={skipped_unknown}")
    if skipped_unknown:
        print(f"Skipped {skipped_unknown} images with ignored/unknown labels.")
    return rows


def _read_affectnet_exp(annotation_path: Path) -> Optional[int]:
    suffix = annotation_path.suffix.lower()
    try:
        if suffix == ".npy":
            import numpy as np
            arr = np.load(annotation_path, allow_pickle=True)
            if hasattr(arr, "item"):
                try:
                    return int(arr.item())
                except Exception:
                    pass
            return int(arr.reshape(-1)[0])
        text = annotation_path.read_text(encoding="utf-8", errors="ignore").strip()
        # Accept files that contain just a number or a simple CSV first field.
        token = text.replace("\n", ",").split(",")[0].strip()
        return int(float(token))
    except Exception:
        return None


def _find_affectnet_exp_file(ann_dir: Path, image_path: Path) -> Optional[Path]:
    rel_stem = image_path.stem
    candidates = [
        ann_dir / f"{rel_stem}_exp.npy",
        ann_dir / f"{rel_stem}_expression.npy",
        ann_dir / f"{rel_stem}_expr.npy",
        ann_dir / f"{rel_stem}.npy",
        ann_dir / f"{rel_stem}_exp.txt",
        ann_dir / f"{rel_stem}.txt",
    ]
    for suffix in ["_exp.npy", "_expression.npy", "_expr.npy", ".npy", "_exp.txt", ".txt"]:
        candidates.append(ann_dir / f"{image_path.with_suffix('').name}{suffix}")
    for c in candidates:
        if c.exists():
            return c
    for pat in [f"**/{rel_stem}_exp.npy", f"**/{rel_stem}.npy", f"**/{rel_stem}_exp.txt", f"**/{rel_stem}.txt"]:
        found = list(ann_dir.glob(pat))
        if found:
            return found[0]
    return None

# 为每张图片匹配对应的表情标注文件
def build_from_affectnet_annotations(
    root: Path,
    keep_original_label: bool = False,
    dataset_name: str = "AffectNet",
    allow_unknown: bool = True,
) -> List[dict]:
    """Build manifest from AffectNet train_set/images + annotations layout."""
    split_dirs = find_affectnet_split_dirs(root)
    if not split_dirs:
        raise RuntimeError(
            f"No AffectNet annotation layout found under {root}. Expected train_set/images and train_set/annotations."
        )
    rows: List[dict] = []
    missing_ann = 0
    skipped_label = 0

    for split, split_dir in sorted(split_dirs.items()):
        img_dir = _find_child_dir(split_dir, ["images", "image", "imgs", "aligned", "faces"])
        ann_dir = _find_child_dir(split_dir, ["annotations", "annotation", "annos", "anno"])
        if img_dir is None or ann_dir is None:
            continue
        for img_path in iter_images(img_dir):
            ann = _find_affectnet_exp_file(ann_dir, img_path)
            if ann is None:
                missing_ann += 1
                continue
            exp = _read_affectnet_exp(ann)
            if exp is None:
                missing_ann += 1
                continue
            label = normalize_label(
                str(exp),
                keep_original_label=keep_original_label,
                dataset_name="AffectNet",
                allow_unknown=allow_unknown,
            )
            if label is None:
                skipped_label += 1
                continue
            rows.append({"path": str(img_path.resolve()), "label": label, "split": split, "dataset": dataset_name})

    if not rows:
        raise RuntimeError(
            f"No usable AffectNet images found under {root}; missing_ann={missing_ann}, skipped_label={skipped_label}"
        )
    if missing_ann:
        print(f"Skipped {missing_ann} images without readable expression annotation.")
    if skipped_label:
        print(f"Skipped {skipped_label} AffectNet samples with ignored labels, e.g. contempt.")
    return rows


def write_manifest(rows: List[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "split", "dataset"]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: List[dict], out: Path) -> None:
    summary_path = out.with_suffix(".summary.csv")
    counts = Counter((r["split"], r["label"]) for r in rows)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "count"])
        for (split, label), count in sorted(counts.items()):
            writer.writerow([split, label, count])

    print(f"Wrote summary to {summary_path}")


def print_summary(rows: List[dict]) -> None:
    by_split = defaultdict(int)
    by_pair = Counter((r["split"], r["label"]) for r in rows)

    for r in rows:
        by_split[r["split"]] += 1

    print("Split counts:")
    for split in ["train", "val", "test"]:
        if split in by_split:
            print(f"  {split}: {by_split[split]}")

    print("Label counts by split:")
    for (split, label), count in sorted(by_pair.items()):
        print(f"  {split:5s} {label:10s} {count}")


def infer_dataset_name(root: Path, provided: str) -> str:
    if provided and provided != "auto":
        return provided
    parts = [root.name] + [p.name for p in list(root.parents)[:3]]
    if any(_is_affectnet_name(x) for x in parts):
        return "AffectNet"
    return "RAF-DB"

# 主清单，每行一张图片的绝对路径、标签、split、数据集名
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create face_manifest.csv for AffectNet, RAF-DB, or generic class-folder face datasets. "
            "Supported layouts include root/train/<label>, root/train_set/<label>, "
            "and AffectNet root/train_set/images + root/train_set/annotations."
        )
    )
    parser.add_argument("--root", required=True, help="Dataset root folder, e.g. /path/to/AffectNet or /path/to/RAF-DB.")
    parser.add_argument("--out", default="data/face_manifest.csv", help="Output CSV path.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "rafdb", "class_folders", "affectnet"],
        help=(
            "auto: detect AffectNet annotation layout or split folders; "
            "rafdb: force split-folder layout; "
            "class_folders: root/<label>/* with random split; "
            "affectnet: force AffectNet train_set/images + annotations layout."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default="auto",
        help="Dataset name stored in manifest. Use AffectNet for AffectNet numeric expression labels. Default: auto.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Only used in class_folders mode.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Only used in class_folders mode.")
    parser.add_argument("--seed", type=int, default=42, help="Only used in class_folders mode.")
    parser.add_argument(
        "--keep-original-label",
        action="store_true",
        help="Keep folder names/annotation ids unchanged. Not recommended unless downstream label mapping supports them.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Skip unknown or unsupported labels instead of raising an error.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    root = maybe_unwrap_nested_root(root)

    dataset_name = infer_dataset_name(root, args.dataset_name)
    out = Path(args.out)

    if args.mode == "affectnet":
        rows = build_from_affectnet_annotations(
            root=root,
            keep_original_label=args.keep_original_label,
            dataset_name=dataset_name,
            allow_unknown=True,
        )
    elif args.mode == "rafdb":
        rows = build_from_rafdb_split_folders(
            root=root,
            keep_original_label=args.keep_original_label,
            dataset_name=dataset_name,
            allow_unknown=args.allow_unknown,
        )
    elif args.mode == "class_folders":
        rows = build_from_single_class_folder_root(
            root=root,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            keep_original_label=args.keep_original_label,
            dataset_name=dataset_name,
            allow_unknown=args.allow_unknown,
        )
    else:
        if find_affectnet_split_dirs(root):
            rows = build_from_affectnet_annotations(
                root=root,
                keep_original_label=args.keep_original_label,
                dataset_name=dataset_name if _is_affectnet_name(dataset_name) else "AffectNet",
                allow_unknown=True,
            )
        elif find_split_dirs(root):
            rows = build_from_rafdb_split_folders(
                root=root,
                keep_original_label=args.keep_original_label,
                dataset_name=dataset_name,
                allow_unknown=args.allow_unknown,
            )
        else:
            nested_root = find_nested_dataset_root(root)
            if nested_root is not None and nested_root != root:
                print(f"Detected nested face dataset root: {nested_root}")
                if find_affectnet_split_dirs(nested_root):
                    rows = build_from_affectnet_annotations(
                        root=nested_root,
                        keep_original_label=args.keep_original_label,
                        dataset_name=dataset_name if _is_affectnet_name(dataset_name) else "AffectNet",
                        allow_unknown=True,
                    )
                elif find_split_dirs(nested_root):
                    rows = build_from_rafdb_split_folders(
                        root=nested_root,
                        keep_original_label=args.keep_original_label,
                        dataset_name=dataset_name,
                        allow_unknown=args.allow_unknown,
                    )
                else:
                    rows = build_from_single_class_folder_root(
                        root=nested_root,
                        val_ratio=args.val_ratio,
                        test_ratio=args.test_ratio,
                        seed=args.seed,
                        keep_original_label=args.keep_original_label,
                        dataset_name=dataset_name,
                        allow_unknown=args.allow_unknown,
                    )
            else:
                rows = build_from_single_class_folder_root(
                    root=root,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=args.seed,
                    keep_original_label=args.keep_original_label,
                    dataset_name=dataset_name,
                    allow_unknown=args.allow_unknown,
                )

    write_manifest(rows, out)
    write_summary(rows, out)
    print(f"Wrote {len(rows)} rows to {out}")
    print_summary(rows)


if __name__ == "__main__":
    main()
