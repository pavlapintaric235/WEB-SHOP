"""CUDA preflight, one-epoch smoke test, and capped YOLO11 instance-segmentation fine-tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Portable segmentation data.yaml")
    parser.add_argument("--weights", type=Path, required=True, help="Pretrained YOLO11-seg .pt checkpoint")
    parser.add_argument("--mode", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/inventorylens_yolo11n_seg"))
    return parser.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d) -> bool:
    eps = 1e-12
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    def on_segment(first, second, point):
        return (min(first[0], second[0]) - eps <= point[0] <= max(first[0], second[0]) + eps
                and min(first[1], second[1]) - eps <= point[1] <= max(first[1], second[1]) + eps)

    return ((abs(o1) <= eps and on_segment(a, b, c)) or (abs(o2) <= eps and on_segment(a, b, d))
            or (abs(o3) <= eps and on_segment(c, d, a)) or (abs(o4) <= eps and on_segment(c, d, b)))


def _self_intersects(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or second == (first + 1) % count or (second + 1) % count == first:
                continue
            if _segments_intersect(a, b, points[second], points[(second + 1) % count]):
                return True
    return False


def parse_segmentation_label(path: Path) -> int:
    count = 0
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if fields[0] != "0" or len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Expected class-0 polygon row at {path}:{line_no}; bbox labels are not accepted")
        values = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid normalized polygon coordinate at {path}:{line_no}")
        points = list(zip(values[0::2], values[1::2]))
        if len(set(points)) < 3:
            raise ValueError(f"Degenerate product polygon at {path}:{line_no}")
        area2 = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])))
        if area2 <= 1e-12:
            raise ValueError(f"Zero-area product polygon at {path}:{line_no}")
        if _self_intersects(points):
            raise ValueError(f"Self-intersecting product polygon at {path}:{line_no}")
        key = tuple((round(x, 9), round(y, 9)) for x, y in points)
        rotations = [key[index:] + key[:index] for index in range(len(key))]
        reverse = tuple(reversed(key))
        rotations.extend(reverse[index:] + reverse[:index] for index in range(len(reverse)))
        canonical = min(rotations)
        if canonical in seen:
            raise ValueError(f"Duplicate product polygon in {path}:{line_no}")
        seen.add(canonical)
        count += 1
    return count


def resolve_and_validate(data_path: Path) -> tuple[dict, Path, dict]:
    data_path = data_path.expanduser().resolve(strict=True)
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = config.get("names") or {}
    names = {int(key): value for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    if names != {0: "product"}:
        raise ValueError(f"Expected one class 0=product, got {names}")
    root = Path(config.get("path") or ".")
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    split_counts = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    manifest_path = root / "dataset_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    for row in manifest:
        group_splits[row["group_id"]].add(row["split"])
        image = root / row["image_path"]
        label = root / row["label_path"]
        if image.stem != label.stem or not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Image-label pair does not align: {image} / {label}")
        if digest(image) != row["image_sha256"] or digest(label) != row["label_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {image.name} or its segmentation label")
        with Image.open(image) as opened:
            opened.verify()
        count = parse_segmentation_label(label)
        if count != int(row["annotation_count"]):
            raise ValueError(f"Polygon count mismatch for {image.name}: {count}")
        if row["original_filename"].lower() == "img_4699.jpeg":
            raise ValueError("Protected IMG_4699 leaked into training/validation manifest")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("One or more photographic arrangement groups cross dataset splits")
    for split, key in (("train", "train"), ("validation", "val")):
        image_dir = root / config[key]
        label_dir = root / "labels" / image_dir.name
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        labels = sorted(label_dir.glob("*.txt"))
        if {path.stem for path in images} != {path.stem for path in labels}:
            raise ValueError(f"Image/label stem mismatch in {split}")
        instances = sum(parse_segmentation_label(label) for label in labels)
        split_counts[split] = {"images": len(images), "instances": instances,
                               "groups": len({row["group_id"] for row in manifest if row["split"] == split})}
    return config, root, {"records": len(manifest), "splits": split_counts}


def require_cuda(device: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Stop before training and select a CUDA-enabled RunPod image.")
    index = int(device) if device.isdigit() else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device} requested, but only {torch.cuda.device_count()} are visible")
    props = torch.cuda.get_device_properties(index)
    print(json.dumps({"cuda": True, "torch": torch.__version__, "cuda_build": torch.version.cuda,
                      "device": torch.cuda.get_device_name(index),
                      "memory_gib": round(props.total_memory / 1024**3, 2)}, indent=2))


def run_train(weights: Path, data_yaml: Path, args: argparse.Namespace,
              name: str, epochs: int, patience: int) -> Path:
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(Path(__file__).resolve().parent),
    )
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if getattr(model, "task", None) != "segment":
        raise ValueError(f"Refusing detection-only checkpoint; expected task='segment', got {model.task!r}")
    save_dir = args.project.expanduser().resolve() / name
    if save_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing overwrite: {save_dir}")
    model.train(
        data=str(data_yaml), task="segment", epochs=epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, device=args.device,
        project=str(args.project), name=name, exist_ok=False, pretrained=True,
        patience=patience, save=True, save_period=5 if epochs > 1 else -1,
        cache=False, plots=True, val=True, amp=True, seed=42, deterministic=True,
        close_mosaic=5 if epochs > 5 else 0, max_det=300, verbose=True,
    )
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.is_file() and not last.is_file():
        raise RuntimeError(f"Training ended without a checkpoint under {save_dir / 'weights'}")
    return best if best.is_file() else last


def main() -> None:
    args = parse_args()
    if not 1 <= args.epochs <= 40:
        raise ValueError("Initial run is capped at 40 epochs")
    if args.patience < 1:
        raise ValueError("patience must be >= 1")
    weights = args.weights.expanduser().resolve(strict=True)
    if weights.suffix.lower() != ".pt":
        raise ValueError("YOLO segmentation fine-tuning requires a PyTorch .pt checkpoint")
    require_cuda(args.device)
    config, root, counts = resolve_and_validate(args.data)
    if counts["splits"] != {
        "train": {"images": 21, "instances": 130, "groups": 17},
        "validation": {"images": 5, "instances": 33, "groups": 5},
    }:
        raise ValueError(f"Unexpected reviewed dataset manifest: {counts}")
    resolved_data = args.project.expanduser().resolve() / "preflight" / "data_resolved.yaml"
    resolved_data.parent.mkdir(parents=True, exist_ok=True)
    portable = dict(config)
    portable["path"] = str(root)
    with resolved_data.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(portable, stream, sort_keys=False)
    print(json.dumps({"dataset": str(resolved_data), "weights": str(weights), "manifest": counts,
                      "protected_scenes_in_training": False, "training_started": False}, indent=2))

    smoke = run_train(weights, resolved_data, args, "smoke_1epoch", epochs=1, patience=1)
    print(f"One-epoch smoke test completed: {smoke}")
    if args.mode == "train":
        checkpoint = run_train(weights, resolved_data, args, "fine_tune", args.epochs, args.patience)
        print(f"Fine-tuning completed: {checkpoint}")
    else:
        print("Smoke-only mode; no multi-epoch training launched")


if __name__ == "__main__":
    main()
