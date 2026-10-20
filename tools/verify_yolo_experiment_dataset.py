"""Validate a materialized YOLO segmentation experiment dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


EXPECTED = {"train": (21, 130), "validation": (5, 33)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )) / 2


def orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    eps = 1e-12
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def intersects(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    eps = 1e-12
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True
    return (
        (abs(o1) <= eps and on_segment(a, b, c))
        or (abs(o2) <= eps and on_segment(a, b, d))
        or (abs(o3) <= eps and on_segment(c, d, a))
        or (abs(o4) <= eps and on_segment(c, d, b))
    )


def self_intersects(points: list[tuple[float, float]]) -> bool:
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or (j + 1) % n == i:
                continue
            c, d = points[j], points[(j + 1) % n]
            if intersects(a, b, c, d):
                return True
    return False


def parse_label(path: Path) -> int:
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[tuple[float, ...]] = set()
    for row_no, fields in enumerate(lines, 1):
        if len(fields) < 7 or fields[0] != "0" or (len(fields) - 1) % 2:
            raise ValueError(f"Invalid class/shape at {path}:{row_no}")
        values = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError(f"Non-finite or out-of-range coordinate at {path}:{row_no}")
        points = list(zip(values[0::2], values[1::2]))
        if len(set(points)) < 3 or polygon_area(points) <= 1e-12 or self_intersects(points):
            raise ValueError(f"Degenerate/self-intersecting polygon at {path}:{row_no}")
        normalized = tuple(round(value, 9) for value in values)
        if normalized in seen:
            raise ValueError(f"Exact duplicate polygon in {path}:{row_no}")
        seen.add(normalized)
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    if config.get("names") not in ({0: "product"}, ["product"]):
        raise SystemExit("data.yaml must define only class 0 = product")
    if Path(config.get("path", "")).resolve() != root:
        raise SystemExit("data.yaml path does not resolve to the dataset root")
    if config.get("train") != "images/train" or config.get("val") != "images/validation":
        raise SystemExit("Unexpected train/validation image paths")

    with (root / "dataset_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 26:
        raise SystemExit(f"Expected 26 manifest records, got {len(rows)}")
    splits = defaultdict(lambda: {"images": 0, "instances": 0})
    groups: dict[str, set[str]] = defaultdict(set)
    filenames: set[str] = set()
    warnings: list[str] = []
    for row in rows:
        split = row["split"]
        if split not in EXPECTED:
            warnings.append(f"Unexpected split: {split}")
            continue
        image = root / row["image_path"]
        label = root / row["label_path"]
        if not image.is_file() or not label.is_file() or image.stem != label.stem:
            warnings.append(f"Missing or mismapped pair: {row['image_path']} / {row['label_path']}")
            continue
        if image.name in filenames:
            warnings.append(f"Duplicate image filename: {image.name}")
        filenames.add(image.name)
        if sha256(image) != row["image_sha256"] or sha256(label) != row["label_sha256"]:
            warnings.append(f"Checksum mismatch: {image.name}")
        with Image.open(image) as opened:
            opened.verify()
        with Image.open(image) as opened:
            if opened.size != (int(row["width"]), int(row["height"])):
                warnings.append(f"Image dimension mismatch: {image.name}")
        count = parse_label(label)
        if count != int(row["annotation_count"]):
            warnings.append(f"Annotation count mismatch: {image.name}")
        splits[split]["images"] += 1
        splits[split]["instances"] += count
        groups[row["group_id"]].add(split)

    expected = {name: {"images": values[0], "instances": values[1]} for name, values in EXPECTED.items()}
    if dict(splits) != expected:
        warnings.append(f"Split totals differ: {dict(splits)}")
    leaked_groups = [group for group, members in groups.items() if len(members) != 1]
    if leaked_groups:
        warnings.append(f"Groups cross split boundary: {leaked_groups}")
    if warnings:
        raise SystemExit("DATASET VALIDATION FAILED:\n- " + "\n- ".join(warnings))

    report = {
        "status": "pass",
        "root": str(root),
        "images": len(rows),
        "annotations": sum(value["instances"] for value in expected.values()),
        "class": {"0": "product"},
        "splits": expected,
        "files_checked": len(rows) * 2 + 1,
        "checks": ["image decoding", "image-label stem mapping", "sha256", "dimensions", "counts", "class ids", "normalized coordinates", "polygon area", "self-intersections", "duplicate polygons within image", "group leakage"],
    }
    (root / "integrity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
