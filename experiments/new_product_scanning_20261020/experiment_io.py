"""File-only helpers for the isolated new-product scanning experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
AUDIT = EXPERIMENT / "audit"
ANNOTATIONS = EXPERIMENT / "annotations"
ARTIFACTS = EXPERIMENT / "artifacts"
RESULTS = EXPERIMENT / "results"
RUNTIME = EXPERIMENT / "runtime"
MANIFEST_PATH = AUDIT / "photo_manifest.csv"
REGISTRY_PATH = AUDIT / "candidate_skus.csv"
REVIEWED_REGISTRY_PATH = ANNOTATIONS / "sku_registry_review.json"
SCENE_LABELS_PATH = ANNOTATIONS / "scene_labels.json"
REGISTRY_EVENTS_PATH = ANNOTATIONS / "sku_registry_review_events.jsonl"
SCENE_EVENTS_PATH = ANNOTATIONS / "scene_review_events.jsonl"
MASK_EVENTS_PATH = ANNOTATIONS / "mask_review_events.jsonl"

os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME / "matplotlib"))
os.environ.setdefault("WEBSHOP_HF_CACHE", str(ROOT / ".venv" / "huggingface_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest() -> list[dict[str, str]]:
    return read_rows(MANIFEST_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_image(row: dict[str, str]):
    from PIL import Image, ImageOps

    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Source photo is unavailable: {path}")
    if sha256_file(path) != row["sha256"]:
        raise ValueError(f"Source image hash changed since audit: {path.name}")
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def load_registry() -> list[dict[str, Any]]:
    if REVIEWED_REGISTRY_PATH.is_file():
        return json.loads(REVIEWED_REGISTRY_PATH.read_text(encoding="utf-8"))
    return [{
        "sku_id": row["sku_id"], "proposed_name": row["proposed_name"],
        "brand": row["brand"], "variant_or_size": row["variant_or_size"],
        "reference_files": row["reference_files"].split(";"),
        "identity_status": row["identity_status"], "angle_status": row["angle_status"],
        "identity_confirmed": False, "owner_notes": row["owner_confirmation"],
    } for row in read_rows(REGISTRY_PATH)]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_registry(records: list[dict[str, Any]]) -> None:
    previous = {row["sku_id"]: row for row in load_registry()}
    atomic_json(REVIEWED_REGISTRY_PATH, records)
    with REGISTRY_EVENTS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        for row in records:
            old = previous.get(row["sku_id"])
            if old != row:
                handle.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "candidate_sku_id": row["sku_id"], "previous": old, "updated": row,
                    "event": "human_registration_review_saved",
                }, ensure_ascii=True) + "\n")


def load_scene_labels() -> dict[str, dict[str, Any]]:
    if not SCENE_LABELS_PATH.is_file():
        return {}
    return json.loads(SCENE_LABELS_PATH.read_text(encoding="utf-8"))


def save_scene_labels(records: dict[str, dict[str, Any]]) -> None:
    previous = load_scene_labels()
    atomic_json(SCENE_LABELS_PATH, records)
    with SCENE_EVENTS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        for name, row in records.items():
            old = previous.get(name)
            if old != row:
                handle.write(json.dumps({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "photo_filename": name, "source_sha256": row.get("source_sha256"),
                    "previous": old, "updated": row,
                    "event": "human_scene_review_saved",
                }, ensure_ascii=True) + "\n")


def latest_mask_events() -> dict[str, dict[str, Any]]:
    latest = {}
    if MASK_EVENTS_PATH.is_file():
        with MASK_EVENTS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    event = json.loads(line)
                    latest[event["source_file"]] = event
    return latest


def append_mask_event(event: dict[str, Any]) -> None:
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    with MASK_EVENTS_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def sku_by_reference() -> dict[str, str]:
    return {filename: item["sku_id"] for item in load_registry() for filename in item["reference_files"]}


def siglip_snapshot() -> Path:
    return (ROOT / ".venv" / "huggingface_cache" / "hub" /
            "models--google--siglip2-base-patch16-224" / "snapshots" /
            "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2")


def load_reviewed_mask(filename: str):
    event = latest_mask_events().get(filename)
    if not event or event.get("decision") != "approved" or not event.get("mask_file"):
        return None, event
    path = EXPERIMENT / event["mask_file"]
    if not path.is_file():
        return None, event
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        return data["mask"].astype(bool), event

