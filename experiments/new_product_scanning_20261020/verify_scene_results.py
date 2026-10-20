"""Verify a downloaded 30-scene RunPod result ZIP before isolated import."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized(name: str) -> str:
    if name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path.as_posix()


def verify(archive_path: Path, expected_sha256: str) -> tuple[dict, dict[str, str]]:
    actual = digest(archive_path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"Result ZIP SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    with ZipFile(archive_path) as archive:
        bad_crc = archive.testzip()
        if bad_crc:
            raise ValueError(f"Result ZIP CRC failed for {bad_crc}")
        members = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = normalized(info.filename)
            if name in members:
                raise ValueError(f"Duplicate archive member: {name}")
            members[name] = info.filename
        required = {"SHA256SUMS", "report.json", "predictions.csv", "detections.json", "embeddings.npz"}
        if not required.issubset(members):
            raise ValueError(f"Missing result members: {sorted(required - members.keys())}")
        expected_files = {}
        for line in archive.read(members["SHA256SUMS"]).decode("utf-8").splitlines():
            hash_text, sep, filename = line.partition("  ")
            if not sep or len(hash_text) != 64:
                raise ValueError(f"Malformed SHA256SUMS line: {line!r}")
            filename = normalized(filename)
            if filename in expected_files:
                raise ValueError(f"Repeated checksum entry: {filename}")
            expected_files[filename] = hash_text.lower()
        actual_files = set(members) - {"SHA256SUMS"}
        if set(expected_files) != actual_files:
            raise ValueError("SHA256SUMS does not cover exactly the ZIP's result files")
        for name, expected in expected_files.items():
            if hashlib.sha256(archive.read(members[name])).hexdigest() != expected:
                raise ValueError(f"Inner result hash mismatch: {name}")
        report = json.loads(archive.read(members["report.json"]))
        detections = json.loads(archive.read(members["detections.json"]))
        predictions = list(csv.DictReader(io.StringIO(archive.read(members["predictions.csv"]).decode("utf-8-sig"))))
        galleries = report.get("recognizer", {}).get("gallery_conditions")
        if report.get("status") != "PASS" or report.get("scene_images") != 30 or galleries != ["original", "masked", "combined"]:
            raise ValueError("Run report is incomplete or has the wrong gallery conditions")
        if report.get("accuracy_metrics") is not None:
            raise ValueError("This inference-only run must not claim scene accuracy")
        if len(detections.get("records", [])) != 30 or len(predictions) != report.get("detected_boxes"):
            raise ValueError("Detector or prediction counts do not match the report")
        required_columns = {"photo_filename", "detection_id", "query_crop_sha256", "x1", "y1", "x2", "y2"}
        required_columns |= {f"{gallery}_top{rank}_sku_id" for gallery in galleries for rank in (1, 2, 3)}
        if predictions and not required_columns.issubset(predictions[0]):
            raise ValueError("Side-by-side gallery prediction columns are missing")
        with np.load(io.BytesIO(archive.read(members["embeddings.npz"])), allow_pickle=False) as embeddings:
            if embeddings["queries"].shape[0] != len(predictions):
                raise ValueError("Saved query embeddings do not match predicted detections")
            if embeddings["original_references"].shape[0] != 34 or embeddings["masked_references"].shape[0] != 34:
                raise ValueError("Saved paired reference embeddings do not match the 34-photo gallery")
            if len(embeddings["query_ids"]) != len(predictions):
                raise ValueError("Saved query IDs do not match predicted detections")
        return {"archive_sha256": actual, "scene_images": 30, "detected_boxes": len(predictions),
                "gallery_conditions": galleries, "verified_result_files": len(expected_files)}, members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--import-results", action="store_true",
                        help="Extract verified files to a new local results directory for the review app")
    args = parser.parse_args()
    summary, members = verify(args.archive, args.expected_sha256)
    if args.import_results:
        output = RESULTS / f"imported_{args.archive.stem}"
        output.mkdir(parents=True, exist_ok=False)
        with ZipFile(args.archive) as archive:
            for name, member in members.items():
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        summary["imported_to"] = str(output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
