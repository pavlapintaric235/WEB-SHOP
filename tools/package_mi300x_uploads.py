"""Build the two beginner-friendly upload ZIPs without replacing existing files."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "data/segmentation_dataset/experiments"
DATASET = EXPERIMENTS / "initial_yolo26n_seg_20261017"
KIT = ROOT / "data/segmentation_dataset/proposals/cvat22_yolo_staging_20261016/amd_rocm_training_package"
DATA_ZIP = EXPERIMENTS / "initial_yolo26n_seg_20261017_upload.zip"
KIT_ZIP = EXPERIMENTS / "mi300x_training_kit.zip"
CHECKSUMS = EXPERIMENTS / "mi300x_upload_checksums.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_tree(source: Path, target: Path, arc_prefix: str, filter_pycache: bool = True) -> None:
    if target.exists():
        raise SystemExit(f"Refusing to overwrite {target}")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for item in sorted(source.rglob("*")):
            if not item.is_file() or (filter_pycache and "__pycache__" in item.parts):
                continue
            relative = item.relative_to(source).as_posix()
            archive.write(item, f"{arc_prefix}/{relative}" if arc_prefix else relative)


def main() -> None:
    if not DATASET.is_dir() or not (DATASET / "integrity_report.json").is_file():
        raise SystemExit("Validated experiment dataset is missing")
    if not KIT.is_dir() or not (KIT / "README.md").is_file():
        raise SystemExit("MI300X training kit is missing")
    if any(path.exists() for path in (DATA_ZIP, KIT_ZIP, CHECKSUMS)):
        raise SystemExit("At least one upload bundle already exists; refusing to overwrite anything")
    estimate = sum(path.stat().st_size for path in DATASET.rglob("*") if path.is_file())
    estimate += sum(path.stat().st_size for path in KIT.rglob("*") if path.is_file())
    free = shutil.disk_usage(ROOT).free
    if free < estimate + 1024**3:
        raise SystemExit(f"Insufficient space to create ZIPs with a 1 GiB reserve: {free:,} free, {estimate:,} input bytes")

    zip_tree(DATASET, DATA_ZIP, "")
    package_files = [
        "README.md", "requirements-cloud.txt", "verify_rocm.py", "configure_cloud_dataset.py",
        "train_yolo26_seg.py", "benchmark_cpu.py", "evaluation_plan.md", "review_outcomes_template.csv",
        "mi300x_experiment_config.yaml",
    ]
    for name in package_files:
        if not (KIT / name).is_file():
            raise SystemExit(f"Missing kit file: {KIT / name}")
    # The standalone verifier is included alongside the scripts expected by train_yolo26_seg.py.
    with zipfile.ZipFile(KIT_ZIP, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name in package_files:
            archive.write(KIT / name, f"amd_rocm_training_package/{name}")
        verifier = ROOT / "tools/verify_yolo_experiment_dataset.py"
        archive.write(verifier, "amd_rocm_training_package/verify_yolo_experiment_dataset.py")
        proposal = ROOT / "data/segmentation_dataset/proposals/cvat22_yolo_staging_20261016/training_experiment_proposal.yaml"
        archive.write(proposal, "amd_rocm_training_package/training_experiment_proposal.yaml")

    report = {
        "data_zip": {"path": str(DATA_ZIP), "bytes": DATA_ZIP.stat().st_size, "sha256": sha256(DATA_ZIP)},
        "kit_zip": {"path": str(KIT_ZIP), "bytes": KIT_ZIP.stat().st_size, "sha256": sha256(KIT_ZIP)},
        "training_started": False,
        "cloud_launched": False,
    }
    CHECKSUMS.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
