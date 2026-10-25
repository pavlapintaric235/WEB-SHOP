"""Package the fine-tuned DINO checkpoint and held-out scenes for TAO inference."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ARTIFACTS = EXP / "artifacts" / "third_set_review_20261020"
INPUTS = ARTIFACTS / "inputs"
PREDICTIONS = ARTIFACTS / "predictions.json"
LOCAL_TRAIN = Path(r"C:\Users\U938\train")
BUILD = Path(__file__).resolve().parent / ".build_dino_eval"
ARCHIVE = Path(__file__).resolve().parent / "dino_eval_20261021.zip"
PACKAGE = "inventorylens-dino-eval"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint = LOCAL_TRAIN / "model_epoch_011.pth"
    experiment = LOCAL_TRAIN / "experiment.yaml"
    for required in (checkpoint, experiment, PREDICTIONS):
        if not required.is_file():
            raise FileNotFoundError(required)

    expected = {
        Path(scene["local_path"]).name: scene["sha256"].lower()
        for scene in json.loads(PREDICTIONS.read_text(encoding="utf-8"))["scenes"]
    }
    if len(expected) != 30:
        raise ValueError(f"Expected 30 held-out scenes in predictions.json, found {len(expected)}")

    sources = [p for p in INPUTS.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if len(sources) != 30 or {p.name for p in sources} != set(expected):
        raise ValueError(f"Held-out input set mismatch: found {len(sources)} files")
    for source in sources:
        if sha256(source) != expected[source.name]:
            raise ValueError(f"Held-out source hash mismatch: {source}")

    if BUILD.exists():
        shutil.rmtree(BUILD)
    package = BUILD / PACKAGE
    image_dir = package / "data" / "images" / "infer"
    image_dir.mkdir(parents=True)
    for source in sources:
        shutil.copy2(source, image_dir / source.name)
    shutil.copy2(checkpoint, package / checkpoint.name)
    shutil.copy2(experiment, package / "experiment.yaml")
    (package / "classmap.txt").write_text("product\n", encoding="utf-8")

    manifest = {
        "checkpoint": checkpoint.name,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "held_out_images": len(sources),
        "employee_images": len(list((INPUTS / "new10_employee").glob("*"))),
        "customer_images": len(list((INPUTS / "new10_customer").glob("*"))),
        "images": {p.name: sha256(p) for p in sorted(sources)},
        "metric_note": "Inference-only review; held-out scene boxes have not been human-annotated.",
    }
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (package / "RUN_INFERENCE.txt").write_text(
        "TAO DINO inference (run inside the TAO 5.5.0 Pod after unzipping here):\n\n"
        "cd /workspace/inventorylens-dino-eval\n"
        "tao model dino inference -e experiment.yaml -r results "
        "-k nvidia_tao inference.checkpoint=model_epoch_011.pth inference.conf_threshold=0.25 "
        "'dataset.infer_data_sources={image_dir:/workspace/inventorylens-dino-eval/data/images/infer,"
        "classmap:/workspace/inventorylens-dino-eval/classmap.txt}'\n\n"
        "The expected outputs are annotated images under results/images_annotated and KITTI-format "
        "prediction labels under results/labels. This run is for visual review; it does not calculate "
        "accuracy because no human-verified boxes exist for these 30 scenes.\n",
        encoding="utf-8",
    )

    with ZipFile(ARCHIVE, "w", ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUILD).as_posix())
    print(json.dumps({"archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size, **{k: manifest[k] for k in ("checkpoint_bytes", "held_out_images", "employee_images", "customer_images")}}, indent=2))


if __name__ == "__main__":
    main()
