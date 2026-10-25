"""Package only the reviewed TAO DINO train/validation split for remote training."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DATA = EXPERIMENT / "training_data"
COCO = DATA / "tao_coco"
BUILD = HERE / ".build_zip"
PACKAGE_ROOT = "inventorylens-dino"


def main() -> None:
    expected = {"train": (25, 144), "val": (5, 33)}
    for split, (n_images, n_boxes) in expected.items():
        image_dir = DATA / "images" / split
        coco_path = COCO / "annotations" / f"instances_{split}.json"
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if (len(images), len(coco["annotations"])) != (n_images, n_boxes):
            raise ValueError(f"{split} changed: images={len(images)}, boxes={len(coco['annotations'])}")
        if any(category != {"id": 1, "name": "product", "supercategory": "product"} for category in coco["categories"]):
            raise ValueError(f"Unexpected categories in {coco_path}")
        for record in coco["images"]:
            path = image_dir / record["file_name"]
            if not path.is_file():
                raise ValueError(f"Missing COCO image: {path}")
            with Image.open(path) as image:
                if image.size != (record["width"], record["height"]):
                    raise ValueError(f"Image dimensions changed: {path}")

    if BUILD.exists():
        import shutil

        shutil.rmtree(BUILD)
    package = BUILD / PACKAGE_ROOT
    for split in expected:
        target = package / "data" / "images" / split
        target.mkdir(parents=True, exist_ok=True)
        for source in (DATA / "images" / split).iterdir():
            if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                (target / source.name).write_bytes(source.read_bytes())
        annotations = package / "data" / "annotations"
        annotations.mkdir(parents=True, exist_ok=True)
        (annotations / f"instances_{split}.json").write_bytes(
            (COCO / "annotations" / f"instances_{split}.json").read_bytes()
        )
    (package / "train.yaml").write_bytes((HERE / "train.yaml").read_bytes())

    archive = HERE / "inventorylens_tao_dino_20261021.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zf:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD).as_posix())
    print(json.dumps({"status": "package_ready_no_upload", "archive": str(archive), "splits": expected}, indent=2))


if __name__ == "__main__":
    main()
