"""Convert SKU-as-category COCO suggestions into product + per-instance SKU attributes."""

from __future__ import annotations

import argparse
import copy
import json
import zipfile
from pathlib import Path


def transform_coco(coco: dict) -> dict:
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    if not categories or any(not name for name in categories.values()):
        raise ValueError("Input COCO must have named SKU categories.")
    result = copy.deepcopy(coco)
    result["categories"] = [{"id": 1, "name": "product", "supercategory": ""}]
    for annotation in result.get("annotations", []):
        sku = categories.get(annotation.get("category_id"))
        if not sku or sku == "product":
            raise ValueError(f"Annotation {annotation.get('id')} has no SKU category.")
        annotation["category_id"] = 1
        annotation["attributes"] = {**annotation.get("attributes", {}), "sku": sku}
    return result


def read_coco(path: Path) -> dict:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            manifests = [name for name in archive.namelist()
                         if Path(name).name == "instances_default.json"]
            if len(manifests) != 1:
                raise ValueError("Input ZIP must contain exactly one instances_default.json.")
            return json.loads(archive.read(manifests[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite: {args.output}")
    coco = transform_coco(read_coco(args.input))
    if len(coco.get("categories", [])) != 1 or coco["categories"][0]["name"] != "product":
        raise RuntimeError("Transformed COCO class validation failed.")
    if any(annotation.get("category_id") != 1 or not annotation.get("attributes", {}).get("sku")
           for annotation in coco.get("annotations", [])):
        raise RuntimeError("A transformed annotation lacks its SKU attribute.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(coco, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("annotations/instances_default.json", payload)
    print(f"Wrote {len(coco['annotations'])} SKU-tagged instances: {args.output}")


if __name__ == "__main__":
    main()
