"""Read-only OpenCV 5 barcode and ORB feasibility checks on owner photos.

Outputs stay inside this isolated experiment. This is not a scene benchmark or
an authorized SKU registry update.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def approved_events() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    with (HERE / "annotations" / "mask_review_events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            latest[event["source_file"]] = event
    return {name: event for name, event in latest.items() if event["decision"] == "approved"}


def descriptor(path: Path, orb: cv2.ORB) -> np.ndarray | None:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    scale = min(1.0, 800.0 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, features = orb.detectAndCompute(gray, None)
    return features


def main() -> None:
    if cv2.__version__.split(".")[0] != "5":
        raise RuntimeError(f"OpenCV 5 required; found {cv2.__version__}")
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        raise RuntimeError("This OpenCV 5 build lacks barcode_BarcodeDetector")

    manifest = csv_rows(HERE / "audit" / "photo_manifest.csv")
    owner = {row["filename"]: row for row in manifest if row["group"] == "owner"}
    registry = csv_rows(HERE / "audit" / "candidate_skus.csv")
    sku_by_photo = {name.strip(): row["sku_id"] for row in registry
                    for name in row["reference_files"].split(";")}
    if len(owner) != 42 or set(sku_by_photo) != set(owner):
        raise ValueError("Expected 42 owner photos mapped to ten provisional groups")

    # Decode visible retail barcodes from the source owner photos. A decoded
    # number is evidence from the pixels, not yet a verified SKU mapping.
    decoder = cv2.barcode_BarcodeDetector()
    barcode_rows = []
    for filename, row in sorted(owner.items()):
        source = Path(row["path"])
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"Source hash changed: {filename}")
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((1800, 1800))
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        try:
            _, values, kinds, _ = decoder.detectAndDecodeWithType(bgr)
        except cv2.error:
            values, kinds = (), ()
        for value, kind in zip(values, kinds):
            if value:
                barcode_rows.append({"photo": filename, "provisional_sku": sku_by_photo[filename],
                                     "barcode_type": kind, "decoded_value": value})

    # Exactly the same 34 approved owner-photo queries as the saved SigLIP2
    # box-versus-mask comparison. Query and gallery use the same crop style.
    events = approved_events()
    if len(events) != 34 or not set(events) <= set(owner):
        raise ValueError("Expected 34 approved owner masks")
    for filename, event in events.items():
        if event["source_sha256"] != owner[filename]["sha256"] or event["sku_id"] != sku_by_photo[filename]:
            raise ValueError(f"Mask/source mismatch: {filename}")

    orb = cv2.ORB_create(nfeatures=800)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    all_rows = []
    summary = {}
    for representation, field in (("box", "box_crop_file"), ("masked", "masked_crop_file")):
        vectors = {filename: descriptor(HERE / event[field].replace("\\", "/"), orb)
                   for filename, event in events.items()}
        correct = 0
        for query_name, query in sorted(vectors.items()):
            scores: dict[str, int] = defaultdict(int)
            for gallery_name, gallery in vectors.items():
                if query_name == gallery_name or query is None or gallery is None or len(gallery) < 2:
                    continue
                pairs = matcher.knnMatch(query, gallery, k=2)
                good = sum(1 for pair in pairs if len(pair) == 2
                           and pair[0].distance < 0.75 * pair[1].distance)
                sku = sku_by_photo[gallery_name]
                scores[sku] = max(scores[sku], good)
            ranked = sorted(scores, key=lambda sku: (-scores[sku], sku))
            predicted = ranked[0] if ranked else ""
            correct += predicted == sku_by_photo[query_name]
            all_rows.append({"representation": representation, "query_photo": query_name,
                             "provisional_sku": sku_by_photo[query_name], "top1_sku": predicted,
                             "top1_correct": predicted == sku_by_photo[query_name],
                             "top1_good_matches": scores.get(predicted, 0),
                             "true_sku_good_matches": scores.get(sku_by_photo[query_name], 0)})
        summary[representation] = {"correct_top1": correct, "queries": len(vectors)}

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "opencv5_barcode_owner_20261020.csv", barcode_rows,
              ["photo", "provisional_sku", "barcode_type", "decoded_value"])
    write_csv(RESULTS / "opencv5_orb_owner_20261020.csv", all_rows,
              ["representation", "query_photo", "provisional_sku", "top1_sku",
               "top1_correct", "top1_good_matches", "true_sku_good_matches"])
    result = {"opencv_version": cv2.__version__, "owner_photos": len(owner),
              "barcode_photos_with_decode": len({row["photo"] for row in barcode_rows}),
              "barcode_decodes": barcode_rows, "orb": summary,
              "orb_method": "ORB nfeatures=800; longest crop side <=800; BF Hamming 2-nearest ratio <0.75; maximum good-match count per provisional SKU; alphabetical tie break; leave query photo out of its own gallery; box and mask tested separately on the same 34 owner photos",
              "limitations": ["No employee/customer scene query or DINO boxes were used.",
                              "These provisional visual groups are not confirmed sellable SKU mappings.",
                              "The ORB setup is a minimal baseline, not a tuned or geometrically verified matcher.",
                              "Barcode absence or failed decode does not imply the product lacks a barcode."]}
    (RESULTS / "opencv5_recognition_feasibility_20261020.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"OpenCV {cv2.__version__}: barcode {result['barcode_photos_with_decode']}/{len(owner)} photos; "
          f"ORB box {summary['box']['correct_top1']}/34, masked {summary['masked']['correct_top1']}/34")


if __name__ == "__main__":
    main()
