"""Leakage-safe owner-photo-only SigLIP2 retrieval ablation."""

from __future__ import annotations

import csv
import itertools
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from experiment_io import EXPERIMENT, RESULTS, load_manifest, load_registry, load_reviewed_mask, load_verified_image, sha256_file, siglip_snapshot


def _features(refs, representation, encoder):
    images, valid_rows = [], []
    for row in refs:
        image = load_verified_image(row)
        if representation == "photo":
            crop = image
        else:
            mask, event = load_reviewed_mask(row["filename"])
            if mask is None or not event or event.get("source_sha256") != row["sha256"]:
                continue
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            pad = max(2, round(max(x2 - x1, y2 - y1) * 0.08))
            box = (max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad))
            crop = image.crop(box)
            if representation == "masked":
                pixels = np.asarray(image).copy()
                pixels[~mask] = (124, 124, 124)
                crop = Image.fromarray(pixels).crop(box)
        images.append(crop)
        valid_rows.append(row)
    if not images:
        return {}, {}
    vectors = encoder.encode(images, batch_size=1)
    return ({row["filename"]: vector for row, vector in zip(valid_rows, vectors)},
            {row["filename"]: row for row in valid_rows})


def _sizes(reference_count):
    max_gallery = reference_count - 1
    if max_gallery < 1:
        return []
    sizes = [(str(n), n) for n in range(1, min(4, max_gallery) + 1)]
    sizes.append(("all", max_gallery))
    return list(dict.fromkeys(sizes))


def _rank(query, gallery):
    scored = [(sku, max(float(np.dot(query, vector)) for vector in vectors))
              for sku, vectors in gallery.items() if vectors]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def main():
    root = EXPERIMENT.parents[1]
    os.environ.setdefault("WEBSHOP_HF_CACHE", str(root / ".venv" / "huggingface_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(root))
    from experiments.hybrid_pipeline.hf_encoders import HuggingFaceImageEncoder, SIGLIP2_REVISION

    registry = load_registry()
    by_name = {row["filename"]: row for row in load_manifest() if row["group"] == "owner"}
    per_sku = {item["sku_id"]: [by_name[name] for name in item["reference_files"] if name in by_name]
               for item in registry}
    per_sku = {sku: refs for sku, refs in per_sku.items() if len(refs) >= 2}
    if not per_sku:
        raise RuntimeError("Need at least two owner references per candidate SKU.")

    snapshot = siglip_snapshot()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pinned local SigLIP2 snapshot is incomplete: {snapshot}")
    encoder = HuggingFaceImageEncoder(str(snapshot), "siglip2", device="cpu",
                                      local_files_only=True, revision=SIGLIP2_REVISION)
    features, valid_by_rep = {}, {}
    for representation in ("photo", "box", "masked"):
        features[representation], valid_by_rep[representation] = {}, {}
        for refs in per_sku.values():
            vectors, rows = _features(refs, representation, encoder)
            features[representation].update(vectors)
            valid_by_rep[representation].update(rows)

    trials = []
    confirmed_by_sku = {item["sku_id"]: bool(item.get("identity_confirmed", False)) for item in registry}

    def record_trial(representation, cohort, sku, query, chosen, count, subset_index, vectors, rivals, size_label=None):
        gallery = {other: [vectors[row["filename"]] for row in rows]
                   for other, rows in rivals.items()}
        gallery[sku] = [vectors[row["filename"]] for row in chosen]
        ranking = _rank(vectors[query["filename"]], gallery)
        ranked = [candidate for candidate, _ in ranking]
        trials.append({
            "representation": representation, "query_cohort": cohort,
            "query_file": query["filename"], "true_sku_id": sku,
            "gallery_size_label": size_label or str(count), "target_gallery_size": count,
            "target_gallery_files": ";".join(row["filename"] for row in chosen),
            "rival_gallery_policy": "all available references",
            "top1_sku_id": ranked[0] if ranked else "",
            "top1_similarity": ranking[0][1] if ranking else None,
            "top5_skus": ";".join(ranked[:5]),
            "candidate_sku_count": len(ranked),
            "top1_correct": bool(ranked and ranked[0] == sku),
            "top5_correct": sku in ranked[:5],
            "identity_confirmed": confirmed_by_sku[sku], "subset_index": subset_index,
        })

    for representation in ("photo", "box", "masked"):
        vectors = features[representation]
        available_by_sku = {
            sku: [row for row in refs if row["filename"] in vectors]
            for sku, refs in per_sku.items()
        }
        available_by_sku = {sku: refs for sku, refs in available_by_sku.items() if len(refs) >= 2}
        if len(available_by_sku) < 2:
            continue

        # Keep a single query cohort for the 1-4 reference curve within each
        # representation. Only SKUs with >=5 photos can provide four references
        # after withholding any one query photo.
        fixed_skus = {sku: refs for sku, refs in available_by_sku.items() if len(refs) >= 5}
        for sku, sku_refs in fixed_skus.items():
            rivals = {other: refs for other, refs in available_by_sku.items() if other != sku}
            for query in sku_refs:
                target_pool = [row for row in sku_refs if row["filename"] != query["filename"]]
                for count in range(1, 5):
                    for subset_index, chosen in enumerate(itertools.combinations(target_pool, count)):
                        record_trial(representation, "fixed_queries_1_to_4", sku, query,
                                     chosen, count, subset_index, vectors, rivals)

        # Separate all-available evaluation across every mapped class. The query
        # is always withheld; these rows are not compared as a fixed-query curve.
        for sku, sku_refs in available_by_sku.items():
            rivals = {other: refs for other, refs in available_by_sku.items() if other != sku}
            for query in sku_refs:
                target_pool = [row for row in sku_refs if row["filename"] != query["filename"]]
                record_trial(representation, "all_available_per_sku", sku, query,
                             target_pool, len(target_pool), 0, vectors, rivals, size_label="all")

    # Paired representation ablation: require the exact same approved-mask query
    # photos for photo, box-crop, and masked-crop features, with enough refs for
    # the same gallery-size sweep in every included SKU.
    paired_files = set.intersection(*(set(features[name]) for name in ("photo", "box", "masked")))
    paired_by_sku = {
        sku: [row for row in refs if row["filename"] in paired_files]
        for sku, refs in per_sku.items()
    }
    paired_by_sku = {sku: refs for sku, refs in paired_by_sku.items() if len(refs) >= 2}
    if len(paired_by_sku) >= 2:
        max_paired_gallery = min(4, min(len(refs) - 1 for refs in paired_by_sku.values()))
        for representation in ("photo", "box", "masked"):
            vectors = features[representation]
            for sku, sku_refs in paired_by_sku.items():
                rivals = {other: refs for other, refs in paired_by_sku.items() if other != sku}
                for query in sku_refs:
                    target_pool = [row for row in sku_refs if row["filename"] != query["filename"]]
                    for count in range(1, max_paired_gallery + 1):
                        for subset_index, chosen in enumerate(itertools.combinations(target_pool, count)):
                            record_trial(representation, "paired_mask_ablation", sku, query,
                                         chosen, count, subset_index, vectors, rivals)
                    record_trial(representation, "paired_mask_ablation", sku, query,
                                 target_pool, len(target_pool), 0, vectors, rivals, size_label="all")
    if not trials:
        raise RuntimeError("No valid query/gallery combinations. Check photo mappings and approved masks.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS / f"siglip_reference_retrieval_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    grouped = {}
    for row in trials:
        grouped.setdefault((row["query_cohort"], row["representation"], row["gallery_size_label"]), []).append(row)
    summary = [{
        "query_cohort": cohort, "representation": representation, "gallery_size": size,
        "query_gallery_trials": len(rows), "unique_queries": len({row["query_file"] for row in rows}),
        "true_sku_classes": len({row["true_sku_id"] for row in rows}),
        "candidate_sku_count_min": min(row["candidate_sku_count"] for row in rows),
        "candidate_sku_count_max": max(row["candidate_sku_count"] for row in rows),
        "top1_accuracy_micro": sum(row["top1_correct"] for row in rows) / len(rows),
        "top5_accuracy_micro": sum(row["top5_correct"] for row in rows) / len(rows),
        "top1_accuracy_macro_query": sum(
            sum(item["top1_correct"] for item in rows if item["query_file"] == query) /
            sum(item["query_file"] == query for item in rows)
            for query in {item["query_file"] for item in rows}
        ) / len({item["query_file"] for item in rows}),
        "top5_accuracy_macro_query": sum(
            sum(item["top5_correct"] for item in rows if item["query_file"] == query) /
            sum(item["query_file"] == query for item in rows)
            for query in {item["query_file"] for item in rows}
        ) / len({item["query_file"] for item in rows}),
        "identity_confirmed_trial_fraction": sum(row["identity_confirmed"] for row in rows) / len(rows),
    } for (cohort, representation, size), rows in sorted(grouped.items())]
    report = {
        "status": "exploratory_owner_photo_retrieval",
        "model": "google/siglip2-base-patch16-224",
        "evaluation": "leave-one-owner-photo-out; query excluded from its SKU gallery; no scene photos or threshold tuning",
        "rival_gallery_policy": "Rival SKUs retain all available owner references; target SKU gallery size is varied to isolate its reference-count effect.",
        "mask_coverage": {name: len(valid_by_rep[name]) for name in features},
        "fixed_query_cohort_skus_by_representation": {
            name: sorted(sku for sku, refs in per_sku.items()
                         if len([row for row in refs if row["filename"] in features[name]]) >= 5)
            for name in features
        },
        "fixed_query_files_by_representation": {
            name: sorted(row["filename"] for sku, refs in per_sku.items()
                         if len([item for item in refs if item["filename"] in features[name]]) >= 5
                         for row in refs if row["filename"] in features[name])
            for name in features
        },
        "paired_mask_ablation_query_count": len(paired_files),
        "identity_confirmed": {item["sku_id"]: bool(item.get("identity_confirmed", False)) for item in registry},
        "limitations": [
            "Controlled owner-reference retrieval does not establish employee/customer scene accuracy.",
            "SKU groupings remain provisional until owner-confirmed.",
            "Unknown-SKU rejection cannot be measured from known-SKU owner references.",
            "Rival classes have their full galleries; this is a target-reference-count ablation, not an equal total-gallery-budget comparison.",
            "The fixed 1-4 curve uses only SKUs with at least five reference photos; the all-available evaluation has a broader, different query cohort and must not be compared as a paired curve.",
            "No confidence-based abstention thresholds are evaluated in this retrieval report.",
        ],
        "summary": summary,
        "per_query_csv": "per_query.csv",
        "source_manifest_sha256": sha256_file(EXPERIMENT / "audit" / "photo_manifest.csv"),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"output": str(run_dir), "mask_coverage": report["mask_coverage"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

