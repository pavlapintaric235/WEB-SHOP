"""Score confirmed human SKU decisions against the saved provisional ranks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "experiments/new_product_scanning_20261020/results/reviewed_item_sku_ranking_20261023T075154Z"
DEFAULT_DECISIONS = ROOT / "experiments/new_product_scanning_20261020/annotations/inventorylens_sku_decisions_20261024.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=DEFAULT_RUN)
    p.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    report_path = args.run / "report.json"
    decision_path = args.decisions
    out = args.out or args.run / "sku_decision_evaluation_20261024"
    out.mkdir(parents=True, exist_ok=True)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    exported = json.loads(decision_path.read_text(encoding="utf-8"))
    expected = {
        f"{photo['filename']}#{item['item_id']}": (photo, item)
        for photo in report["photos"] for item in photo["items"]
    }
    decisions = exported.get("decisions", [])
    decision_keys = [row.get("item_key") for row in decisions]
    duplicates = sorted(k for k, count in Counter(decision_keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate decision item keys: {duplicates[:10]}")
    unexpected = sorted(set(decision_keys) - set(expected))
    if unexpected:
        raise ValueError(f"Decision file contains unknown item keys: {unexpected[:10]}")

    missing = sorted(set(expected) - set(decision_keys))
    states = Counter()
    rows = []
    by_sku = defaultdict(lambda: {"n": 0, "top1_correct": 0, "top3_correct": 0})
    for decision in decisions:
        key = decision["item_key"]
        state = decision.get("state", "missing_state")
        sku = decision.get("sku_id", "")
        states[state] += 1
        if sku and sku != "__unknown__" and sku not in report["gallery"]:
            raise ValueError(f"Unknown candidate SKU {sku} at {key}")
        photo, item = expected[key]
        ranked = item["rankings"]["combined"]
        ranked_ids = [x["sku_id"] for x in ranked]
        is_scorable = state == "confirmed" and sku in report["gallery"]
        top1 = bool(is_scorable and sku == ranked_ids[0])
        top3 = bool(is_scorable and sku in ranked_ids[:3])
        if is_scorable:
            by_sku[sku]["n"] += 1
            by_sku[sku]["top1_correct"] += int(top1)
            by_sku[sku]["top3_correct"] += int(top3)
        rows.append({
            "item_key": key, "photo_filename": photo["filename"], "group": photo["group"],
            "item_id": item["item_id"], "human_sku_id": sku,
            "review_state": state, "scored": is_scorable,
            "top1_sku_id": ranked_ids[0], "top1_similarity": ranked[0]["similarity"],
            "top1_correct": top1, "human_sku_in_top3": top3,
        })
    scored = [r for r in rows if r["scored"]]
    n = len(scored)
    top1_correct = sum(r["top1_correct"] for r in scored)
    top3_correct = sum(r["human_sku_in_top3"] for r in scored)
    report_out = {
        "status": "PROVISIONAL_CONFIRMED_LABEL_SUBSET_EVALUATION",
        "decision_export_created_at": exported.get("created_at"),
        "source_decisions": str(decision_path.relative_to(ROOT)) if decision_path.is_relative_to(ROOT) else decision_path.name,
        "source_decisions_sha256": sha256(decision_path),
        "source_rank_report": str(report_path.relative_to(ROOT)),
        "source_rank_report_sha256": sha256(report_path),
        "candidate_product_identities_confirmed": sum(x.get("identity_confirmed", False) for x in report["gallery"].values()),
        "decision_coverage": {
            "expected_items": len(expected), "exported_rows": len(decisions),
            "unique_exported_items": len(set(decision_keys)), "missing_items": missing,
            "review_states": dict(states), "confirmed_rows_with_candidate_id": n,
            "rows_not_scored": len(decisions) - n,
        },
        "metrics": {
            "top1_correct": top1_correct, "top1_accuracy": top1_correct / n if n else None,
            "top3_correct": top3_correct, "top3_accuracy": top3_correct / n if n else None,
        },
        "per_candidate": {
            sku: {**v, "top1_accuracy": v["top1_correct"] / v["n"] if v["n"] else None,
                  "top3_accuracy": v["top3_correct"] / v["n"] if v["n"] else None}
            for sku, v in sorted(by_sku.items())
        },
        "limitations": [
            "This scores only human decisions marked confirmed with a selected candidate ID.",
            "The ten candidate IDs still map to unconfirmed product/SKU identities; results are candidate-group agreement, not verified sellable-SKU accuracy.",
            "The human decision subset is not a blinded test set because reviewers saw model suggestions.",
            "This evaluates manually cropped item boxes and does not measure automatic DINO detection or end-to-end inventory counting.",
            "No confirmed unknown/not-listed decisions were present, so unknown rejection cannot be evaluated.",
        ],
    }
    (out / "report.json").write_text(json.dumps(report_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "item_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(out), "metrics": report_out["metrics"],
                      "decision_coverage": report_out["decision_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
