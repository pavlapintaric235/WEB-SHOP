"""Build a read-only review page from the frozen owner-reference retrieval run."""

from __future__ import annotations

import csv
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


HERE = Path(__file__).resolve().parent
RUN = HERE / "results" / "siglip_reference_retrieval_20261019T233831Z"
OUTPUT = HERE / "results" / "latest_recognition_review_20261020.html"
REPRESENTATIONS = ("photo", "box", "masked")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def label(value: str) -> str:
    return html.escape(value, quote=True)


def relative_asset(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    relative = Path(os.path.relpath(path, OUTPUT.parent)).as_posix()
    return quote(relative, safe="/.-_")


def artifact_path(value: str) -> Path:
    return HERE / value.replace("\\", "/")


def main() -> None:
    skus = read_csv(HERE / "audit" / "candidate_skus.csv")
    sku_by_id = {row["sku_id"]: row for row in skus}
    if len(skus) != 10:
        raise ValueError("Expected ten provisional candidate groups")

    with (RUN / "report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report["paired_mask_ablation_query_count"] != 34 or any(report["identity_confirmed"].values()):
        raise ValueError("Frozen run no longer matches the documented cohort")

    trials = read_csv(RUN / "per_query.csv")
    paired: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in trials:
        if row["query_cohort"] == "paired_mask_ablation" and row["gallery_size_label"] == "all":
            photo = row["query_file"]
            rep = row["representation"]
            if rep in paired[photo]:
                raise ValueError(f"Duplicate paired result: {photo} / {rep}")
            paired[photo][rep] = row
    if len(paired) != 34 or any(set(rows) != set(REPRESENTATIONS) for rows in paired.values()):
        raise ValueError("Paired result must have photo, box, and mask for 34 unique owner photos")

    approved: dict[str, dict] = {}
    with (HERE / "annotations" / "mask_review_events.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            approved[event["source_file"]] = event
    approved = {name: event for name, event in approved.items() if event["decision"] == "approved"}
    if set(approved) != set(paired):
        raise ValueError("Approved mask photos differ from frozen paired queries")

    by_sku: dict[str, list[str]] = defaultdict(list)
    changed, worsened = [], []
    for photo, rows in paired.items():
        true_sku = rows["box"]["true_sku_id"]
        if any(row["true_sku_id"] != true_sku for row in rows.values()):
            raise ValueError(f"SKU disagreement for {photo}")
        by_sku[true_sku].append(photo)
        box_ok = rows["box"]["top1_correct"] == "True"
        mask_ok = rows["masked"]["top1_correct"] == "True"
        if box_ok != mask_ok:
            (changed if mask_ok else worsened).append(photo)
    if changed != ["IMG_4832.jpeg"] or worsened:
        raise ValueError("Frozen rank-change result differs from documented observation")

    correct = {rep: sum(rows[rep]["top1_correct"] == "True" for rows in paired.values())
               for rep in REPRESENTATIONS}
    if correct != {"photo": 25, "box": 24, "masked": 25}:
        raise ValueError(f"Unexpected paired top-1 counts: {correct}")

    def sku_name(sku_id: str) -> str:
        return sku_by_id[sku_id]["proposed_name"] if sku_id in sku_by_id else sku_id

    def result_text(row: dict[str, str]) -> str:
        sku = row["top1_sku_id"]
        return f"{label(sku)} · {label(sku_name(sku))}"

    def crop_pair(photo: str, size_class: str = "") -> str:
        event = approved[photo]
        box = relative_asset(artifact_path(event["box_crop_file"]))
        masked = relative_asset(artifact_path(event["masked_crop_file"]))
        return (
            f'<div class="crop-pair {size_class}">'
            f'<figure><img loading="lazy" src="{box}" alt="Unmasked box crop for {label(photo)}">'
            '<figcaption>Box crop</figcaption></figure>'
            f'<figure><img loading="lazy" src="{masked}" alt="Masked crop for {label(photo)}">'
            '<figcaption>Masked crop</figcaption></figure></div>'
        )

    cards = []
    for sku in skus:
        sku_id = sku["sku_id"]
        photos = sorted(by_sku[sku_id])
        representative = "IMG_4832.jpeg" if sku_id == "IL-NEW-005" else photos[0]
        box_hits = sum(paired[p]["box"]["top1_correct"] == "True" for p in photos)
        masked_hits = sum(paired[p]["masked"]["top1_correct"] == "True" for p in photos)
        cards.append(
            '<article class="sku-card">'
            f'<h3>{label(sku_id)} · {label(sku["proposed_name"])}</h3>'
            f'<p class="variant">{label(sku["variant_or_size"])}</p>'
            f'<p class="score">Box {box_hits}/{len(photos)} → mask {masked_hits}/{len(photos)}</p>'
            f'{crop_pair(representative)}'
            f'<p class="caption">Example reference: {label(representative)}</p>'
            '</article>'
        )

    rows_html = []
    for photo in sorted(paired):
        rows = paired[photo]
        true_sku = rows["box"]["true_sku_id"]
        box_ok = rows["box"]["top1_correct"] == "True"
        mask_ok = rows["masked"]["top1_correct"] == "True"
        outcome = "Fixed by mask" if not box_ok and mask_ok else "Correct in both" if box_ok else "Wrong in both"
        tone = "fixed" if outcome == "Fixed by mask" else "wrong" if outcome == "Wrong in both" else "correct"
        rows_html.append(
            f'<tr class="{tone}"><td><strong>{label(photo)}</strong><br><small>{label(true_sku)} · {label(sku_name(true_sku))}</small></td>'
            f'<td>{crop_pair(photo, "table-crops")}</td>'
            f'<td>{result_text(rows["box"])}</td><td>{result_text(rows["masked"])}</td>'
            f'<td><span class="badge {tone}">{outcome}</span></td></tr>'
        )

    employee = relative_asset(HERE / "audit" / "contact_employee.jpg")
    customer = relative_asset(HERE / "audit" / "contact_customer.jpg")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Newest product recognition review · InventoryLens / WEB&amp;SHOP</title>
<style>
:root{{font-family:system-ui,Segoe UI,sans-serif;color:#18212c;background:#f4f6f8}}*{{box-sizing:border-box}}
body{{margin:0}}header{{background:#143a48;color:white;padding:2.5rem max(1.2rem,calc((100vw - 1200px)/2))}}
main{{max-width:1200px;margin:auto;padding:1.5rem}}h1{{margin:.3rem 0;font-size:2rem}}h2{{margin-top:2.5rem}}
.eyebrow{{text-transform:uppercase;letter-spacing:.09em;font-size:.8rem;color:#a9dbe3}}
.lede{{max-width:780px;line-height:1.5;color:#e3f2f4}}.note{{background:#eaf2fa;border-left:4px solid #4383b5;padding:1rem;margin:1.2rem 0;line-height:1.5}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.75rem;margin:1.4rem 0}}.metric,.sku-card,.scene-card{{background:white;border:1px solid #d9e2e6;border-radius:12px;padding:1rem}}
.metric b{{font-size:1.7rem;display:block}}.metric span{{color:#52616c}}.sku-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem}}
.sku-card h3{{margin:.1rem 0}}.variant,.caption{{font-size:.85rem;color:#5f6870}}.score{{font-weight:700;color:#175d69}}
.crop-pair{{display:grid;grid-template-columns:1fr 1fr;gap:.4rem}}figure{{margin:0}}.crop-pair img{{width:100%;height:180px;object-fit:contain;background:#edf0f0;border-radius:6px}}
figcaption{{font-size:.76rem;color:#68747d;text-align:center}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border-bottom:1px solid #dce4e8;text-align:left;padding:.65rem;vertical-align:middle}}th{{background:#e8eff2;position:sticky;top:0}}td small{{color:#66737b}}
tr.fixed{{background:#eaf7ed}}tr.wrong{{background:#fff3ef}}.table-crops{{min-width:170px;max-width:210px}}.table-crops img{{height:96px}}.badge{{display:inline-block;border-radius:20px;padding:.25rem .6rem;font-size:.8rem;white-space:nowrap}}.badge.fixed{{background:#bce9c6}}.badge.wrong{{background:#f8d4c8}}.badge.correct{{background:#e2e9ec}}
.scene-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:1rem}}.scene-card img{{width:100%;height:auto;border:1px solid #ccd7dc}}a{{color:#135e82}}footer{{padding:2rem 0;color:#60717b;font-size:.9rem}}
</style></head><body>
<header><div class="eyebrow">WEB&amp;SHOP · InventoryLens evidence · 20 Oct 2026</div><h1>Newest product recognition review</h1>
<p class="lede">See the ten provisional product groups and exactly what masking changed in the saved SigLIP2 owner-reference test. The newest employee and customer scene photos are shown below as a dataset preview; their DINO/SigLIP2 scene predictions have not been imported.</p></header>
<main>
<div class="metrics"><div class="metric"><b>34</b><span>paired owner-reference queries</span></div><div class="metric"><b>24 / 34</b><span>box-crop top-1 correct</span></div><div class="metric"><b>25 / 34</b><span>masked-crop top-1 correct</span></div><div class="metric"><b>0 / 30</b><span>new scene predictions imported</span></div></div>
<div class="note"><strong>What these numbers mean.</strong> Each of 34 approved owner reference photos is queried against the other available photos, with the query excluded from its own SKU gallery. The photo, box, and masked versions use the same 34 queries; all ten SKU identities remain provisional. This tests controlled reference retrieval, not detection, employee/customer recognition, or SKU accuracy in a live scene. Top-5 was 33/34 for both box and mask.</div>
<h2>What masking changed</h2><p><strong>IMG_4832.jpeg</strong> (provisional Dobro Mogador chocolate oatmeal) changed from a wrong top choice of rice mash with a box crop to the correct oatmeal group with a masked crop. No paired query became wrong because of the mask. The other 33 had the same correct/wrong outcome. In this owner-photo test, <em>both the query and gallery references</em> use the selected representation. The mask replaces background pixels with neutral gray before SigLIP2 encodes the crop; DINO scene detection does not use this reference mask.</p>
{crop_pair("IMG_4832.jpeg", "example-crops")}
<h2>Ten provisional product groups</h2><div class="sku-grid">{''.join(cards)}</div>
<h2>All 34 paired owner-photo results</h2><p>Green marks the one corrected top choice; peach marks queries still wrong in both versions. These are per-photo retrieval results, not scene detections.</p>
<div class="table-wrap"><table><thead><tr><th>Owner query / proposed SKU</th><th>Same reference, two views</th><th>Top-1 from box</th><th>Top-1 from mask</th><th>Outcome</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>
<h2>Newest employee and customer photos</h2><div class="note"><strong>Scene inference pending.</strong> The prepared input bundle contains 25 employee scenes and 5 customer scenes. No new DINO detection boxes or SigLIP2 scene rankings were found locally. The current bundle uses 34 approved <em>box</em> reference crops, so it cannot measure the mask's effect on later scene recognition until a matched masked-gallery pass is run. Most scenes also need one verified label box per physical item before accuracy can be scored.</div>
<div class="scene-grid"><section class="scene-card"><h3>25 employee photos</h3><img loading="lazy" src="{employee}" alt="Contact sheet of employee scene photos"></section><section class="scene-card"><h3>5 customer photos</h3><img loading="lazy" src="{customer}" alt="Contact sheet of customer scene photos"></section></div>
<h2>Traceability and next step</h2><p>Frozen run: <a href="{relative_asset(RUN / 'summary.csv')}">summary.csv</a>, <a href="{relative_asset(RUN / 'per_query.csv')}">per_query.csv</a>, and <a href="{relative_asset(RUN / 'report.json')}">report.json</a>. Proposed identities: <a href="{relative_asset(HERE / 'audit' / 'candidate_skus.csv')}">candidate_skus.csv</a>. The source-photo hashes remain in <a href="{relative_asset(HERE / 'audit' / 'photo_manifest.csv')}">photo_manifest.csv</a>.</p>
<p>First confirm the ten SKU identities and correct the scene labels. Then run the already-prepared GPU bundle to obtain actual DINO boxes and SigLIP2 rankings for the 30 scenes. Review misses and merged boxes before changing detector settings; evaluate any change on a separate held-out scene set. A second matched run with masked reference crops can test whether masking helps scene recognition.</p>
<footer>Generated from saved experiment outputs. No source photos, production inventory, AWS resources, or model predictions were changed.</footer>
</main></body></html>'''
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(paired)} paired queries, {len(skus)} provisional groups)")


if __name__ == "__main__":
    main()
