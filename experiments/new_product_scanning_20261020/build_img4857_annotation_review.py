"""Build an offline HTML view of the unchanged IMG_4857 annotation records."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "audit" / "photo_manifest.csv"
BASELINE = HERE / "annotations" / "scene_labels_before_img4857_review_20261020.json"
OUTPUT = HERE / "results" / "img_4857_duplicate_annotation_review_20261020.html"
NAME = "IMG_4857.jpeg"


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(item for item in csv.DictReader(handle) if item["filename"] == NAME)
    photo = Path(row["path"])
    source_bytes = photo.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != row["sha256"]:
        raise RuntimeError("Original IMG_4857 source hash changed")
    record = json.loads(BASELINE.read_text(encoding="utf-8"))[NAME]
    boxes = [item["box_xyxy"] for item in record["instances"]]
    if len(boxes) != 2 or boxes[0] != boxes[1]:
        raise RuntimeError("Expected the two unchanged duplicate annotation boxes")
    width, height = int(row["display_width"]), int(row["display_height"])
    x1, y1, x2, y2 = boxes[0]
    b64 = base64.b64encode(source_bytes).decode("ascii")
    name = html.escape(NAME)
    picture = f'<img src="data:image/jpeg;base64,{b64}" alt="Original {name} photograph">'
    overlay = f'''<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-label="Two exactly overlapping saved annotation boxes">
      <rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#ff493b" stroke-width="14"/>
      <rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#00dcff" stroke-width="5" stroke-dasharray="22 12"/>
      <text x="20" y="90" fill="white" stroke="#111" stroke-width="9" paint-order="stroke" font-size="47">records 1 + 2: same box</text>
    </svg>'''
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>IMG_4857 duplicate annotation review</title><style>
      body{{font:16px/1.45 system-ui,sans-serif;background:#f5f7f8;color:#17222c;max-width:1180px;margin:24px auto;padding:0 18px}}
      h1{{font-size:1.7rem}} .note{{background:#fff;border-left:5px solid #e8534a;padding:14px 18px;margin:18px 0}}
      .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}
      figure{{background:#fff;margin:0;padding:12px;border:1px solid #d5dee2;border-radius:10px}}
      .frame{{position:relative}} .frame img{{display:block;width:100%;height:auto}} .frame svg{{position:absolute;inset:0;width:100%;height:100%}}
      figcaption{{padding:10px 0 2px}} table{{border-collapse:collapse;background:white;width:100%;max-width:700px}}
      th,td{{border:1px solid #d5dee2;padding:9px;text-align:left}} th{{background:#e9f0f3}}
    </style></head><body>
    <h1>IMG_4857: two cans, two duplicate annotation boxes</h1>
    <p>Source photo: {width} x {height} EXIF-corrected pixels. SHA-256: <code>{row['sha256']}</code>. The original JPEG is embedded unchanged for offline review.</p>
    <div class="note"><strong>What the photo shows:</strong> two stacked physical cans. <strong>What the saved annotations show:</strong> two records with precisely the same coordinates [{x1}, {y1}, {x2}, {y2}]. A repeated box does not distinguish the upper and lower cans. The coordinates span about 25% of each original-image dimension, suggesting a thumbnail-scale entry; that cause is an inference, not a verified annotation history.</div>
    <div class="grid"><figure><div class="frame">{picture}</div><figcaption>Original photo: inspect the upper and lower cans separately.</figcaption></figure>
    <figure><div class="frame">{picture}{overlay}</div><figcaption>Saved records over the original: red solid and cyan dashed lines coincide exactly. They are <em>annotation duplicates</em>, even though two products are visible.</figcaption></figure></div>
    <h2>Saved records before correction</h2><table><tr><th>Record</th><th>Candidate SKU</th><th>Box in original pixels</th></tr>
    <tr><td>1</td><td>{html.escape(record['instances'][0]['sku_id'])}</td><td>{boxes[0]}</td></tr>
    <tr><td>2</td><td>{html.escape(record['instances'][1]['sku_id'])}</td><td>{boxes[1]}</td></tr></table>
    <p>Use the local Scene review page to redraw separate boxes in original-image pixels, check both identities, and save the completed review. The current annotation and this baseline copy are preserved until you approve a correction. No box on this page is accepted ground truth.</p>
    </body></html>'''
    OUTPUT.write_text(document, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
