from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts" / "third_set_review_20261020"


def main() -> None:
    data = json.loads((OUT / "predictions.json").read_text(encoding="utf-8"))
    index = {}
    for model in ("baseline", "finetuned"):
        for item in data["models"][model]["detections"]:
            index.setdefault(Path(item["image"]).name, {})[model] = item
    rows = []
    for scene in data["scenes"]:
        filename = Path(scene["local_path"]).name
        prediction_name = f"{Path(filename).stem}.jpg"
        group = "Employee" if scene["source_group"] == "new10_employee" else "Customer"
        cells = []
        for title, src in (
            ("Original", scene["local_path"]),
            ("Pretrained baseline", f"baseline/predictions/{prediction_name}"),
            ("Fine-tuned", f"finetuned/predictions/{prediction_name}"),
        ):
            meta = ""
            key = {"Pretrained baseline": "baseline", "Fine-tuned": "finetuned"}.get(title)
            if key:
                detection = index[filename][key]
                meta = f'<div class="count">{detection["count"]} boxes</div>'
            cells.append(f'<figure><figcaption>{title}{meta}</figcaption><a href="{html.escape(src)}"><img loading="lazy" src="{html.escape(src)}"></a></figure>')
        rows.append(f'<section><h2>{group} · {html.escape(filename)}</h2><div class="trio">{"".join(cells)}</div></section>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Held-out product detector review</title><style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#f4f5f7;color:#18212d}}header,section{{max-width:1500px;margin:auto}}header{{margin-bottom:24px}}.note{{color:#4d5a68;line-height:1.5}}section{{background:white;border:1px solid #dce1e7;border-radius:12px;padding:14px;margin:16px auto}}h2{{font-size:17px;margin:4px 0 12px}}.trio{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}figure{{margin:0;min-width:0}}figcaption{{font-weight:650;padding:8px 0}}.count{{display:inline-block;font-weight:500;color:#586577;margin-left:8px}}img{{display:block;width:100%;height:auto;border-radius:6px;background:#e9edf2}}@media(max-width:800px){{.trio{{grid-template-columns:1fr}}}}
</style><header><h1>Held-out product detector review</h1><p class="note">30 scenes · 25 employee, 5 customer. Each row shows the source and CPU predictions from the pretrained and fine-tuned checkpoints at 640 px, confidence 0.25, IoU 0.7. Counts are candidate boxes, not verified item counts. Both models use a generic product class; detections do not identify SKUs. No scene ground truth has been reviewed, so this gallery makes no accuracy claim.</p></header>{''.join(rows)}</html>'''
    (OUT / "review_gallery.html").write_text(page, encoding="utf-8")
    print(f"Gallery written: {OUT / 'review_gallery.html'}")


if __name__ == "__main__":
    main()
