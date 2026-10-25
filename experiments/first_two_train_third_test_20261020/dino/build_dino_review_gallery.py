"""Build a compact interactive review report for the TAO DINO held-out run."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts" / "third_set_review_20261020"
INFERENCE = ARTIFACTS / "dino_inference" / "results"
PREDICTIONS = ARTIFACTS / "predictions.json"
OVERLAYS = ARTIFACTS / "dino_inference" / "overlays"
OUTPUT_HTML = ARTIFACTS / "dino_review_gallery.html"


def render_boxes(scenes: list[dict]) -> list[dict]:
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    report_scenes = []
    for scene in scenes:
        image_name = Path(scene["original_filename"]).name
        stem = Path(image_name).stem
        source_path = ARTIFACTS / scene["local_path"]
        labels_path = INFERENCE / "labels" / f"{stem}.txt"
        if not source_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"Missing source or DINO labels for {image_name}")

        boxes = []
        with Image.open(source_path) as opened:
            canvas = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        line_width = max(3, round(min(canvas.size) / 350))
        try:
            font = ImageFont.truetype("arial.ttf", max(18, round(min(canvas.size) / 45)))
        except OSError:
            font = ImageFont.load_default()

        for line in labels_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 16:
                continue
            x1, y1, x2, y2 = (float(value) for value in fields[4:8])
            score = float(fields[-1])
            box_id = len(boxes) + 1
            # TAO's label coordinates already use the upright dimensions (confirmed
            # against its images_annotated outputs), so only transpose the source pixels.
            x1, x2 = sorted((max(0, min(canvas.width - 1, round(x1))), max(0, min(canvas.width - 1, round(x2)))))
            y1, y2 = sorted((max(0, min(canvas.height - 1, round(y1))), max(0, min(canvas.height - 1, round(y2)))))
            draw.rectangle((x1, y1, x2, y2), outline="#e23b35", width=line_width)
            caption = f"#{box_id}  {score:.2f}"
            bounds = draw.textbbox((x1, y1), caption, font=font)
            text_height = bounds[3] - bounds[1]
            label_y = max(0, y1 - text_height - line_width * 2)
            draw.rectangle((x1, label_y, bounds[2] + line_width, label_y + text_height + line_width), fill="#e23b35")
            draw.text((x1 + line_width // 2, label_y), caption, fill="white", font=font)
            boxes.append({"id": box_id, "confidence": round(score, 4), "box": [x1, y1, x2, y2]})
        canvas.save(OVERLAYS / image_name, quality=92)

        report_scenes.append({
            "group": "Employee" if scene["source_group"] == "new10_employee" else "Customer",
            "filename": image_name,
            "source": scene["local_path"],
            "overlay": f"dino_inference/overlays/{image_name}",
            "boxes": boxes,
        })
    return report_scenes


def main() -> None:
    metadata = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    scenes = metadata["scenes"]
    if len(scenes) != 30:
        raise ValueError(f"Expected 30 scenes, got {len(scenes)}")
    report_scenes = render_boxes(scenes)
    scene_json = json.dumps(report_scenes, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>InventoryLens · DINO review</title>
<style>
:root{color-scheme:light;--ink:#17242f;--muted:#5e6c78;--line:#dce3e8;--paper:#fff;--bg:#f3f6f8;--green:#19734a;--red:#b33b39;--amber:#8b5b00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:auto;padding:28px 20px 60px}h1{font-size:30px;line-height:1.15;margin:0 0 8px}h2{font-size:18px;margin:0}p{margin:8px 0}.muted{color:var(--muted)}.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;font-weight:700;color:var(--muted)}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}.top-actions{display:flex;gap:8px;flex-wrap:wrap}.button{border:1px solid #bdc9d2;background:white;color:var(--ink);border-radius:8px;padding:9px 13px;font:inherit;font-weight:650;cursor:pointer}.button.primary{background:#183e55;color:white;border-color:#183e55}.button:disabled{opacity:.5;cursor:default}
.summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:22px 0}.metric,.panel,.alert{border:1px solid var(--line);border-radius:12px;background:var(--paper)}.metric{padding:14px 16px}.metric strong{display:block;font-size:23px}.metric span{font-size:13px;color:var(--muted)}
.alert{padding:16px 18px;margin:14px 0}.alert h2{margin-bottom:5px}.alert.sku{border-left:5px solid #526f8a}.alert.warn{border-left:5px solid #d39a2f}.report-grid{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(300px,1fr);gap:16px;margin-top:18px}.panel{padding:16px}.panel-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}.scene-toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.scene-toolbar select{font:inherit;padding:9px;border:1px solid #bdc9d2;border-radius:8px;background:white;max-width:100%}.nav{display:flex;gap:6px}
.photo-wrap{position:relative;background:#e9edf1;border-radius:10px;overflow:hidden;min-height:150px}.photo-wrap img{display:block;width:100%;height:auto}.photo-tabs{display:flex;gap:6px;margin:12px 0}.photo-tabs button{flex:1}.photo-tabs button.active{border-color:#183e55;background:#e9f1f5}.detail-list{display:grid;gap:9px}.box-card{border:1px solid var(--line);border-radius:9px;padding:10px}.box-row{display:flex;justify-content:space-between;gap:10px;align-items:center}.box-id{font-weight:750}.score{font-variant-numeric:tabular-nums;color:var(--muted)}.decision{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.decision .selected{color:white;background:var(--green);border-color:var(--green)}.decision .reject.selected{background:var(--red);border-color:var(--red)}.decision .merged.selected{background:var(--amber);border-color:var(--amber)}.decision-state{font-size:12px;color:var(--muted);margin-top:5px}.empty{color:var(--muted);padding:20px 4px}.progress{color:var(--muted);font-size:13px}.method{font-size:13px;color:var(--muted)}
@media(max-width:800px){main{padding:18px 12px 40px}.top{display:block}.top-actions{margin-top:12px}.report-grid{grid-template-columns:1fr}.summary{gap:8px}.metric{padding:10px}.metric strong{font-size:19px}.panel{padding:12px}}
</style></head><body><main>
<div class="top"><div><div class="eyebrow">Experiment review · 30 held-out photos</div><h1>DINO product detection</h1><p class="muted">Review one photo at a time. The red outlines are detector boxes; box numbers match the review choices.</p></div>
<div class="top-actions"><button class="button primary" id="export">Download review report</button><button class="button" id="clear">Clear this browser’s decisions</button></div></div>
<div class="summary"><div class="metric"><strong id="scene-count">30</strong><span>photos</span></div><div class="metric"><strong id="box-count">278</strong><span>DINO boxes · confidence threshold 0.25</span></div><div class="metric"><strong id="review-count">0 / 278</strong><span>boxes reviewed</span></div></div>
<section class="alert warn"><h2>Box quality needs review</h2><p>Some boxes may cover multiple products, duplicate another box, or include background. Mark a merged box so it can be split and relabeled before another detector training run. Approving a merged box will incorrectly teach the model that several items are one object.</p></section>
<section class="alert sku"><h2>SKU recognition: not run for this DINO report</h2><p>This checkpoint predicts only a generic <strong>product</strong> class. It has no SKU labels or SKU predictions to show here. InventoryLens has a separate recognition path that segments item regions and compares them with registered product reference photos; it has not been run on these DINO detections. Merged boxes also need to be separated before a reliable SKU match can be assigned.</p><p class="method">A future report should show the approved item box, matched SKU and product name, match confidence, and an uncertain state when the reference match is weak.</p></section>
<div class="scene-toolbar"><button class="button" id="prev">← Previous</button><select id="scene-select" aria-label="Select photo"></select><button class="button" id="next">Next →</button><span class="progress" id="position"></span></div>
<div class="report-grid"><section class="panel"><div class="panel-head"><div><div class="eyebrow" id="group"></div><h2 id="filename"></h2></div><span class="progress" id="scene-box-count"></span></div><div class="photo-tabs"><button class="button active" data-view="overlay">Show boxes</button><button class="button" data-view="source">Original photo</button></div><div class="photo-wrap"><img id="photo" alt="Selected inventory photo with DINO detection boxes"></div></section>
<section class="panel"><div class="panel-head"><h2>Review detections</h2><span class="progress" id="scene-review-count"></span></div><p class="muted">Choose one decision per box. “Merged” means the box contains more than one product and needs to be split.</p><div class="detail-list" id="boxes"></div></section></div>
</main><script>
const scenes = __SCENES__;
const allBoxes = scenes.reduce((sum, scene) => sum + scene.boxes.length, 0);
const storeKey = "inventorylens-dino-review-v2";
let decisions = {};
try { decisions = JSON.parse(localStorage.getItem(storeKey) || "{}"); } catch (_) {}
let sceneIndex = 0;
let imageView = "overlay";
const select = document.getElementById("scene-select");
scenes.forEach((scene, index) => { const option = document.createElement("option"); option.value = index; option.textContent = `${String(index+1).padStart(2,"0")} · ${scene.group} · ${scene.filename} · ${scene.boxes.length} boxes`; select.append(option); });
function keyFor(scene, box){ return `${scene.filename}:${box.id}`; }
function save(){ localStorage.setItem(storeKey, JSON.stringify(decisions)); render(); }
function render(){
  const scene = scenes[sceneIndex]; select.value = sceneIndex;
  document.getElementById("group").textContent = `${scene.group} photo · ${sceneIndex+1} of ${scenes.length}`;
  document.getElementById("filename").textContent = scene.filename;
  document.getElementById("scene-box-count").textContent = `${scene.boxes.length} boxes`;
  document.getElementById("position").textContent = `${sceneIndex+1} / ${scenes.length}`;
  document.getElementById("photo").src = imageView === "overlay" ? scene.overlay : scene.source;
  document.getElementById("photo").alt = imageView === "overlay" ? "DINO predictions with numbered boxes" : "Original photo without boxes";
  document.querySelectorAll("[data-view]").forEach(button => button.classList.toggle("active", button.dataset.view === imageView));
  const list = document.getElementById("boxes"); list.replaceChildren();
  if (!scene.boxes.length) list.innerHTML = '<div class="empty">DINO returned no boxes for this photo.</div>';
  scene.boxes.forEach(box => {
    const key = keyFor(scene, box), current = decisions[key];
    const card = document.createElement("div"); card.className = "box-card";
    const row = document.createElement("div"); row.className = "box-row";
    row.innerHTML = `<span class="box-id">Box ${box.id}</span><span class="score">confidence ${box.confidence.toFixed(2)}</span>`;
    const actions = document.createElement("div"); actions.className = "decision";
    [["approve","Approve single item",""],["reject","Reject box","reject"],["merged","Merged items","merged"]].forEach(([value,label,extra]) => {
      const button = document.createElement("button"); button.className = `button ${extra} ${current === value ? "selected" : ""}`; button.textContent = label;
      button.addEventListener("click", () => { if (decisions[key] === value) delete decisions[key]; else decisions[key] = value; save(); });
      actions.append(button);
    });
    const state = document.createElement("div"); state.className = "decision-state"; state.textContent = current ? `Saved decision: ${current === "approve" ? "approved as one item" : current === "reject" ? "rejected" : "merged; split required"}` : "Not reviewed";
    card.append(row, actions, state); list.append(card);
  });
  const reviewed = Object.keys(decisions).length;
  document.getElementById("review-count").textContent = `${reviewed} / ${allBoxes}`;
  document.getElementById("scene-review-count").textContent = `${scene.boxes.filter(box => decisions[keyFor(scene,box)]).length} / ${scene.boxes.length} reviewed`;
}
select.addEventListener("change", () => { sceneIndex = Number(select.value); render(); });
document.getElementById("prev").addEventListener("click", () => { sceneIndex = (sceneIndex + scenes.length - 1) % scenes.length; render(); });
document.getElementById("next").addEventListener("click", () => { sceneIndex = (sceneIndex + 1) % scenes.length; render(); });
document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => { imageView = button.dataset.view; render(); }));
document.getElementById("clear").addEventListener("click", () => { if (confirm("Clear all saved review decisions in this browser?")) { decisions = {}; save(); } });
document.getElementById("export").addEventListener("click", () => {
  const payload = { report: "DINO held-out detection review", exported_at: new Date().toISOString(), sku_recognition: "not_run", total_scenes: scenes.length, total_boxes: allBoxes, reviewed: Object.keys(decisions).length, decisions };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"}); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "dino_review_report.json"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});
render();
</script></body></html>'''.replace("__SCENES__", scene_json)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML} ({len(scenes)} scenes, {sum(len(s['boxes']) for s in report_scenes)} boxes)")


if __name__ == "__main__":
    main()
