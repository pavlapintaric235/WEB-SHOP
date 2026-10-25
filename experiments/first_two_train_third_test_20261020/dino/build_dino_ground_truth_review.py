"""Build an isolated browser UI to annotate one box per visible physical item."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts" / "third_set_review_20261020"
PREDICTIONS = ARTIFACTS / "predictions.json"
REVIEW = HERE.parent / "annotations" / "dino_review_report_20261022.json"
OUTPUT = ARTIFACTS / "dino_ground_truth_review.html"


def scenes_data() -> list[dict]:
    source = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    user_decisions = review["decisions"]
    result = []
    for scene in source["scenes"]:
        image_path = ARTIFACTS / scene["local_path"]
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        name = Path(scene["original_filename"]).name
        labels_path = ARTIFACTS / "dino_inference" / "results" / "labels" / f"{Path(name).stem}.txt"
        dino_boxes = []
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 16:
                dino_boxes.append([int(round(float(value))) for value in fields[4:8]])
        result.append({
            "filename": name,
            "group": "Employee" if scene["source_group"] == "new10_employee" else "Customer",
            "src": scene["local_path"].replace("\\", "/"),
            "width": width,
            "height": height,
            "sha256": digest,
            "dino_boxes": dino_boxes,
            "dino_review": {key.split(":", 1)[1]: value for key, value in user_decisions.items() if key.startswith(name + ":")},
        })
    if len(result) != 30:
        raise ValueError(f"Expected 30 held-out scenes, got {len(result)}")
    return result


def main() -> None:
    data = json.dumps(scenes_data(), ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINO · product ground truth review</title>
<style>
:root{color-scheme:light;--ink:#17242f;--muted:#5e6c78;--line:#dce3e8;--paper:#fff;--bg:#f3f6f8;--blue:#245c7a;--green:#19734a;--amber:#8b5b00}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1400px;margin:auto;padding:24px 18px 50px}h1{font-size:28px;margin:0 0 4px}h2{font-size:17px;margin:0}.muted{color:var(--muted)}.top{display:flex;justify-content:space-between;gap:15px;align-items:start}.actions,.toolbar,.legend{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.btn,select{font:inherit;border:1px solid #bdc9d2;background:white;color:var(--ink);border-radius:8px;padding:9px 12px;cursor:pointer}.btn.primary{background:var(--blue);border-color:var(--blue);color:white}.btn:disabled{opacity:.5}.summary{display:flex;gap:10px;margin:16px 0}.pill{border:1px solid var(--line);border-radius:9px;background:white;padding:9px 13px}.pill b{font-size:19px;margin-right:5px}.notice{background:#fff7e6;border:1px solid #ead39d;border-radius:10px;padding:12px 15px;margin-bottom:14px}.grid{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:14px}.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:13px}.toolbar{margin-bottom:10px}.toolbar select{max-width:440px}.canvasbox{position:relative;background:#e9edf1;border-radius:8px;overflow:hidden;min-height:180px;touch-action:none}.canvasbox img{display:block;width:100%;height:auto;pointer-events:none}.canvasbox canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.legend{font-size:13px;margin:8px 0;color:var(--muted)}.dot{display:inline-block;width:13px;height:13px;background:#16834f;margin-right:4px;border-radius:2px}.red{background:#e23b35}.items{display:grid;gap:7px;margin-top:10px;max-height:62vh;overflow:auto}.item{border:1px solid var(--line);border-radius:8px;padding:8px;display:flex;justify-content:space-between;gap:5px;align-items:center;font-size:13px}.item button{padding:4px 8px;border:1px solid #bdc9d2;background:white;border-radius:6px;cursor:pointer}.item.active{border-color:var(--blue);background:#edf5f9}.small{font-size:12px;color:var(--muted)}.progress{margin-left:auto;color:var(--muted);font-size:13px}.status{display:flex;gap:6px;align-items:center;margin-top:10px}.status input{width:18px;height:18px}
@media(max-width:850px){.grid{grid-template-columns:1fr}.top{display:block}.actions{margin-top:10px}.items{max-height:280px}.summary{flex-wrap:wrap}}
</style></head><body><main>
<div class="top"><div><h1>Mark the visible products</h1><p class="muted">Draw one green box around each physical product you can see, including items the detector missed. Use the red boxes only as a reference.</p></div><div class="actions"><button class="btn primary" id="export">Download item boxes JSON</button><button class="btn" id="clear">Clear this browser’s annotations</button></div></div>
<div class="notice"><b>Review tip:</b> If an item is partly hidden, box the visible part and include it once. If you cannot tell whether a product is present, leave that photo incomplete and add a note. The earlier “merged” decisions are shown as context; they do not create these new item boxes.</div>
<div class="summary"><div class="pill"><b id="scene-count">0 / 30</b> photos marked complete</div><div class="pill"><b id="item-count">0</b> item boxes</div><span class="progress" id="position"></span></div>
<div class="toolbar"><button class="btn" id="prev">← Previous</button><select id="scene"></select><button class="btn" id="next">Next →</button><button class="btn" id="undo">Undo last box</button><span class="progress" id="review-context"></span></div>
<div class="grid"><section class="panel"><div class="canvasbox"><img id="photo" alt="Photo to annotate"><canvas id="canvas"></canvas></div><div class="legend"><span class="dot"></span> Your item boxes &nbsp; <span class="dot red"></span> Earlier DINO boxes (context only)</div><p class="small">Click and drag on the photo to draw each product box. Coordinates are saved in upright image pixels.</p></section>
<aside class="panel"><h2>Items in this photo</h2><div class="items" id="items"></div><label class="status"><input type="checkbox" id="complete"><span>All visible products reviewed</span></label><p class="small">Mark complete only after checking the full photo. “Unsure” is okay: add a note and leave the photo incomplete.</p><textarea id="note" rows="3" placeholder="Optional note about uncertain or crowded areas" style="width:100%;font:inherit;margin-top:8px;border:1px solid #bdc9d2;border-radius:7px;padding:8px"></textarea></aside></div>
</main><script>
const scenes=__DATA__;const key="inventorylens-dino-groundtruth-v1";let state={};try{state=JSON.parse(localStorage.getItem(key)||"{}")}catch(e){}let idx=0,drag=null;
const sel=document.getElementById("scene"),img=document.getElementById("photo"),canvas=document.getElementById("canvas"),ctx=canvas.getContext("2d");
scenes.forEach((s,i)=>{let o=document.createElement("option");o.value=i;o.textContent=`${String(i+1).padStart(2,"0")} · ${s.group} · ${s.filename}`;sel.append(o)});
function entry(){return state[scenes[idx].filename] ||= {boxes:[],complete:false,note:""}}
function persist(){localStorage.setItem(key,JSON.stringify(state));render()}
function fitCanvas(){const r=img.getBoundingClientRect();if(!r.width||!r.height)return;const d=window.devicePixelRatio||1;canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0);draw()}
function draw(){if(!img.naturalWidth)return;const r=canvas.getBoundingClientRect(),sx=r.width/scenes[idx].width,sy=r.height/scenes[idx].height;ctx.clearRect(0,0,r.width,r.height);const s=scenes[idx];
 s.dino_boxes.forEach((b,n)=>{ctx.strokeStyle="#e23b35";ctx.lineWidth=2;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);ctx.font="bold 14px system-ui";const t=`#${n+1}`;ctx.fillStyle="#e23b35";ctx.fillRect(b[0]*sx,Math.max(0,b[1]*sy-19),ctx.measureText(t).width+8,19);ctx.fillStyle="white";ctx.fillText(t,b[0]*sx+4,Math.max(14,b[1]*sy-4))});
 entry().boxes.forEach((b,n)=>{ctx.strokeStyle="#16834f";ctx.lineWidth=3;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);ctx.font="bold 16px system-ui";const t=`Item ${n+1}`;ctx.fillStyle="#16834f";ctx.fillRect(b[0]*sx,Math.max(0,b[1]*sy-21),ctx.measureText(t).width+10,21);ctx.fillStyle="white";ctx.fillText(t,b[0]*sx+5,Math.max(15,b[1]*sy-5))});
 if(drag){ctx.strokeStyle="#16834f";ctx.lineWidth=2;ctx.setLineDash([6,4]);ctx.strokeRect(drag.x,drag.y,drag.w,drag.h);ctx.setLineDash([])} }
function render(){const s=scenes[idx],e=entry();sel.value=idx;img.src=s.src;img.alt=`${s.filename}, ${s.group}`;document.getElementById("position").textContent=`${idx+1} / ${scenes.length}`;document.getElementById("review-context").textContent=`Earlier DINO review: ${Object.values(s.dino_review).filter(x=>x==="merged").length} merged · ${Object.values(s.dino_review).filter(x=>x==="approve").length} approved`;document.getElementById("complete").checked=e.complete;document.getElementById("note").value=e.note||"";let list=document.getElementById("items");list.replaceChildren();if(!e.boxes.length)list.innerHTML='<div class="small">No item boxes yet.</div>';e.boxes.forEach((b,n)=>{let row=document.createElement("div");row.className="item";row.innerHTML=`<span>Item ${n+1} · ${b.map(Math.round).join(", ")}</span><button>Remove</button>`;row.querySelector("button").onclick=()=>{e.boxes.splice(n,1);persist()};list.append(row)});document.getElementById("item-count").textContent=Object.values(state).reduce((a,x)=>a+x.boxes.length,0);document.getElementById("scene-count").textContent=`${Object.values(state).filter(x=>x.complete).length} / ${scenes.length}`;requestAnimationFrame(fitCanvas)}
img.onload=fitCanvas;window.addEventListener("resize",fitCanvas);sel.onchange=()=>{idx=+sel.value;render()};document.getElementById("prev").onclick=()=>{idx=(idx+scenes.length-1)%scenes.length;render()};document.getElementById("next").onclick=()=>{idx=(idx+1)%scenes.length;render()};document.getElementById("undo").onclick=()=>{entry().boxes.pop();persist()};document.getElementById("complete").onchange=e=>{entry().complete=e.target.checked;persist()};document.getElementById("note").oninput=e=>{entry().note=e.target.value;localStorage.setItem(key,JSON.stringify(state))};
function point(ev){const r=canvas.getBoundingClientRect();return{x:ev.clientX-r.left,y:ev.clientY-r.top}}
canvas.onpointerdown=e=>{const p=point(e);drag={x:p.x,y:p.y,w:0,h:0};canvas.setPointerCapture(e.pointerId);draw()};canvas.onpointermove=e=>{if(!drag)return;const p=point(e);drag.w=p.x-drag.x;drag.h=p.y-drag.y;draw()};canvas.onpointerup=e=>{if(!drag)return;const r=canvas.getBoundingClientRect(),s=scenes[idx],sx=s.width/r.width,sy=s.height/r.height,x1=Math.max(0,Math.min(drag.x,drag.x+drag.w)*sx),y1=Math.max(0,Math.min(drag.y,drag.y+drag.h)*sy),x2=Math.min(s.width,Math.max(drag.x,drag.x+drag.w)*sx),y2=Math.min(s.height,Math.max(drag.y,drag.y+drag.h)*sy);if(x2-x1>5&&y2-y1>5){entry().boxes.push([x1,y1,x2,y2]);entry().complete=false}drag=null;persist()};
document.getElementById("export").onclick=()=>{const payload={schema_version:1,report:"Human item boxes for DINO held-out review",exported_at:new Date().toISOString(),source_review_report:"dino_review_report_20261022.json",coordinate_space:"EXIF-corrected upright image pixels; xyxy",scenes:scenes.map(s=>({filename:s.filename,group:s.group,source_sha256:s.sha256,width:s.width,height:s.height,boxes_xyxy:(state[s.filename]?.boxes||[]),review_complete:!!state[s.filename]?.complete,note:state[s.filename]?.note||""}))};const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}));a.download="dino_item_boxes_review.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById("clear").onclick=()=>{if(confirm("Clear all item boxes and completion marks saved in this browser?")){state={};persist()}};render();
</script></body></html>'''.replace("__DATA__", data)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT} with 30 photo scenes")


if __name__ == "__main__":
    main()
