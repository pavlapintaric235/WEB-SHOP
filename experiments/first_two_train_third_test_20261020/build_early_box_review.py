"""Build a local browser review for the four unlabelled early scenes."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "draft_early_boxes.json"
OUTPUT = HERE / "early_box_review.html"
ARCHIVE = ROOT / "data/archives/20261014-223956-pre-grocery/test_runs"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    draft = json.loads(SOURCE.read_text(encoding="utf-8"))
    expected = {
        "20261014-205753-ae1ea812-test-original.png": 2,
        "20261014-210209-0bae16f4-test-original.png": 2,
        "20261014-213938-6d88db09-test-original.png": 6,
        "20261014-221857-d507c3fa-test-original.png": 4,
    }
    scenes = []
    for scene in draft["scenes"]:
        source = ARCHIVE / scene["source_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        scenes.append({
            "source_name": scene["source_name"],
            "source_sha256": scene["source_sha256"],
            "url": source.relative_to(ROOT).as_posix(),
            "width": scene["width"], "height": scene["height"],
            "visual_count_to_verify": expected[scene["source_name"]],
            "model_proposals": scene["boxes"],
        })
    dataset = json.dumps(scenes, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Early product box review</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202532;background:#f6f7f9}
h1{margin-bottom:.3rem}p{line-height:1.45}button,input{font:inherit}button{padding:.55rem .8rem;margin:.2rem;border:1px solid #8893a0;border-radius:.35rem;background:white;cursor:pointer}
button:hover{background:#e9eff8}.panel{background:white;padding:1rem;border-radius:.7rem;margin:1rem 0;border:1px solid #dce1e6}
#stage{position:relative;width:min(100%,950px);line-height:0;user-select:none;touch-action:none}#photo{display:block;width:100%;height:auto}#overlay{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}
.muted{color:#586173}.warning{color:#a04511}.ok{color:#136e3d}#status{font-weight:600}label{display:inline-block;margin:.5rem .8rem .5rem 0}
</style>
<h1>K18 / WOW / coconut: box review</h1>
<p>Draw <b>one blue box around each physical product</b>. Drag on the photo to add a box; click an existing box to select it, then delete it. Orange outlines are saved model proposals for reference only. The scene with a black background appears to be another version of the two-bottle K18 arrangement and must stay in the same training group.</p>
<p class="warning">The count shown is a visual draft, not approved ground truth. Check the entire photo, correct the count, and mark each image reviewed. This page does not train a model or edit the source photos.</p>
<div class="panel"><div id="selector"></div><h2 id="title"></h2><div id="stage"><img id="photo" alt="Early product scene"><canvas id="overlay"></canvas></div>
<div><button id="undo">Undo last box</button><button id="delete">Delete selected box</button><button id="clear">Clear boxes</button></div>
<label>Visible product count: <input id="count" type="number" min="0" max="100" style="width:5rem"></label>
<label><input id="complete" type="checkbox"> I checked every visible product and each blue box</label>
<p id="status"></p></div>
<div class="panel"><button id="download">Download reviewed boxes JSON</button><span id="exportStatus" class="muted"></span></div>
<script>const scenes=__DATA__;
const state=scenes.map(s=>({boxes:[],count:s.visual_count_to_verify,complete:false}));
let current=0,selected=-1,start=null,drag=null;
const photo=document.getElementById('photo'),canvas=document.getElementById('overlay'),ctx=canvas.getContext('2d');
function pos(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width,(e.clientX-r.left)*canvas.width/r.width)),Math.max(0,Math.min(canvas.height,(e.clientY-r.top)*canvas.height/r.height))]}
function rect(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])].map(x=>Math.round(x))}
function draw(){const s=scenes[current],st=state[current];ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=Math.max(5,canvas.width/450);ctx.font=`bold ${Math.max(22,canvas.width/100)}px system-ui`;
for(const p of s.model_proposals){const [x1,y1,x2,y2]=p.xyxy;ctx.strokeStyle='#fb8c00';ctx.setLineDash([20,12]);ctx.strokeRect(x1,y1,x2-x1,y2-y1)}ctx.setLineDash([]);
st.boxes.forEach((b,i)=>{const [x1,y1,x2,y2]=b;ctx.strokeStyle=i===selected?'#f02075':'#0078d7';ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.lineWidth=Math.max(5,canvas.width/450);ctx.fillStyle='#0078d7';ctx.fillRect(x1,y1-38,60,38);ctx.fillStyle='white';ctx.fillText(String(i+1),x1+7,y1-8)});
if(drag){const [x1,y1,x2,y2]=rect(start,drag);ctx.strokeStyle='#0078d7';ctx.strokeRect(x1,y1,x2-x1,y2-y1)}
const status=document.getElementById('status');status.textContent=`Blue boxes: ${st.boxes.length} / visible count: ${st.count} | ${st.complete?'marked reviewed':'still unreviewed'}`;status.className=st.complete&&st.boxes.length===st.count?'ok':'warning'}
function show(i){current=i;selected=-1;start=null;drag=null;const s=scenes[i],st=state[i];document.getElementById('title').textContent=`${i+1} of ${scenes.length}: ${s.source_name}`;photo.src='../../'+s.url;photo.onload=()=>{canvas.width=s.width;canvas.height=s.height;draw()};document.getElementById('count').value=st.count;document.getElementById('complete').checked=st.complete;
document.getElementById('selector').innerHTML='';scenes.forEach((item,j)=>{const b=document.createElement('button');b.textContent=`${j+1}: ${j<2?'K18 pair':j===2?'Mixed six':'Mixed four'} ${state[j].complete?'✓':''}`;b.onclick=()=>show(j);document.getElementById('selector').appendChild(b)});draw()}
canvas.addEventListener('pointerdown',e=>{e.preventDefault();canvas.setPointerCapture(e.pointerId);start=pos(e);drag=start});
canvas.addEventListener('pointermove',e=>{if(start){drag=pos(e);draw()}});
canvas.addEventListener('pointerup',e=>{if(!start)return;const end=pos(e),b=rect(start,end),st=state[current];if(b[2]-b[0]>=12&&b[3]-b[1]>=12){st.boxes.push(b);st.complete=false;document.getElementById('complete').checked=false;selected=-1}else{selected=st.boxes.findIndex(q=>end[0]>=q[0]&&end[0]<=q[2]&&end[1]>=q[1]&&end[1]<=q[3])}start=null;drag=null;draw()});
document.getElementById('count').onchange=e=>{state[current].count=Math.max(0,Number(e.target.value)||0);state[current].complete=false;document.getElementById('complete').checked=false;draw()};
document.getElementById('complete').onchange=e=>{const st=state[current];if(e.target.checked&&st.boxes.length!==st.count){alert('The number of blue boxes must equal the verified visible-product count.');e.target.checked=false;return}st.complete=e.target.checked;show(current)};
document.getElementById('undo').onclick=()=>{state[current].boxes.pop();state[current].complete=false;document.getElementById('complete').checked=false;selected=-1;draw()};
document.getElementById('delete').onclick=()=>{if(selected>=0){state[current].boxes.splice(selected,1);selected=-1;state[current].complete=false;document.getElementById('complete').checked=false;draw()}};
document.getElementById('clear').onclick=()=>{state[current].boxes=[];state[current].complete=false;document.getElementById('complete').checked=false;selected=-1;draw()};
document.getElementById('download').onclick=()=>{if(state.some(s=>!s.complete||s.boxes.length!==s.count)){document.getElementById('exportStatus').textContent='Review every scene and match each visible count first.';return}
const result={schema_version:1,status:'human_review_export_pending_import',reviewed_at:new Date().toISOString(),scenes:scenes.map((s,i)=>({source_name:s.source_name,source_sha256:s.source_sha256,width:s.width,height:s.height,visible_product_count:state[i].count,boxes_xyxy:state[i].boxes,review_complete:true}))};
const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='inventorylens_early_product_boxes_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('exportStatus').textContent='Downloaded. Send this JSON for validation and training preparation.'};show(0);
</script></html>'''.replace("__DATA__", dataset)
    OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
