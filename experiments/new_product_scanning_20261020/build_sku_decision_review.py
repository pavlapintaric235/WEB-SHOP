"""Build a local, browser-only decision sheet from provisional SKU ranks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="reviewed_item_sku_ranking_* result directory")
    parser.add_argument("--decisions", type=Path, help="optional exported human decisions to preload")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    data = json.loads((run / "report.json").read_text(encoding="utf-8"))
    candidates = data["gallery"]
    payload = json.dumps(data["photos"], ensure_ascii=False).replace("</", "<\\/")
    ids = json.dumps(candidates, ensure_ascii=False).replace("</", "<\\/")
    seed = {}
    if args.decisions:
        decisions_path = args.decisions if args.decisions.is_absolute() else ROOT / args.decisions
        exported = json.loads(decisions_path.read_text(encoding="utf-8"))
        for row in exported.get("decisions", []):
            if row.get("item_key"):
                seed[row["item_key"]] = {"sku_id": row.get("sku_id", ""),
                                         "state": row.get("state", "needs_review")}
    seed_json = json.dumps(seed, ensure_ascii=False).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SKU decision review</title><style>
*{box-sizing:border-box}body{font:15px/1.5 system-ui,sans-serif;background:#f3f6f8;color:#17242f;margin:0}main{max-width:1500px;margin:auto;padding:22px}h1{margin:0}.muted{color:#586b78}.warn{padding:13px;background:#fff4dc;border:1px solid #e8d39f;border-radius:9px;margin:14px 0}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}.toolbar select,button{font:inherit;padding:8px;border:1px solid #bdc9d2;border-radius:7px;background:white}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(420px,1fr);gap:12px}.panel{background:#fff;border:1px solid #dce3e8;border-radius:10px;padding:12px;min-width:0}.panel img{width:100%;height:auto;display:block;border-radius:6px}.tablewrap{overflow:auto;max-height:78vh}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e3e9ed;padding:8px}th{position:sticky;top:0;background:#f6f8f9}.rank{white-space:nowrap}.itemhead{font-weight:700}td select{width:100%;max-width:240px;padding:6px}.status{min-width:110px;font-weight:600}.good{color:#147341}.todo{color:#9a5900}.bad{color:#a8332b}button.primary{background:#174f74;color:white;border-color:#174f74}#progress{font-weight:600}@media(max-width:950px){.grid{grid-template-columns:1fr}main{padding:12px}}
</style></head><body><main><h1>Review product SKU suggestions</h1><p class="muted">Each green box is one manually annotated item. Choose the right provisional product ID for each item, or mark it unknown. Similarity is a ranking signal only.</p><div class="warn"><b>Before confirming:</b> the candidate IDs are provisional groups, and none of their sellable SKU identities has been confirmed. Do not use these decisions for stock updates. Confirm the product identity and package variant first. Decisions are saved in this browser; export the JSON when done.</div><div class="toolbar"><label for="scene">Photo</label><select id="scene"></select><span id="progress"></span><button id="export" class="primary">Export decisions JSON</button><button id="clear">Clear saved decisions</button><span id="message" class="muted"></span></div><div class="grid"><section class="panel"><img id="overlay" alt="Manual item boxes with top provisional SKU suggestion"></section><section class="panel tablewrap"><table><thead><tr><th>Item</th><th>Top suggestions</th><th>Your decision</th><th>Review state</th></tr></thead><tbody id="rows"></tbody></table></section></div></main><script>
const photos=__PHOTOS__, candidates=__CANDIDATES__, seed=__SEED__, key='inventorylens-sku-decisions-v1';let decisions={...JSON.parse(localStorage.getItem(key)||'{}'),...seed};const total=photos.reduce((n,s)=>n+s.items.length,0),sel=document.getElementById('scene');photos.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${s.group} · ${s.filename} · ${s.items.length} items`;sel.append(o)});function esc(x){const d=document.createElement('div');d.textContent=x;return d.innerHTML}function id(s,it){return `${s.filename}#${it.item_id}`}function persist(){localStorage.setItem(key,JSON.stringify(decisions));document.getElementById('message').textContent='Saved in this browser.';progress()}function progress(){const done=Object.values(decisions).filter(x=>x.sku_id&&['confirmed','uncertain'].includes(x.state)).length;document.getElementById('progress').textContent=`${done} / ${total} items reviewed`}function show(){const s=photos[+sel.value];document.getElementById('overlay').src=s.overlay;document.getElementById('overlay').alt=`${s.filename}: reviewed item boxes`;const b=document.getElementById('rows');b.replaceChildren();s.items.forEach(it=>{const k=id(s,it),d=decisions[k]||{sku_id:'',state:'needs_review'},tr=document.createElement('tr'),opts=[...new Map(it.rankings.combined.map(x=>[x.sku_id,x])).values()];let options='<option value="">Choose…</option>'+opts.map(x=>`<option value="${esc(x.sku_id)}">${esc(x.sku_id)} · ${esc(x.name)}</option>`).join('')+Object.entries(candidates).filter(([sku])=>!opts.some(x=>x.sku_id===sku)).map(([sku,v])=>`<option value="${esc(sku)}">${esc(sku)} · ${esc(v.name)}</option>`).join('')+'<option value="__unknown__">Unknown / not listed</option>';tr.innerHTML=`<td class="itemhead">Item ${it.item_id}<br><span class="muted">${it.box_xyxy.map(Math.round).join(', ')}</span></td><td>${it.rankings.combined.map((x,i)=>`<div class="rank">${i+1}. ${esc(x.sku_id)} · ${esc(x.name)} · ${x.similarity.toFixed(3)}</div>`).join('')}</td><td><select aria-label="SKU decision for ${esc(s.filename)} item ${it.item_id}">${options}</select></td><td><select class="status" aria-label="Review state"><option value="needs_review">Needs review</option><option value="confirmed">Confirmed by human</option><option value="uncertain">Uncertain</option></select></td>`;const sku=tr.querySelector('td:nth-child(3) select'),state=tr.querySelector('.status');sku.value=d.sku_id||'';state.value=d.state||'needs_review';sku.onchange=()=>{decisions[k]={...(decisions[k]||{}),sku_id:sku.value};persist()};state.onchange=()=>{decisions[k]={...(decisions[k]||{}),sku_id:sku.value,state:state.value};persist()};b.append(tr)});progress()}sel.onchange=show;document.getElementById('export').onclick=()=>{const out={format:'inventorylens_sku_decisions_v1',created_at:new Date().toISOString(),source:'provisional SigLIP2 ranks; human decisions only',decisions:Object.entries(decisions).map(([item_key,d])=>({item_key,...d}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='inventorylens_sku_decisions.json';a.click();URL.revokeObjectURL(a.href)};document.getElementById('clear').onclick=()=>{if(confirm('Clear all saved decisions from this browser?')){decisions={};persist();show()}};show();
</script></body></html>'''.replace("__PHOTOS__", payload).replace("__CANDIDATES__", ids).replace("__SEED__", seed_json)
    out = run / ("sku_decision_review_preloaded.html" if args.decisions else "sku_decision_review.html")
    out.write_text(page, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
