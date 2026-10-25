"""OpenCV evidence and decision UI for the isolated release, never SKU identity."""
import hashlib
import time

import cv2
import numpy as np
import streamlit as st

from live_scan import normalize_photo
from experiments.opencv5_guided_scan.capture_gate import quality, capture_decision


def analyze(data, provider='LOCAL'):
    started = time.perf_counter()
    image, _ = normalize_photo(data)
    pixels = np.asarray(image)[:, :, ::-1].copy()
    metrics = quality(pixels)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    scale = min(1, 1024/max(image.size))
    gray = cv2.resize(gray, (round(image.width*scale), round(image.height*scale)), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 80, 160)
    margin = max(1, min(edges.shape)//30)
    border = np.concatenate([edges[:margin].ravel(), edges[-margin:].ravel(), edges[:, :margin].ravel(), edges[:, -margin:].ravel()])
    metrics['border_edge_pct'] = round(float(np.mean(border > 0)*100), 2)
    return dict(provider=provider, cv2_version=cv2.__version__, image_sha256=hashlib.sha256(data).hexdigest(),
                metrics=metrics, decision=capture_decision(metrics),
                framing_review=metrics['border_edge_pct'] > 15,
                elapsed_seconds=round(time.perf_counter()-started, 3))


def aws_analyze(data):
    from experiments.opencv5_guided_scan.aws_capture_client import run_capture
    started = time.perf_counter()
    result = run_capture(data, hashlib.sha256(data).hexdigest())
    return dict(provider='AWS', cv2_version=result['runtime']['cv2_version'],
                image_sha256=result['image_sha256'], metrics=result['quality'],
                decision=result['capture_decision'], elapsed_seconds=round(time.perf_counter()-started, 3),
                aws_evidence=result, cache_notice='The existing service reuses results for identical image hashes.')


def accepts(record, override=''):
    if record.get('provider') not in {'AWS','LOCAL'} or str(record.get('cv2_version','')).split('.')[0] != '5':
        return False
    if record['decision']['action'] == 'proceed':
        return True
    m = record['metrics']
    # Exposure-only review may be overridden for a product on a bright background.
    # Blur, decode failure and insufficient resolution require replacement.
    return (len(override.strip()) >= 8 and m.get('decode_ok') and
            min(m['width'],m['height']) >= 720 and m['lap_var_1024'] >= 100 and
            record['decision']['reason'] in {'Reduce glare','Use more light'})


def diagnostics(record, title='Image quality'):
    st.markdown(f'**{title}**')
    provider = record.get('provider','UNAVAILABLE')
    decision = record.get('decision',{}).get('action','unavailable').upper()
    reason = record.get('decision',{}).get('reason', record.get('error','Analysis unavailable'))
    columns = st.columns(3)
    columns[0].metric('Analysis provider', provider)
    columns[1].metric('Decision', 'PROCEED' if decision=='PROCEED' else 'RETAKE' if decision=='RETAKE' else 'UNAVAILABLE')
    columns[2].metric('OpenCV', record.get('cv2_version','unavailable'))
    m = record.get('metrics',{})
    st.caption(f"{reason} · {record.get('elapsed_seconds',0):.2f}s")
    st.caption(f"Sharpness {m.get('lap_var_1024','—')} · dark {m.get('dark_pct','—')}% · bright {m.get('bright_pct','—')}% · {m.get('width','?')} × {m.get('height','?')}")
    st.caption('SHA-256: ' + record.get('image_sha256','unavailable'))
    if record.get('framing_review'):
        st.warning('Edges near the frame: inspect framing and keep the whole package visible. This is not an identity check.')
    if record.get('cache_notice'):
        st.caption(record['cache_notice'])


def box_quality(image, detections):
    """Geometry and crop focus warnings; overlap is not visible-product percentage."""
    width,height=image.size
    rows=[]
    for i, item in enumerate(detections,1):
        x1,y1,x2,y2=map(float,item['box'])
        polygon=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=np.float32)
        near_edge=min(x1,y1,width-x2,height-y2)<=max(2,min(width,height)*.01)
        crop=np.asarray(image.crop((int(x1),int(y1),max(int(x1)+1,int(x2)),max(int(y1)+1,int(y2)))))
        focus=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var()) if crop.size else 0
        overlaps=[]
        for j,other in enumerate(detections,1):
            if i==j: continue
            a,b,c,d=map(float,other['box'])
            other_poly=np.array([[a,b],[c,b],[c,d],[a,d]],dtype=np.float32)
            intersection,_=cv2.intersectConvexConvex(polygon,other_poly)
            union=cv2.contourArea(polygon)+cv2.contourArea(other_poly)-intersection
            iou=float(intersection/union) if union else 0
            if iou>=.35: overlaps.append({'box':f'D{j}','iou':round(iou,3)})
        rows.append(dict(item_id=f'D{i}',near_frame=near_edge,crop_laplacian=round(focus,2),overlaps=overlaps))
    return dict(provider='LOCAL',cv2_version=cv2.__version__,kind='post_dino_geometry',boxes=rows,
                guidance=['Review overlapping boxes; split merged products or take another angle.' if any(r['overlaps'] for r in rows) else 'No high-IoU pairs flagged.',
                          'Frame-edge boxes need inspection or camera repositioning.' if any(r['near_frame'] for r in rows) else 'No frame-edge boxes flagged.'],
                limitation='Box IoU and edge contact do not measure precise visible product percentage.')
