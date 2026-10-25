"""Actual worker-to-local-SKU smoke test on an explicitly historical photo."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import db
from live_scan import ShopMatcher, normalize_photo, request_dino, render_items


def main():
    folder = ROOT / 'data/webshop_demo'
    config = json.loads((folder / 'worker_connection.json').read_text())
    os.environ['WEBSHOP_DINO_URL'] = config['url']
    os.environ['WEBSHOP_DINO_TOKEN'] = config['token']
    path = ROOT / 'data/segmentation_dataset/experiments/runpod_yolo11n_detection_20261017/package/final_test/images/IMG_4699.jpeg'
    image, normalized = normalize_photo(path.read_bytes())
    started = time.perf_counter()
    result = request_dino(normalized, image.size)
    print(json.dumps({'worker_seconds':result['elapsed_seconds'],'detections':len(result['detections'])}), flush=True)
    db.DATA_DIR, db.DB_FILE = folder, folder / 'inventorylens.sqlite3'
    matcher = ShopMatcher(db.get_business_products(1))
    ranked = matcher.rank_boxes(image, result['detections'])
    evidence = {'kind':'live_worker_smoke_on_historical_photo_not_blind_accuracy',
        'source':str(path), 'detector':result,'ranked_items':ranked,
        'elapsed_seconds':round(time.perf_counter()-started,2),
        'note':'New inference execution on an existing historical photo, not a new-photo generalization test.'}
    (folder / 'live_worker_smoke.json').write_text(json.dumps(evidence,indent=2))
    render_items(image, ranked).save(folder / 'live_worker_smoke.png')
    print(json.dumps({'seconds':evidence['elapsed_seconds'],'items':[
        {'id':r['item_id'],'sku':r['sku'],'top3':r['ranked_scores'][:3]} for r in ranked]},indent=2))


if __name__ == '__main__':
    main()
