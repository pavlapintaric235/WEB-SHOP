"""Bounded performance check. --live sends one genuine existing-worker request."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(WEBSHOP_DATA_DIR=str(ROOT/'data/webshop_ux_demo'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')


def main():
    args=argparse.ArgumentParser();args.add_argument('--live',action='store_true');options=args.parse_args()
    import auth,db
    from inventory import InventoryService
    from gallery_8509 import CachedShopMatcher,gallery_fingerprint
    from scanner_8509 import prepared_photo
    from live_scan import request_dino
    products=InventoryService(auth.authenticate('owner','owner123')['id']).products()
    acceptance=json.loads((db.DATA_DIR/'live_registration_acceptance.json').read_text())
    data=Path(acceptance['scene']).read_bytes()
    report=dict(kind='actual local model timings; archived boxes unless live explicitly requested')
    def measured(name,fn):
        started=time.perf_counter();value=fn();report[name]=round(time.perf_counter()-started,4)
        print(name,report[name],flush=True);return value
    image,normalized,_=measured('photo_first_seconds',lambda:prepared_photo(data))
    measured('photo_repeat_seconds',lambda:prepared_photo(data))
    measured('fingerprint_first_seconds',lambda:gallery_fingerprint(products))
    measured('fingerprint_repeat_seconds',lambda:gallery_fingerprint(products))
    class ExistingVectorsOnly:
        def encode(self,*args,**kwargs):raise RuntimeError('Missing real cached vector; this bounded check will not regenerate the gallery.')
    matcher=measured('gallery_initial_seconds',lambda:CachedShopMatcher(products,image_encoder=ExistingVectorsOnly()))
    measured('gallery_snapshot_seconds',lambda:CachedShopMatcher(products,image_encoder=ExistingVectorsOnly()))
    matcher.encoder=None
    measured('model_prepare_seconds',matcher.prepare)
    boxes=acceptance['scene_result']['detector']['detections']
    if options.live:
        connection=json.loads((ROOT/'data/webshop_demo/worker_connection.json').read_text())
        os.environ.update(WEBSHOP_DINO_URL=connection['url'],WEBSHOP_DINO_TOKEN=connection['token'])
        detector=measured('live_dino_request_seconds',lambda:request_dino(normalized,image.size))
        boxes=detector['detections'];report['live_detector']=detector
    serial=measured('serial_siglip_matching_seconds',lambda:matcher.rank_boxes(image,boxes,query_batch_size=1))
    items=measured('actual_siglip_matching_seconds',lambda:matcher.rank_boxes(image,boxes))
    assert [i['sku'] for i in serial]==[i['sku'] for i in items]
    report['assignments']=[i['sku'] for i in items]
    report['same_assignments_as_prior_actual_scene']=report['assignments']==[i['sku'] for i in acceptance['scene_result']['items']]
    (db.DATA_DIR/'speed_after.json').write_text(json.dumps(report,indent=2))
    assert report['same_assignments_as_prior_actual_scene']
    if options.live:report['prepared_live_recognition_seconds']=round(report['live_dino_request_seconds']+report['actual_siglip_matching_seconds'],4)
    (db.DATA_DIR/'speed_after.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
