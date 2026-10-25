"""Same SigLIP2 scoring, with persisted approved references and content-bound caches."""
import hashlib
import json
import math
from pathlib import Path
from uuid import uuid4
from functools import lru_cache

import numpy as np

import db
from live_scan import ShopMatcher, reference_records, SIGLIP_REVISION
from registration_8509 import encoder
from live_scan import ROOT
from sam_vision import REFERENCE_CACHE_VERSION

REFERENCE_CACHE_ROOT=ROOT/'data/models/reference_cache'


def file_stamp(path):
    info=Path(path).stat()
    return (info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_ino)


@lru_cache(maxsize=4096)
def _source_hashes(path, stamp):
    data=Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(),hashlib.sha256(REFERENCE_CACHE_VERSION.encode('ascii')+data).hexdigest()


@lru_cache(maxsize=4096)
def _file_hash(path, stamp):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_hash(record):
    image=record['crop']
    return hashlib.sha256(SIGLIP_REVISION.encode()+record['representation'].encode()+str(image.size).encode()+image.tobytes()).hexdigest()


def gallery_fingerprint(products):
    from images import product_references
    records=[]
    for product in products:
        for reference in product_references(product):
            path=Path(reference['image_path'])
            source,key=_source_hashes(str(path),file_stamp(path))
            masked=REFERENCE_CACHE_ROOT/(key+'.npz')
            representation_hash=_file_hash(str(masked),file_stamp(masked)) if masked.exists() else 'full-reference'
            records.append((product['business_id'],product['sku'],product['name'],source,representation_hash))
    if not records:raise ValueError('This shop has no active reference photos. Complete product registration first.')
    with db.connect() as c:
        assets=db.rows(c.execute('SELECT source_sha256,embedding_path,revision FROM reference_assets WHERE business_id=? AND active=1',(products[0]['business_id'],)))
    embeddings=[(a['source_sha256'],a['revision'],_file_hash(a['embedding_path'],file_stamp(a['embedding_path']))) for a in assets]
    return hashlib.sha256(json.dumps([SIGLIP_REVISION,records,embeddings]).encode()).hexdigest()


class CachedShopMatcher(ShopMatcher):
    def __init__(self, products, image_encoder=None):
        if not products: raise ValueError('Register shop reference photos first.')
        shops={p['business_id'] for p in products}
        if len(shops)!=1: raise PermissionError('A gallery must belong to exactly one shop.')
        business_id=next(iter(shops))
        self.records=[]
        self.encoder=image_encoder
        folder=db.DATA_DIR/'siglip_gallery'/str(business_id);folder.mkdir(parents=True,exist_ok=True)
        fingerprint=gallery_fingerprint(products)
        snapshot=folder/('gallery_'+fingerprint+'.npz')
        if snapshot.exists():
            try:
                with np.load(snapshot,allow_pickle=False) as saved:
                    records=json.loads(str(saved['records']))
                    features=saved['features']
                    manifest=json.loads(str(saved['manifest']))
                    if (str(saved['fingerprint'])!=fingerprint or features.shape!=(len(records),768)
                        or not np.isfinite(features).all() or not np.all((np.linalg.norm(features,axis=1)>.9)&(np.linalg.norm(features,axis=1)<1.1))):
                        raise ValueError('Invalid gallery snapshot')
                self.records,self.features,self.manifest=records,features,manifest
                self._index();return
            except (OSError,ValueError,KeyError):
                pass  # Rebuild an invalid acceleration cache from authoritative references.
        with db.connect() as c:
            assets=db.rows(c.execute('SELECT * FROM reference_assets WHERE business_id=? AND active=1',(business_id,)))
        registered={(a['source_sha256'],a['representation']):a for a in assets}
        features=[]
        self.manifest=[]
        def records():
            # Decode one product at a time, never the whole megapixel gallery.
            for product in products:
                if __import__('images').product_references(product):
                    yield from reference_records([product])
        for record in records():
            self.records.append(record)
            digest=record_hash(record);cache=folder/(digest+'.npy')
            asset=registered.get((record['source_sha256'],record['representation']))
            if asset:
                with np.load(asset['embedding_path'],allow_pickle=False) as saved:
                    if str(saved['revision'])!=SIGLIP_REVISION or str(saved['source_sha256'])!=record['source_sha256'] or str(saved['representation'])!=record['representation']:
                        raise ValueError('Registered embedding metadata is stale. Re-register its references.')
                    vector=saved['feature']
                source='registered_embedding'
            elif cache.exists():
                vector=np.load(cache,allow_pickle=False);source='content_bound_cache'
            else:
                self.encoder=self.encoder or encoder()
                vector=self.encoder.encode([record['crop']],batch_size=1)[0]
                temporary=cache.with_name(cache.name+'.'+uuid4().hex+'.tmp')
                try:
                    with temporary.open('wb') as stream: np.save(stream,vector,allow_pickle=False)
                    temporary.replace(cache)
                finally:
                    temporary.unlink(missing_ok=True)
                source='encoded_current_representation'
            if vector.shape!=(768,) or not np.isfinite(vector).all() or not .9<np.linalg.norm(vector)<1.1:
                raise ValueError('Invalid cached SigLIP2 vector; registration needs repair.')
            features.append(vector)
            self.manifest.append(dict(sku=record['sku'],source_sha256=record['source_sha256'],representation=record['representation'],pixel_hash=digest,feature_source=source))
            # Scoring uses vectors and SKU metadata; do not retain megapixel RGB buffers.
            record['crop']=None
        self.features=np.stack(features)
        temporary=snapshot.with_name(snapshot.name+'.'+uuid4().hex+'.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,features=self.features,records=json.dumps(self.records),manifest=json.dumps(self.manifest),fingerprint=fingerprint)
            temporary.replace(snapshot)
        finally:
            temporary.unlink(missing_ok=True)
        self._index()

    def _index(self):
        self.by_sku={}
        for i,r in enumerate(self.records):self.by_sku.setdefault(r['sku'],[]).append(i)

    def prepare(self):
        self.encoder=self.encoder or encoder()

    def rank_boxes(self,image,detections,minimum=.55,margin=.05,query_batch_size=4):
        from live_scan import valid_box
        from experiments.hybrid_pipeline.hybrid_core import abstaining_decision
        if not detections:return []
        self.prepare()
        boxes=[valid_box(row['box'],image.size) for row in detections]
        crops=[image.crop((math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3]))) for b in boxes]
        vectors=self.encoder.encode(crops,batch_size=query_batch_size)
        output=[]
        for number,(detection,vector,box) in enumerate(zip(detections,vectors,boxes),1):
            sims=self.features @ vector
            scores={sku:float(np.mean(sorted((float(sims[i]) for i in indices),reverse=True)[:2])) for sku,indices in self.by_sku.items()}
            output.append(dict(item_id=number,box=box,detector_confidence=detection.get('confidence'),**abstaining_decision(scores,minimum,margin)))
        return output
