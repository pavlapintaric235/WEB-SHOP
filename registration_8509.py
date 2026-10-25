"""Atomic visual registration in the existing catalogue, isolated 8509 UI only."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

import numpy as np
from PIL import Image
import streamlit as st

import db
from live_scan import normalize_photo, DEFAULT_MODEL, SIGLIP_REVISION
from quality_8509 import accepts


def init_registration():
    with db.connect() as c:
        columns={r['name'] for r in c.execute('PRAGMA table_info(products)')}
        for name in ('brand','variant','package_size','barcode','description','identity_key'):
            if name not in columns:
                c.execute(f'ALTER TABLE products ADD COLUMN {name} TEXT')
        c.executescript('''CREATE UNIQUE INDEX IF NOT EXISTS registered_identity ON products(business_id,identity_key) WHERE identity_key IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS registered_barcode ON products(business_id,barcode) WHERE barcode IS NOT NULL AND barcode <> '';
            CREATE TABLE IF NOT EXISTS reference_assets (
                id INTEGER PRIMARY KEY, business_id INTEGER NOT NULL REFERENCES businesses(id),
                product_id INTEGER NOT NULL REFERENCES products(id), source_sha256 TEXT NOT NULL,
                original_path TEXT NOT NULL, normalized_path TEXT NOT NULL, embedding_path TEXT NOT NULL,
                mask_path TEXT, representation TEXT NOT NULL, angle TEXT NOT NULL,
                quality_json TEXT NOT NULL, revision TEXT NOT NULL, active INTEGER NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(product_id,source_sha256));''')


@st.cache_resource(show_spinner='Loading the existing local SigLIP2 encoder…')
def encoder():
    import os
    from siglip_8509 import ImageTowerEncoder
    return ImageTowerEncoder(os.environ.get('WEBSHOP_SIGLIP_PATH',str(DEFAULT_MODEL)),SIGLIP_REVISION)


def assets_for(service, product_id):
    user=service.user()
    with db.connect() as c:
        return db.rows(c.execute('SELECT * FROM reference_assets WHERE business_id=? AND product_id=? ORDER BY id',(user['business_id'],product_id)))


def details_checked(details):
    result={k:str(details.get(k,'')).strip() for k in ('sku','name','brand','variant','package_size','unit','shelf_id','barcode','description')}
    if not all(result[k] for k in ('name','brand','package_size','unit','shelf_id')):
        raise ValueError('Enter name, brand, package size, quantity unit and shelf.')
    result['sku']=result['sku'].upper() or 'P-'+uuid4().hex[:16].upper()
    if not re.fullmatch(r'[A-Z0-9_-]{1,40}',result['sku']):
        raise ValueError('SKU: 1–40 letters, numbers, hyphens or underscores.')
    if result['barcode'] and not re.fullmatch(r'\d{8,14}',result['barcode']):
        raise ValueError('Barcode/EAN must contain 8–14 digits, or be blank.')
    price=float(details.get('price',0))
    if not math.isfinite(price) or price<0: raise ValueError('Enter a nonnegative euro price.')
    stock=details.get('initial_quantity',0);db.validate_quantity(stock)
    result.update(price=price,initial_quantity=stock,brand_variant_size=' '.join(result[k] for k in ('brand','variant','package_size')).strip())
    result['identity_key']=hashlib.sha256('|'.join(result[k].casefold() for k in ('name','brand','variant','package_size')).encode()).hexdigest()
    return result


def register(service, details, photos, *, confirmed=False, incomplete=False, existing_id=None, image_encoder=None):
    user=service.user(owner=True)
    if not confirmed: raise ValueError('Confirm the product summary first.')
    d=details_checked(details)
    with db.connect() as c:
        location=c.execute('SELECT id FROM inventory_locations WHERE business_id=? AND name=?',(user['business_id'],d['shelf_id'])).fetchone()
        if not location: raise PermissionError('Choose a shelf belonging to this shop.')
        existing=c.execute('SELECT * FROM products WHERE id=? AND business_id=?',(existing_id,user['business_id'])).fetchone() if existing_id else None
        if existing_id and not existing: raise PermissionError('Product belongs to another shop.')
        if existing and existing['sku']!=d['sku']: raise ValueError('An existing product SKU is stable and cannot be replaced.')
        duplicate=c.execute('SELECT id FROM products WHERE business_id=? AND (sku=? OR identity_key=? OR (? <> \'\' AND barcode=?))',
                            (user['business_id'],d['sku'],d['identity_key'],d['barcode'],d['barcode'])).fetchall()
        if any(r['id']!=existing_id for r in duplicate): raise ValueError('SKU, exact package identity or barcode already exists. Open that product instead of creating a duplicate.')
        active=db.rows(c.execute('SELECT * FROM product_references WHERE product_id=?',(existing_id,))) if existing else []
    old_hashes={hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest() for r in active}
    unique=set();approved=[]
    for photo in photos:
        digest=hashlib.sha256(photo['data']).hexdigest()
        if digest in unique: raise ValueError('Use different photographs for reference views.')
        unique.add(digest)
        if digest in old_hashes: raise ValueError('This reference photograph is already registered.')
        record=photo['quality']
        if not photo.get('approved') or record.get('image_sha256')!=digest or not accepts(record,photo.get('override','')):
            raise ValueError('Replace unusable photographs and approve each eligible image-quality result.')
        approved.append((photo,digest))
    approved.sort(key=lambda item: item[0].get('angle')!='front')
    count=len(active)+len(approved)
    if count>5 or (not incomplete and count<3): raise ValueError('Recognition registration needs 3–5 approved reference photographs.')
    ready=count>=3 and not incomplete
    folder=db.DATA_DIR/'registered_references'/str(user['business_id'])/uuid4().hex
    saved=[]
    try:
        for index,(photo,digest) in enumerate(approved):
            image,normalized=normalize_photo(photo['data'])
            vector=np.asarray((image_encoder or encoder()).encode([image],batch_size=1)[0],dtype=np.float32)
            if vector.ndim!=1 or not np.isfinite(vector).all() or np.linalg.norm(vector)<.9:
                raise ValueError('SigLIP2 returned an invalid reference embedding.')
            folder.mkdir(parents=True,exist_ok=True)
            with Image.open(io.BytesIO(photo['data'])) as original: ext='.png' if original.format=='PNG' else '.jpg'
            original_path=folder/f'{index}_original{ext}';original_path.write_bytes(photo['data'])
            normalized_path=folder/f'{index}_upright.png';normalized_path.write_bytes(normalized)
            embedding=folder/f'{index}_siglip2.npz'
            np.savez_compressed(embedding,feature=vector,revision=SIGLIP_REVISION,source_sha256=digest,representation='full_reference_photo')
            saved.append(dict(photo=photo,digest=digest,original=original_path,normalized=normalized_path,embedding=embedding))
        with db.connect() as c:
            if existing:
                product_id=existing_id
                if not active: c.execute('DELETE FROM reference_assets WHERE product_id=?',(product_id,))
            else:
                product_id=c.execute('INSERT INTO products(business_id,sku,name,image_path,created_at) VALUES (?,?,?,?,?)',
                                     (user['business_id'],d['sku'],d['name'],'',db.now())).lastrowid
            c.execute('UPDATE products SET name=?,price=?,unit=?,shelf_id=?,brand_variant_size=?,brand=?,variant=?,package_size=?,barcode=?,description=?,identity_key=?,registration_status=?,opening_stock=? WHERE id=? AND business_id=?',
                      tuple(d[k] for k in ('name','price','unit','shelf_id','brand_variant_size','brand','variant','package_size','barcode','description','identity_key'))+
                      ('reference ready; recognition test pending' if ready else 'incomplete registration draft',d['initial_quantity'],product_id,user['business_id']))
            for saved_photo in saved:
                photo=saved_photo['photo']
                c.execute('INSERT INTO reference_assets(business_id,product_id,source_sha256,original_path,normalized_path,embedding_path,representation,angle,quality_json,revision,active,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (user['business_id'],product_id,saved_photo['digest'],str(saved_photo['original']),str(saved_photo['normalized']),str(saved_photo['embedding']),
                     'full_reference_photo',photo.get('angle','unknown'),json.dumps({'analysis':photo['quality'],'override':photo.get('override',''),'approved_by':user['id']}),SIGLIP_REVISION,int(ready),db.now()))
                if ready: c.execute('INSERT INTO product_references(product_id,image_path,created_at) VALUES (?,?,?)',(product_id,str(saved_photo['original']),db.now()))
            if ready:
                front=next((p for p in saved if p['photo'].get('angle')=='front'),None)
                primary=str(front['original']) if front else active[0]['image_path'] if active else str(saved[0]['original'])
                c.execute('UPDATE products SET image_path=? WHERE id=?',(primary,product_id))
                if not active and not c.execute('SELECT 1 FROM inventory_counts WHERE business_id=? AND location_id=? AND product_id=?',(user['business_id'],location['id'],product_id)).fetchone():
                    receipt=c.execute("INSERT INTO scan_sessions(business_id,user_id,location_id,photo_hash,submission_key,scope,status,notes,created_at,operation) VALUES (?,?,?,?,?,'partial','approved',?,?,'audit')",
                        (user['business_id'],user['id'],location['id'],hashlib.sha256(''.join(sorted(unique)).encode()).hexdigest(),'registration-'+str(product_id),'Owner-approved initial stock at registration',db.now())).lastrowid
                    c.execute('INSERT INTO scan_items(session_id,product_id,predicted_quantity,approved_quantity,before_quantity,after_quantity) VALUES (?,?,0,?,0,?)',(receipt,product_id,d['initial_quantity'],d['initial_quantity']))
                    db._set_inventory_count(c,user['business_id'],location['id'],product_id,d['initial_quantity'])
        return dict(id=product_id,sku=d['sku'],ready=ready,references=count,recognition_test='pending')
    except Exception:
        if folder.exists():
            for file in folder.iterdir(): file.unlink()
            folder.rmdir()
        raise


def update_details(service, product_id, price, shelf_id):
    user=service.user(owner=True)
    if not math.isfinite(float(price)) or price<0: raise ValueError('Invalid price.')
    if shelf_id not in {r['name'] for r in service.locations()}: raise PermissionError('Unknown shop shelf.')
    with db.connect() as c:
        if not c.execute('UPDATE products SET price=?,shelf_id=? WHERE id=? AND business_id=?',(price,shelf_id,product_id,user['business_id'])).rowcount:
            raise PermissionError('Unknown shop product.')


def export_catalog(service):
    service.user(owner=True)
    fields=['sku','name','brand','variant','package_size','unit','price','opening_stock','shelf_id','barcode','description','registration_status']
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
    for p in service.products(): writer.writerow({k:p.get(k,'') for k in fields})
    return output.getvalue()
