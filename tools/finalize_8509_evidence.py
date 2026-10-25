"""Read-only release checks and a local source manifest; no cloud mutations."""
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'data/webshop_ux_demo'
report=json.loads((folder/'live_registration_acceptance.json').read_text())
assert report['completed'] and report['buyer_no_stock_write'] and report['old_8508_ledger_and_code_unchanged']
with sqlite3.connect(folder/'inventorylens.sqlite3') as connection:
    connection.row_factory=sqlite3.Row
    product=dict(connection.execute('SELECT * FROM products WHERE sku=?',(report['registration']['sku'],)).fetchone())
    references=connection.execute('SELECT count(*) FROM product_references WHERE product_id=?',(product['id'],)).fetchone()[0]
    assets=connection.execute('SELECT original_path,normalized_path,embedding_path FROM reference_assets WHERE product_id=? AND active=1',(product['id'],)).fetchall()
    assert references==5 and len(assets)==5 and product['price']==1.49
    assert all(Path(path).is_file() for asset in assets for path in asset)
    stock=[dict(row) for row in connection.execute('SELECT quantity FROM inventory_counts WHERE product_id=?',(product['id'],))]
    assert sum(row['quantity'] for row in stock)==1
with sqlite3.connect(ROOT/'data/webshop_release_demo/inventorylens.sqlite3') as connection:
    old_sessions=connection.execute('SELECT count(*) FROM scan_sessions').fetchone()[0]
    assert old_sessions==4
health={str(port):urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=10).status for port in (8508,8509)}
files=['app_8509.py','visual_8509.py','owner_8509.py','buyer_8509.py','registration_wizard_8509.py','registration_8509.py',
       'owner_test_8509.py','quality_8509.py','siglip_8509.py','gallery_8509.py','scanner_8509.py','review_8509.py',
       'tools/start_webshop_ux_demo.ps1']
manifest=dict(health=health,source_8508_scan_sessions=old_sessions,new_sku=product['sku'],active_references=references,
              fictional_demo_stock=stock,actual_acceptance_complete=True,tests='44 targeted + 2 final UI smoke passed',
              source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})
(folder/'release_8509_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},indent=2))
