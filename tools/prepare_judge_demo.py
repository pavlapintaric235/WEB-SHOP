"""Create an isolated fictional shop with no private images or default passwords."""
import json
import os
from pathlib import Path
import secrets
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    folder=ROOT/'data/webshop_judge_demo'
    access=ROOT/'data/judge_release/access.json'
    if (folder/'inventorylens.sqlite3').exists():
        print('Existing judge sandbox preserved; no reset or credential overwrite.');return
    os.environ['WEBSHOP_DATA_DIR']=str(folder)
    import db,auth
    from registration_8509 import init_registration
    from inventory import InventoryService
    from shop_catalog import read_catalog_bundle
    db.init_db();init_registration()
    business=db.get_or_create_business('WEB&SHOP Judge Sandbox - FICTIONAL DEMO')
    with db.connect() as c:c.execute('UPDATE businesses SET is_demo=1 WHERE id=?',(business['id'],))
    credentials=dict(owner_username='judge-owner',owner_password=secrets.token_urlsafe(24),employee_username='judge-employee',employee_password=secrets.token_urlsafe(24),access_code=secrets.token_urlsafe(24))
    for role in ('owner','employee'):
        auth.create_user(business['id'],credentials[role+'_username'],credentials[role+'_password'],role)
    owner=InventoryService(auth.authenticate(credentials['owner_username'],credentials['owner_password'])['id'])
    sample=ROOT/'demo_data/WEB_SHOP_mock_catalog_prices_shelves_map.xlsx'
    bundle=read_catalog_bundle(sample.read_bytes(),sample.name)
    owner.import_catalog(bundle['products'],confirmed=True,opening=False,shelves=bundle['shelves'],fictional=True)
    assert all(not p['references'] and not p['image_path'] for p in owner.products())
    access.parent.mkdir(parents=True,exist_ok=True)
    access.write_text(json.dumps(credentials,indent=2))
    access.chmod(0o600)
    print('Judge sandbox prepared with fictional catalogue/shelves and no copied images or embeddings.')
    print('Separate generated credentials are in ignored data/judge_release/access.json. No credentials printed.')


if __name__=='__main__':main()
