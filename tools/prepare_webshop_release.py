"""Create a new isolated release demo; never alter source catalogues or balances."""
import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auth
import db
from shop_catalog import import_catalog, read_catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'data/webshop_demo')
    parser.add_argument('--target', type=Path, default=ROOT / 'data/webshop_release_demo')
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite existing demo: {target}')
    records = read_catalog((source / 'demo_products.csv').read_bytes(), 'demo_products.csv')
    # Read the existing source only. Copy approved registered reference paths, not its stock/history.
    with sqlite3.connect(f'{(source / "inventorylens.sqlite3").as_uri()}?mode=ro', uri=True) as connection:
        source_refs = connection.execute('SELECT p.sku,r.image_path FROM product_references r JOIN products p ON p.id=r.product_id ORDER BY p.sku,r.id').fetchall()
    for sku, path in source_refs:
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing registered reference for {sku}')
    target.mkdir(parents=True)
    db.DATA_DIR, db.DB_FILE, db.CATALOG_FILE = target, target / 'inventorylens.sqlite3', target / 'catalog.json'
    db.init_db()
    business = auth.seed_demo_users()
    with db.connect() as connection:
        connection.execute('UPDATE businesses SET name=? WHERE id=?', ('WEB&SHOP Corner Market (FICTIONAL DEMO)', business['id']))
    import_catalog(business['id'], records, confirmed=True, opening=True)
    with db.connect() as connection:
        for index, (sku, path) in enumerate(source_refs):
            folder = target / 'references' / sku
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / f'{index}{Path(path).suffix}'
            shutil.copy2(path, dest)
            product = connection.execute('SELECT id FROM products WHERE business_id=? AND sku=?', (business['id'], sku)).fetchone()
            if product is None:
                raise ValueError(f'Reference SKU is absent from workbook: {sku}')
            connection.execute('INSERT INTO product_references (product_id,image_path,created_at) VALUES (?,?,?)', (product['id'], str(dest), db.now()))
            connection.execute("UPDATE products SET image_path=? WHERE id=? AND image_path=''", (str(dest), product['id']))
    for name in ('demo_products.csv', 'demo_catalog.xlsx', 'demo_shop.json'):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    manifest = dict(kind='isolated_fictional_demo', products=len(records), references=len(source_refs),
                    note='Prices, opening stock and shelf labels are fictional. Package identities require owner confirmation. Fripsy remains pending.')
    (target / 'release_seed.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(dict(data_dir=str(target), **manifest), indent=2))


if __name__ == '__main__':
    main()
