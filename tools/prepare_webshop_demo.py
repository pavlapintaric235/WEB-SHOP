"""Seed an isolated second-set demo database, catalog workbook and shelf layout."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import auth
import db

SELECTED = ['ENERGIA-POWER-250', 'DUKAT-MLIJEKO-1000', 'MAKS-CRISPY-50',
            'JANA-ANTIOX-500', 'FENTIMANS-GINGER-200']


def write_workbook(path, sheets):
    """Write a small standards-based XLSX fixture without an optional Excel package."""
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with ZipFile(path, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' +
            ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets) + 1)) + '</Types>')
        archive.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="' + relns + '/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{relns}"><sheets>' +
            ''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheets, 1)) + '</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
            ''.join(f'<Relationship Id="rId{i}" Type="{relns}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1)) + '</Relationships>')
        for i, (_, rows) in enumerate(sheets, 1):
            xml_rows = []
            for row_number, row in enumerate(rows, 1):
                cells = []
                for column, value in enumerate(row):
                    address = f'{chr(65 + column)}{row_number}'
                    if isinstance(value, (int, float)):
                        cells.append(f'<c r="{address}"><v>{value}</v></c>')
                    else:
                        cells.append(f'<c r="{address}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
                xml_rows.append(f'<row r="{row_number}">' + ''.join(cells) + '</row>')
            archive.writestr(f'xl/worksheets/sheet{i}.xml', f'<worksheet xmlns="{ns}"><sheetData>' + ''.join(xml_rows) + '</sheetData></worksheet>')


def main():
    source_db = ROOT / 'data/inventorylens.sqlite3'
    target = ROOT / 'data/webshop_demo'
    if db.DB_FILE != source_db:
        raise RuntimeError('Run preparation without WEBSHOP_DATA_DIR; the source must be the existing catalog.')
    if target.exists():
        raise FileExistsError('Demo data already exists. It was left untouched: ' + str(target))
    source = {p['sku']: p for p in db.get_business_products(1)}
    missing = set(SELECTED) - set(source)
    if missing:
        raise ValueError('Missing catalog entries: ' + ', '.join(sorted(missing)))
    for sku in SELECTED:
        for ref in source[sku]['references']:
            if not Path(ref['image_path']).is_file():
                raise FileNotFoundError(ref['image_path'])
    target.mkdir(parents=True)
    db.DATA_DIR, db.DB_FILE, db.CATALOG_FILE = target, target / 'inventorylens.sqlite3', target / 'catalog.json'
    db.init_db()
    business = auth.seed_demo_users()
    with db.connect() as connection:
        connection.execute('UPDATE businesses SET name=? WHERE id=?', ('Web&Shop Corner Market (demo)', business['id']))
    shelves = [('A1', 'Drinks', 1, 1, 2, 1), ('A2', 'Milk', 4, 1, 2, 1), ('B1', 'Snacks', 1, 4, 2, 1)]
    for shelf in shelves:
        db.add_location(business['id'], shelf[0])
    locations = {r['name']: r['id'] for r in db.get_locations(business['id'])}
    rows = []
    for index, sku in enumerate(SELECTED):
        product = source[sku]
        folder = target / 'references' / sku
        folder.mkdir(parents=True)
        references = []
        for number, ref in enumerate(product['references']):
            original = Path(ref['image_path'])
            dest = folder / f'{number}{original.suffix}'
            shutil.copy2(original, dest)
            references.append(str(dest))
        db.add_product(business['id'], sku, product['name'], references[0], references)
        saved = db.get_product_by_sku(business['id'], sku)
        shelf = 'A2' if sku.startswith('DUKAT') else 'B1' if sku.startswith('MAKS') else 'A1'
        db.set_inventory_count(business['id'], locations[shelf], saved['id'], 12)
        rows.append(dict(sku=sku, name=product['name'], gtin='', unit='one retail package',
                         price=round(1.5 + index * .35, 2), shelf_id=shelf, initial_quantity=12,
                         identity_status='check exact Fripsy flavour and size' if sku == 'MAKS-CRISPY-50' else 'existing catalog identity'))
    with (target / 'demo_products.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shelf_rows = [dict(zip(['shelf_id', 'label', 'x', 'y', 'width', 'height'], shelf)) for shelf in shelves]
    fixture = {'demo_data': True, 'shop': 'Web&Shop Corner Market', 'products': rows, 'shelves': shelf_rows,
               'note': 'Prices, opening quantities and map positions are illustrative. References are copied from the existing catalog. Fripsy variant needs owner verification.'}
    (target / 'demo_shop.json').write_text(json.dumps(fixture, indent=2))
    write_workbook(target / 'demo_catalog.xlsx', [('Products', [list(rows[0])] + [list(row.values()) for row in rows]),
        ('Shelves', [list(shelf_rows[0])] + [list(row.values()) for row in shelf_rows])])
    print(json.dumps({'demo_data_dir': str(target), 'products': len(rows), 'workbook': str(target / 'demo_catalog.xlsx')}))


if __name__ == '__main__':
    main()
