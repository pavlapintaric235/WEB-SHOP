"""Bounded spreadsheet import and public, read-only catalogue access."""
import csv
import io
import math
import re
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import db


def _xlsx_tables(data):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(io.BytesIO(data)) as archive:
        if sum(i.file_size for i in archive.infolist()) > 10 * 1024 * 1024:
            raise ValueError('Expanded workbook is too large.')
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        rels = {r.get('Id'): r.get('Target') for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        tables = {}
        for sheet in workbook.findall('s:sheets/s:sheet', ns):
            name = sheet.get('name')
            if name not in {'Products', 'Products_Import', 'Shelves', 'Shelf_Locations'}:
                continue  # Workbook instructions and simulated carts are not import data.
            rid = sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            target = rels[rid]
            path = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            matrix = []
            for row in ET.fromstring(archive.read(path)).findall('s:sheetData/s:row', ns):
                cells = {}
                for cell in row:
                    letters = re.match(r'[A-Z]+', cell.get('r', 'A1')).group()
                    index = 0
                    for letter in letters:
                        index = index * 26 + ord(letter) - 64
                    value = cell.findtext('s:v', '', ns)
                    if cell.get('t') == 's':
                        value = strings[int(value)]
                    elif cell.get('t') == 'inlineStr':
                        value = ''.join(n.text or '' for n in cell.findall('s:is//s:t', ns))
                    if cell.find('s:f', ns) is not None:
                        raise ValueError(f'{name}: use values, not spreadsheet formulas.')
                    cells[index - 1] = value
                matrix.append(cells)
            header = next((i for i, row in enumerate(matrix)
                           if any(str(v).strip().lower() in {'sku', 'shelf_id'} for v in row.values())), None)
            if header is None:
                raise ValueError(f'{name}: no SKU / Shelf_ID header found.')
            headers = {i: str(v).strip().lower() for i, v in matrix[header].items() if str(v).strip()}
            tables[name] = [{key: row.get(i, '') for i, key in headers.items()}
                            for row in matrix[header + 1:] if any(str(v).strip() for v in row.values())]
        return tables


def read_catalog_bundle(data, filename):
    if len(data) > 2 * 1024 * 1024:
        raise ValueError('Catalogue must be at most 2 MiB.')
    if filename.lower().endswith('.csv'):
        return {'products': list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')))),
                'shelves': [], 'fictional': False}
    tables = _xlsx_tables(data)
    source = tables.get('Products_Import', tables.get('Products'))
    if source is None:
        raise ValueError('Workbook needs a Products or Products_Import sheet.')
    aliases = {'product_name': 'name', 'count_unit': 'unit', 'price_eur': 'price',
               'opening_stock': 'initial_quantity'}
    products = []
    for row in source:
        mapped = {aliases.get(k, k): v for k, v in row.items()}
        if row.get('brand') or row.get('variant_or_size'):
            mapped['brand_variant_size'] = ' '.join(str(row.get(k, '')) for k in ('brand', 'variant_or_size')).strip()
        if 'Products_Import' in tables:
            mapped['registration_status'] = 'pending identity verification'
        products.append(mapped)
    shelves = tables.get('Shelf_Locations', tables.get('Shelves', []))
    return {'products': products, 'shelves': shelves, 'fictional': 'Products_Import' in tables}


def read_catalog(data, filename):
    return read_catalog_bundle(data, filename)['products']


def validate_shelves(records):
    clean, seen = [], set()
    for row in records:
        shelf = str(row.get('shelf_id', '')).strip()
        if not shelf or len(shelf) > 80 or shelf in seen:
            raise ValueError('Shelf IDs must be nonempty, unique and at most 80 characters.')
        seen.add(shelf)
        values = {}
        for key in ('map_x', 'map_y'):
            value = row.get(key)
            if value is None or str(value).strip() == '':
                values[key] = None
            else:
                number = float(value)
                if not math.isfinite(number) or number != int(number) or not 0 <= number < 40:
                    raise ValueError('Shelf map coordinates must be whole numbers from 0 to 39.')
                values[key] = int(number)
        clean.append(dict(shelf_id=shelf, zone=str(row.get('zone', '')), aisle=str(row.get('aisle', '')), **values))
    occupied = [(r['map_x'], r['map_y']) for r in clean if r['map_x'] is not None and r['map_y'] is not None]
    if len(occupied) != len(set(occupied)):
        raise ValueError('Two shelves occupy the same map cell.')
    return clean


def validate_catalog(records):
    if not records or len(records) > 1000:
        raise ValueError('Import 1–1000 products.')
    clean, seen = [], set()
    for row in records:
        if None in row:
            raise ValueError('Unexpected extra spreadsheet cells.')
        row = {k: str('' if v is None else v).strip() for k, v in row.items()}
        sku = row.get('sku', '').upper()
        if not re.fullmatch(r'[A-Z0-9_-]{1,40}', sku) or sku in seen or not row.get('name'):
            raise ValueError('Each row needs a unique valid SKU and product name.')
        seen.add(sku)
        price = float(row['price']) if row.get('price') else None
        if price is not None and (not math.isfinite(price) or price < 0):
            raise ValueError('Prices must be finite and nonnegative.')
        quantity = None
        if row.get('initial_quantity'):
            quantity = float(row['initial_quantity'])
            if not math.isfinite(quantity) or quantity != int(quantity):
                raise ValueError('Opening stock must be a whole number.')
            quantity = int(quantity)
            db.validate_quantity(quantity)
            if not row.get('shelf_id'):
                raise ValueError('Opening stock needs a shelf_id.')
        # Uncertain package mappings remain pending even when a spreadsheet calls them confirmed.
        status = row.get('registration_status', 'pending')
        if sku == 'MAKS-CRISPY-50' or 'check' in row.get('identity_status', '').lower():
            status = 'pending identity verification'
        clean.append(dict(sku=sku, name=row['name'], price=price, unit=row.get('unit') or 'pending',
                          shelf_id=row.get('shelf_id') or None, initial_quantity=quantity,
                          registration_status=status, brand_variant_size=row.get('brand_variant_size', '')))
    return clean


def import_catalog(business_id, records, confirmed=False, opening=False, shelves=(), fictional=False):
    clean = validate_catalog(records)
    shelf_rows = validate_shelves(shelves)
    if fictional and not db.get_business(business_id).get('is_demo'):
        raise ValueError('Import fictional workbook data into an isolated demo shop only.')
    if shelf_rows:
        known = {r['shelf_id'] for r in shelf_rows}
        if any(r['shelf_id'] and r['shelf_id'] not in known for r in clean):
            raise ValueError('A product refers to a shelf absent from Shelf_Locations.')
    if fictional:
        for row in clean:
            row['registration_status'] = 'pending identity verification'
    if not confirmed:
        raise ValueError('Confirm the catalogue preview before importing.')
    with db.connect() as connection:
        for shelf in shelf_rows:
            connection.execute('INSERT INTO shop_shelves VALUES (?,?,?,?,?,?) ON CONFLICT(business_id,shelf_id) DO UPDATE SET zone=excluded.zone,aisle=excluded.aisle,map_x=excluded.map_x,map_y=excluded.map_y',
                               (business_id,) + tuple(shelf[k] for k in ('shelf_id','zone','aisle','map_x','map_y')))
            connection.execute('INSERT OR IGNORE INTO inventory_locations (business_id,name,created_at) VALUES (?,?,?)',
                               (business_id, shelf['shelf_id'], db.now()))
        for row in clean:
            connection.execute('INSERT INTO products (business_id,sku,name,image_path,created_at) VALUES (?,?,?,?,?) '
                               'ON CONFLICT(business_id,sku) DO NOTHING',
                               (business_id, row['sku'], row['name'], '', db.now()))
            connection.execute('UPDATE products SET name=?,price=?,unit=?,shelf_id=?,registration_status=?,brand_variant_size=? '
                               'WHERE business_id=? AND sku=?',
                               tuple(row[k] for k in ('name', 'price', 'unit', 'shelf_id', 'registration_status', 'brand_variant_size')) + (business_id, row['sku']))
            if row['initial_quantity'] is not None:
                connection.execute('UPDATE products SET opening_stock=? WHERE business_id=? AND sku=?',
                                   (row['initial_quantity'], business_id, row['sku']))
            if row['shelf_id']:
                connection.execute('INSERT OR IGNORE INTO inventory_locations (business_id,name,created_at) VALUES (?,?,?)',
                                   (business_id, row['shelf_id'], db.now()))
            if opening and row['initial_quantity'] is not None:
                location = connection.execute('SELECT id FROM inventory_locations WHERE business_id=? AND name=?', (business_id, row['shelf_id'])).fetchone()['id']
                product = connection.execute('SELECT id FROM products WHERE business_id=? AND sku=?', (business_id, row['sku'])).fetchone()['id']
                # Opening balances can only initialize untouched scopes. Never overwrite live stock.
                if connection.execute('SELECT 1 FROM inventory_counts WHERE business_id=? AND location_id=? AND product_id=?', (business_id, location, product)).fetchone():
                    raise ValueError('Opening stock already exists; use a reviewed stock action.')
                db._set_inventory_count(connection, business_id, location, product, row['initial_quantity'])
    return len(clean)


class BuyerCatalog:
    """Buyer capability exposes shared products and shelves, with no stock writer."""
    def __init__(self, business_id):
        if not db.get_business(business_id):
            raise ValueError('Unknown shop.')
        self.business_id = business_id

    def user(self):
        return {'id': f'buyer-{self.business_id}', 'business_id': self.business_id, 'role': 'buyer', 'username': 'buyer'}

    def products(self):
        return db.get_business_products(self.business_id)

    def locations(self):
        return db.get_locations(self.business_id)


def shopping_list(products, quantities):
    allowed = {p['sku']: p for p in products}
    result = []
    for sku, quantity in quantities.items():
        if sku not in allowed:
            raise ValueError('Unknown shop SKU.')
        db.validate_quantity(quantity)
        if quantity:
            p = allowed[sku]
            result.append(dict(sku=sku, name=p['name'], quantity=quantity, price=p['price'], shelf_id=p['shelf_id']))
    return result
