"""Current shop Excel snapshots, generated in memory from the shared database."""
import io
import re
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

import db


def _text(value):
    return escape(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(value if value is not None else '')))


def _xlsx(sheets):
    output = io.BytesIO()
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' +
            ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets)+1)) + '</Types>')
        archive.writestr('_rels/.rels', f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{rel}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>' +
            ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheets, 1)) + '</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
            ''.join(f'<Relationship Id="rId{i}" Type="{rel}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets)+1)) + '</Relationships>')
        for i, (_, rows) in enumerate(sheets, 1):
            xml_rows = []
            for number, row in enumerate(rows, 1):
                cells = []
                for column, value in enumerate(row, 1):
                    letters, index = '', column
                    while index:
                        index, remainder = divmod(index-1, 26)
                        letters = chr(65+remainder) + letters
                    address = f'{letters}{number}'
                    if isinstance(value, (int, float)):
                        cells.append(f'<c r="{address}"><v>{value}</v></c>')
                    else:
                        cells.append(f'<c r="{address}" t="inlineStr"><is><t xml:space="preserve">{_text(value)}</t></is></c>')
                xml_rows.append(f'<row r="{number}">' + ''.join(cells) + '</row>')
            archive.writestr(f'xl/worksheets/sheet{i}.xml', f'<worksheet xmlns="{ns}"><sheetData>' + ''.join(xml_rows) + '</sheetData></worksheet>')
    return output.getvalue()


def export_workbook(service):
    user = service.user(owner=True)
    shop = user['business_id']
    with db.connect() as connection:
        connection.execute('BEGIN')
        products = db.rows(connection.execute('SELECT * FROM products WHERE business_id=? ORDER BY sku', (shop,)))
        shelves = db.rows(connection.execute('''SELECT l.name AS shelf_id,s.zone,s.aisle,s.map_x,s.map_y
            FROM inventory_locations l LEFT JOIN shop_shelves s ON s.business_id=l.business_id AND s.shelf_id=l.name
            WHERE l.business_id=? ORDER BY l.name''', (shop,)))
        counts = db.rows(connection.execute('''SELECT p.sku,p.name,l.name AS shelf_id,i.quantity,i.updated_at
            FROM inventory_counts i JOIN products p ON p.id=i.product_id AND p.business_id=i.business_id
            JOIN inventory_locations l ON l.id=i.location_id AND l.business_id=i.business_id
            WHERE i.business_id=? ORDER BY p.sku,l.name''', (shop,)))
        references = dict(connection.execute('''SELECT p.id,COUNT(r.id) FROM products p
            LEFT JOIN product_references r ON r.product_id=p.id WHERE p.business_id=? GROUP BY p.id''', (shop,)).fetchall())
    current_stock={(row['sku'],row['shelf_id']):row['quantity'] for row in counts}
    fields = ['sku','name','brand','variant','package_size','unit','price','initial_quantity','current_stock','shelf_id','barcode','description','registration_status','reference_count']
    product_rows = []
    for product in products:
        row={**product,'initial_quantity':product.get('opening_stock'),
             'current_stock':current_stock.get((product['sku'],product.get('shelf_id'))),
             'reference_count':references[product['id']]}
        product_rows.append([row.get(k) for k in fields])
    shelf_fields = ['shelf_id','zone','aisle','map_x','map_y']
    stock_fields = ['sku','name','shelf_id','quantity','updated_at']
    return _xlsx([
        ('Products', [fields] + product_rows),
        ('Shelves', [shelf_fields] + [[s[k] for k in shelf_fields] for s in shelves]),
        ('Inventory', [stock_fields] + [[r[k] for k in stock_fields] for r in counts]),
        ('Readme', [['Shop ID',shop],['Generated UTC',db.now()],
                    ['Source','Current shared shop database; generated again after saved changes.'],
                    ['Stock','Products.initial_quantity is original opening stock; current_stock is the confirmed balance at the assigned shelf. Inventory lists all confirmed shelf balances. Missing balances are uncounted.'],
                    ['Download','This file is a snapshot. Download again after approving stock changes to get updated values.'],
                    ['Recognition','References are registered separately in the app. Spreadsheet rows alone do not enable recognition.']]),
    ])


def export_scan_workbook(service, draft, preview, operation):
    """Employee-visible review snapshot; downloading never applies a stock action."""
    user = service.user()
    products = {p['sku']:p for p in service.products()}
    fields = ['box_id','sku','product','quantity','reviewed','excluded']
    rows = []
    for item in draft['rows']:
        sku = item.get('sku') or ''
        if sku and sku not in products:
            raise PermissionError('Unknown current-shop SKU.')
        rows.append([item['item_id'],sku,products[sku]['name'] if sku else 'Unknown',
                     item['quantity'],str(bool(item['confirmed'])),str(bool(item['exclude']))])
    stock_fields = ['sku']+(['shelf_id'] if any('shelf_id' in r for r in preview) else [])+['before','reviewed_units','after']
    if any(row['sku'] not in products for row in preview):
        raise PermissionError('Unknown current-shop SKU.')
    return _xlsx([
        ('Scan review',[fields]+rows),
        ('Stock preview',[stock_fields]+[[r[k] for k in stock_fields] for r in preview]),
        ('Readme',[['Shop ID',user['business_id']],['Action',operation],
                   ['Status','Preview only. Downloading this workbook does not change inventory.']]),
    ])


def workbook_panel(service, key):
    import streamlit as st
    from shop_catalog import read_catalog_bundle
    data = export_workbook(service)
    with st.expander('Current Excel sheets', expanded=False):
        st.caption('Current stock updates after approved ADD/SUBTRACT actions. Opening stock stays unchanged. Download a fresh workbook after changes; previously downloaded files are snapshots.')
        rows=read_catalog_bundle(data,'current.xlsx')['products']
        st.dataframe([{'SKU':p['sku'],'Product':p['name'],'Price':p['price'],
                       'Opening stock':p['initial_quantity'],'Current stock':p['current_stock'],
                       'Shelf':p['shelf_id'],'Registration status':p['registration_status']} for p in rows],hide_index=True,width='stretch')
        st.download_button('Download current Excel workbook', data, 'webshop_current_catalogue.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key=key)
