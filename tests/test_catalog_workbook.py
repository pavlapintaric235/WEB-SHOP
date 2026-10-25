"""Excel reflects saved shop data without writing a second catalogue."""
import io
import unittest
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import db
from catalog_workbook import export_workbook
from registration_8509 import init_registration
from shop_catalog import read_catalog_bundle, validate_shelves
from tests import test_workflow as workflow


class CatalogWorkbookTests(unittest.TestCase):
    setUp = workflow.WorkflowTests.setUp

    def test_current_metadata_stock_and_literal_strings(self):
        init_registration()
        with db.connect() as connection:
            connection.execute('''UPDATE products SET name=?,price=2.75,brand='New brand',variant='Blue',
                package_size='500 g',opening_stock=4,shelf_id='Main Shelf',barcode='12345678',description='Demo'
                WHERE business_id=? AND sku='A' ''', ('=HYPERLINK("private")', self.business['id']))
        first = export_workbook(self.owner)
        bundle = read_catalog_bundle(first, 'current.xlsx')
        product = next(p for p in bundle['products'] if p['sku']=='A')
        self.assertEqual(product['price'], '2.75')
        self.assertEqual(product['initial_quantity'], '4')
        self.assertEqual(product['variant'], 'Blue')
        self.assertEqual(product['reference_count'], '1')
        self.assertEqual(product['name'], '=HYPERLINK("private")')
        self.assertEqual(len(validate_shelves(bundle['shelves'])), 1)
        self.employee.approve(self.location, 'query-photo', 'unique-confirmation', {'A':7})
        second = export_workbook(self.owner)
        self.assertNotEqual(first, second)
        exported=next(p for p in read_catalog_bundle(second,'current.xlsx')['products'] if p['sku']=='A')
        self.assertEqual(exported['initial_quantity'], '4')
        self.assertEqual(exported['current_stock'],'7')
        with ZipFile(io.BytesIO(second)) as archive:
            for name in archive.namelist():
                if name.endswith('.xml'):
                    root = ET.fromstring(archive.read(name))
                    self.assertFalse(root.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}f'))
            stock = archive.read('xl/worksheets/sheet3.xml').decode()
            self.assertIn('<v>7</v>', stock)
            self.assertNotIn('image_path', stock)

    def test_owner_only_and_shop_isolation(self):
        init_registration()
        with self.assertRaises(PermissionError):
            export_workbook(self.employee)
        other = db.get_or_create_business('Other shop')
        with db.connect() as connection:
            connection.execute('INSERT INTO products(business_id,sku,name,image_path,created_at) VALUES (?,?,?,?,?)',
                               (other['id'],'SECRET','Private other shop','',db.now()))
        rows = read_catalog_bundle(export_workbook(self.owner), 'current.xlsx')['products']
        self.assertEqual({p['sku'] for p in rows}, {'A','B'})

    def test_employee_scan_export_is_read_only_and_shop_scoped(self):
        from catalog_workbook import export_scan_workbook
        from scan_review import new_draft
        draft=new_draft([dict(sku='A',box=[0,0,10,10]),dict(sku='',box=[10,0,20,10])])
        before=self.owner.history()
        data=export_scan_workbook(self.employee,draft,[], 'review')
        with ZipFile(io.BytesIO(data)) as archive:
            self.assertIn('Unknown',archive.read('xl/worksheets/sheet1.xml').decode())
            self.assertIn('Preview only',archive.read('xl/worksheets/sheet3.xml').decode())
        self.assertEqual(self.owner.history(),before)
        draft['rows'][0]['sku']='OTHER-SHOP'
        with self.assertRaises(PermissionError):
            export_scan_workbook(self.employee,draft,[],'add')
