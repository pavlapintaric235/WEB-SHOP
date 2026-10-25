"""Mixed-shelf actions stay atomic, scoped, and idempotent."""
import unittest
import db
from tests import test_workflow as workflow


class CatalogStockTests(unittest.TestCase):
    setUp=workflow.WorkflowTests.setUp

    def prepare_shelves(self):
        self.owner.add_location('Second Shelf')
        second=next(l['id'] for l in self.owner.locations() if l['name']=='Second Shelf')
        with db.connect() as c:
            c.execute("UPDATE products SET shelf_id='Main Shelf' WHERE business_id=? AND sku='A'",(self.business['id'],))
            c.execute("UPDATE products SET shelf_id='Second Shelf' WHERE business_id=? AND sku='B'",(self.business['id'],))
        self.employee.approve(self.location,'initial-a','initial-a',{'A':12})
        self.employee.approve(second,'initial-b','initial-b',{'B':12})
        return second

    def test_removes_from_assigned_shelves_and_retry_is_idempotent(self):
        second=self.prepare_shelves()
        counts={'A':3,'B':2}
        preview=self.employee.preview_catalog_action(counts,'subtract')
        self.assertEqual([(r['shelf_id'],r['before'],r['after']) for r in preview],[('Main Shelf',12,9),('Second Shelf',12,10)])
        receipt=self.employee.approve_catalog_action('fresh-photo','mixed-scan',counts,operation='subtract',expected_preview=preview)
        self.assertEqual(len(receipt),2)
        self.assertEqual(self.employee.approve_catalog_action('fresh-photo','mixed-scan',counts,operation='subtract',expected_preview=preview),receipt)
        self.assertEqual(next(r['quantity'] for r in self.owner.inventory(self.location) if r['sku']=='A'),9)
        self.assertEqual(next(r['quantity'] for r in self.owner.inventory(second) if r['sku']=='B'),10)
        with self.assertRaises(ValueError):self.employee.approve_catalog_action('fresh-photo','mixed-scan',{'A':4,'B':2},operation='subtract')

    def test_failed_second_shelf_rolls_back_first_shelf_and_receipts(self):
        self.prepare_shelves()
        before=self.owner.history()
        with db.connect() as c:
            c.execute("CREATE TRIGGER fail_second_receipt BEFORE INSERT ON scan_sessions WHEN NEW.submission_key LIKE 'atomic-fail:shelf:%' BEGIN SELECT RAISE(ABORT,'simulated failure'); END")
        with self.assertRaises(Exception):
            self.employee.approve_catalog_action('photo','atomic-fail',{'A':1,'B':1},operation='subtract')
        self.assertEqual(self.owner.history(),before)
        self.assertEqual(next(r['quantity'] for r in self.owner.inventory(self.location) if r['sku']=='A'),12)

    def test_no_negative_stock_and_stale_preview_requires_confirmation(self):
        self.prepare_shelves()
        before=self.owner.history()
        with self.assertRaises(ValueError):self.employee.approve_catalog_action('p','too-many',{'A':1,'B':13},operation='subtract')
        self.assertEqual(self.owner.history(),before)
        preview=self.employee.preview_catalog_action({'A':1},'subtract')
        self.employee.approve(self.location,'other','other',{'A':1},operation='add')
        with self.assertRaisesRegex(ValueError,'changed'):
            self.employee.approve_catalog_action('p','stale',{'A':1},operation='subtract',expected_preview=preview)

    def test_invalid_shelf_and_unknown_shop_product_are_rejected(self):
        self.prepare_shelves()
        with self.assertRaises(PermissionError):self.employee.preview_catalog_action({'OTHER-SHOP':1},'subtract')
        with db.connect() as c:c.execute("UPDATE products SET shelf_id='Missing Shelf' WHERE business_id=? AND sku='B'",(self.business['id'],))
        with self.assertRaisesRegex(ValueError,'valid shelf'):
            self.employee.preview_catalog_action({'B':1},'subtract')
