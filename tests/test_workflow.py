"""Business boundaries and atomic inventory updates, using isolated databases."""

import csv
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import auth
import db
from images import read_image
from inventory import InventoryService


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for field, value in (
            ("DATA_DIR", self.root), ("DB_FILE", self.root / "test.sqlite3"),
            ("CATALOG_FILE", self.root / "catalog.json"),
        ):
            patched = patch.object(db, field, value)
            patched.start()
            self.addCleanup(patched.stop)
        db.init_db()
        self.business = auth.seed_demo_users()
        self.location = db.default_location(self.business["id"])["id"]
        self.owner_id = auth.authenticate("owner", "owner123")["id"]
        self.employee_id = auth.authenticate("employee", "employee123")["id"]
        self.owner = InventoryService(self.owner_id)
        self.employee = InventoryService(self.employee_id)
        self.image = Image.new("RGB", (64, 96), "red")
        self.owner.register_product("A", "Product A", self.image)
        self.owner.register_product("B", "Product B", self.image)

    def approve(self, counts, key="scan1", **options):
        return self.employee.approve(self.location, "photo_hash", key, counts, **options)

    def quantities(self, location=None):
        return {item["sku"]: item["quantity"] for item in self.owner.inventory(location or self.location)}

    def test_registration_persistence_and_duplicate_sku(self):
        products = InventoryService(self.owner_id).products()
        self.assertEqual([item["sku"] for item in products], ["A", "B"])
        self.assertTrue(Path(products[0]["image_path"]).is_file())
        with self.assertRaises(ValueError):
            self.owner.register_product("a", "Duplicate", self.image)
        self.owner.update_product("A", "Renamed")
        self.assertEqual(self.owner.products()[0]["name"], "Renamed")

    def test_multiple_references_persist_under_one_sku(self):
        blue = Image.new("RGB", self.image.size, "blue")
        self.owner.register_product("MULTI", "Multi-view product", [self.image, blue])
        product = next(item for item in InventoryService(self.owner_id).products() if item["sku"] == "MULTI")
        self.assertEqual(len(product["references"]), 2)
        self.assertEqual(product["image_path"], product["references"][0]["image_path"])
        self.assertTrue(all(Path(item["image_path"]).is_file() for item in product["references"]))
        self.assertEqual(len(self.employee.products()), 3)

    def test_add_references_keeps_primary_and_existing_inventory(self):
        original = self.owner.products()[0]["image_path"]
        self.approve({"A": 4})
        self.assertEqual(self.owner.add_references("a", [Image.new("RGB", (50, 80), "blue")]), 1)
        product = self.employee.products()[0]
        self.assertEqual(product["image_path"], original)
        self.assertEqual(len(product["references"]), 2)
        self.assertEqual(self.quantities(), {"A": 4, "B": None})

    def test_duplicate_and_invalid_uploads_do_not_partially_add_references(self):
        before = list((self.root / "references").rglob("*.png"))
        for images in ([], [None], [self.image], [Image.new("RGB", (50, 80), "blue"), self.image]):
            with self.subTest(images=images), self.assertRaises(ValueError):
                self.owner.add_references("A", images)
        self.assertEqual(len(self.owner.products()[0]["references"]), 1)
        self.assertEqual(list((self.root / "references").rglob("*.png")), before)
        with self.assertRaises(ValueError):
            self.owner.register_product("C", "Duplicate views", [self.image, self.image])
        self.assertEqual(len(self.owner.products()), 2)

    def test_reference_limit_is_enforced_without_partial_writes(self):
        views = [Image.new("RGB", (50, 80), (index, 0, 0)) for index in range(db.MAX_PRODUCT_REFERENCES - 1)]
        self.owner.add_references("A", views)
        before = list((self.root / "references").rglob("*.png"))
        with self.assertRaises(ValueError):
            self.owner.add_references("A", [Image.new("RGB", (50, 80), "blue")])
        with self.assertRaises(ValueError):
            db.add_product_references(self.business["id"], "A", ["extra.png"])
        self.assertEqual(len(self.owner.products()[0]["references"]), db.MAX_PRODUCT_REFERENCES)
        self.assertEqual(list((self.root / "references").rglob("*.png")), before)

    def test_registration_accepts_four_views_and_rejects_five(self):
        self.assertEqual(db.MAX_PRODUCT_REFERENCES, 4)
        views = [Image.new("RGB", (50, 80), (index, 0, 0)) for index in range(5)]
        self.owner.register_product("FOUR", "Four views", views[:4])
        product = next(item for item in self.owner.products() if item["sku"] == "FOUR")
        self.assertEqual(len(product["references"]), 4)
        before = set((self.root / "references").rglob("*.png"))
        with self.assertRaises(ValueError):
            self.owner.register_product("FIVE", "Too many views", views)
        paths = [f"view-{index}.png" for index in range(5)]
        with self.assertRaises(ValueError):
            db.add_product(self.business["id"], "FIVE", "Too many views", paths[0], paths)
        self.assertIsNone(db.get_product_by_sku(self.business["id"], "FIVE"))
        self.assertEqual(set((self.root / "references").rglob("*.png")), before)

    def test_removing_primary_promotes_another_view_and_preserves_historic_file(self):
        original = self.owner.products()[0]["references"][0]
        self.owner.add_references("A", [Image.new("RGB", (50, 80), "blue")])
        replacement = self.owner.products()[0]["references"][1]
        self.owner.remove_reference("A", original["id"])
        db.init_db()
        product = self.owner.products()[0]
        self.assertEqual(product["image_path"], replacement["image_path"])
        self.assertEqual(product["references"], [replacement])
        self.assertTrue(Path(original["image_path"]).is_file())
        with self.assertRaises(ValueError):
            self.owner.remove_reference("A", replacement["id"])

    def test_reference_ids_cannot_cross_product_or_business_boundaries(self):
        self.owner.add_references("A", [Image.new("RGB", (50, 80), "blue")])
        other = db.get_or_create_business("Other")
        db.add_product(other["id"], "A", "Other A", "private.png")
        foreign_id = db.get_business_products(other["id"])[0]["references"][0]["id"]
        for reference_id in (self.owner.products()[1]["references"][0]["id"], foreign_id):
            with self.subTest(reference_id=reference_id), self.assertRaises(ValueError):
                self.owner.remove_reference("A", reference_id)
        with self.assertRaises(ValueError):
            self.owner.add_references("PRIVATE", [self.image])
        self.assertEqual(len(self.owner.products()[0]["references"]), 2)

    def test_reference_backfill_is_idempotent_and_preserves_original_paths(self):
        originals = {item["sku"]: item["image_path"] for item in self.owner.products()}
        with db.connect() as connection:
            connection.execute("DROP TABLE product_references")
        db.init_db()
        db.init_db()
        for item in self.owner.products():
            self.assertEqual(item["image_path"], originals[item["sku"]])
            self.assertEqual([view["image_path"] for view in item["references"]], [originals[item["sku"]]])

    def test_existing_startup_preserves_database_bytes_and_sequences(self):
        before = db.DB_FILE.read_bytes()
        db.init_db()
        db.default_location(self.business['id'])
        db.get_or_create_business(self.business['name'])
        auth.seed_demo_users()
        self.assertEqual(db.DB_FILE.read_bytes(), before)

    def test_failed_reference_batch_rolls_back_database_and_new_files(self):
        product_id = self.owner.products()[0]["id"]
        with db.connect() as connection:
            connection.execute(
                f"CREATE TRIGGER fail_reference BEFORE INSERT ON product_references "
                f"WHEN NEW.product_id = {product_id} AND "
                f"(SELECT COUNT(*) FROM product_references WHERE product_id = {product_id}) >= 2 "
                "BEGIN SELECT RAISE(ABORT, 'test failure'); END"
            )
        before = list((self.root / "references").rglob("*.png"))
        with self.assertRaises(sqlite3.IntegrityError):
            self.owner.add_references("A", [Image.new("RGB", (50, 80), "blue"),
                                            Image.new("RGB", (50, 80), "green")])
        self.assertEqual(len(self.owner.products()[0]["references"]), 1)
        self.assertEqual(list((self.root / "references").rglob("*.png")), before)

    def test_authentication_and_salted_hashes(self):
        first, second = auth.hash_password("password123"), auth.hash_password("password123")
        self.assertNotEqual(first, second)
        self.assertTrue(auth.verify_password("password123", first))
        self.assertFalse(auth.verify_password("wrong", first))
        self.assertFalse(auth.verify_password("wrong", "broken"))
        self.assertIsNone(auth.authenticate("owner", "incorrect"))
        self.assertNotIn("password_hash", auth.authenticate("owner", "owner123"))

    def test_owner_creates_employee(self):
        self.owner.add_employee("new.employee", "password123")
        employee = auth.authenticate("new.employee", "password123")
        self.assertEqual(employee["role"], "employee")
        self.assertEqual(employee["business_id"], self.business["id"])

    def test_employee_email_login_normalization_and_duplicates(self):
        address = "Demo.Employee+video@Example.org"
        self.owner.add_employee("  " + address + "  ", "password123")
        employee = auth.authenticate("  " + address.upper() + "  ", "password123")
        self.assertEqual(employee["username"], address.lower())
        self.assertEqual(employee["role"], "employee")
        self.assertEqual(employee["business_id"], self.business["id"])
        self.assertIsNone(auth.authenticate(address, "wrong-password"))
        with self.assertRaisesRegex(ValueError, "already taken"):
            self.owner.add_employee(address.lower(), "password123")
        long_address = "long." + "a" * 55 + "@example.org"
        self.owner.add_employee(long_address, "password123")
        self.assertIsNotNone(auth.authenticate(long_address, "password123"))

    def test_invalid_employee_emails_do_not_create_users(self):
        before = self.owner.employees()
        for address in ("a@@example.org", "a@localhost", "a b@example.org",
                        ".a@example.org", "a.@example.org", "a..b@example.org",
                        "a@-example.org", "a@example-.org", "a@example..org",
                        "a\n@example.org", "a" * 65 + "@example.org",
                        "a@" + "b" * 64 + ".org"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                self.owner.add_employee(address, "password123")
        self.assertEqual(self.owner.employees(), before)

    def test_employee_cannot_perform_owner_actions(self):
        actions = [
            lambda: self.employee.register_product("C", "Forbidden", self.image),
            lambda: self.employee.update_product("A", "Forbidden"),
            lambda: self.employee.add_references("A", [self.image]),
            lambda: self.employee.remove_reference("A", self.owner.products()[0]["references"][0]["id"]),
            lambda: self.employee.add_location("Forbidden"),
            lambda: self.employee.add_employee("forbidden", "password123"),
            lambda: self.employee.import_csv(self.location, "test.csv", "sku,quantity\nA,7\n", True),
            lambda: self.employee.export_csv(self.location),
            lambda: self.employee.history(),
        ]
        for action in actions:
            with self.subTest(action=action), self.assertRaises(PermissionError):
                action()

    def test_cross_business_read_and_write_are_rejected(self):
        other = db.get_or_create_business("Other business")
        location = db.default_location(other["id"])["id"]
        db.add_product(other["id"], "PRIVATE", "Private product", "private.png")
        self.assertEqual([item["sku"] for item in self.owner.products()], ["A", "B"])
        with self.assertRaises(PermissionError):
            self.employee.approve(location, "hash", "key", {"A": 1})
        with self.assertRaises(PermissionError):
            self.owner.inventory(location)
        with self.assertRaises(PermissionError):
            self.owner.export_csv(location)
        self.assertEqual(self.quantities(), {"A": None, "B": None})

    def test_partial_snapshot_preserves_unselected_skus_and_locations(self):
        self.owner.import_csv(self.location, "counts.csv", "sku,quantity\nA,8\nB,9\n", True)
        self.owner.add_location("Storage")
        storage = next(item["id"] for item in self.owner.locations() if item["name"] == "Storage")
        self.owner.import_csv(storage, "counts.csv", "sku,quantity\nA,12\nB,13\n", True)
        self.approve({"A": 0}, predicted_counts={"A": 1})
        self.assertEqual(self.quantities(), {"A": 0, "B": 9})
        self.assertEqual(self.quantities(storage), {"A": 12, "B": 13})
        row = self.owner.history()[0]
        self.assertEqual((row["username"], row["predicted_quantity"], row["approved_quantity"]), ("employee", 1, 0))

    def test_complete_snapshot_requires_every_sku(self):
        with self.assertRaises(ValueError):
            self.approve({"A": 2}, scope="complete")
        self.approve({"A": 2, "B": 0}, scope="complete")
        self.assertEqual(self.quantities(), {"A": 2, "B": 0})

    def test_idempotent_approval_does_not_reapply_old_counts(self):
        first = self.approve({"A": 2})
        self.approve({"A": 4}, "scan2")
        self.assertEqual(self.approve({"A": 2}), first)
        self.assertEqual(self.quantities()["A"], 4)
        self.assertEqual(len(self.owner.history()), 2)
        with self.assertRaises(ValueError):
            self.approve({"A": 7})

    def test_invalid_counts_never_write_partial_data(self):
        invalid = [{"UNKNOWN": 1}, {"A": -1}, {"A": 1.5}, {"A": True}, {"A": "2"}, {}, {"A": 2**40}]
        for counts in invalid:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.approve(counts)
        self.assertEqual(self.owner.history(), [])
        self.assertEqual(self.quantities(), {"A": None, "B": None})

    def test_failed_write_rolls_back_entire_approval(self):
        product_id = self.owner.products()[1]["id"]
        with db.connect() as connection:
            connection.execute(
                f"CREATE TRIGGER fail_count BEFORE INSERT ON inventory_counts "
                f"WHEN NEW.product_id = {product_id} BEGIN SELECT RAISE(ABORT, 'test failure'); END"
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.approve({"A": 3, "B": 4})
        self.assertEqual(self.quantities(), {"A": None, "B": None})
        self.assertEqual(self.owner.history(), [])

    def test_csv_preserves_columns_names_and_unrelated_rows(self):
        text = "sku,name,quantity,price,note\n a ,Template name,8,1.50,keep\nB,Original B,9,2.00,also keep\nZ,Unregistered,99,3.00,untouched\n"
        self.owner.import_csv(self.location, "inventory.csv", text, True)
        self.approve({"A": 3})
        exported = self.owner.export_csv(self.location).decode("utf-8-sig")
        parsed = list(csv.DictReader(io.StringIO(exported)))
        self.assertEqual(parsed[0], {"sku": " a ", "name": "Template name", "quantity": "3", "price": "1.50", "note": "keep"})
        self.assertEqual(parsed[1]["quantity"], "9")
        self.assertEqual(parsed[2], {"sku": "Z", "name": "Unregistered", "quantity": "99", "price": "3.00", "note": "untouched"})

    def test_csv_validation_and_explicit_import_confirmation(self):
        invalid = ["sku,quantity\nA,1\n a ,2\n", "sku,quantity\nA,4\nB,-1\n",
                   "sku,quantity\nA,1.5\n", "sku,name\nA,Missing\n", "sku,quantity\nA,1,extra\n",
                   "sku,quantity\n,1\n", "sku,quantity,quantity\nA,1,1\n"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.owner.import_csv(self.location, "bad.csv", text, True)
        with self.assertRaises(ValueError):
            self.owner.import_csv(self.location, "valid.csv", "sku,quantity\nA,2\n")
        self.assertIsNone(db.latest_csv_template(self.business["id"], self.location))
        self.assertEqual(self.quantities(), {"A": None, "B": None})

    def test_catalogue_migration_runs_once(self):
        path = self.root / "reference.png"
        self.image.save(path)
        db.CATALOG_FILE.write_text(json.dumps([{"sku": "LEGACY", "name": "Legacy", "image_path": str(path)}]), encoding="utf-8")
        db.migrate_catalog_json(self.business["id"])
        db.migrate_catalog_json(self.business["id"])
        self.assertEqual(len(self.owner.products()), 3)
        migrated = next(item for item in self.owner.products() if item["sku"] == "LEGACY")
        self.assertEqual(len(migrated["references"]), 1)
        self.assertTrue(db.CATALOG_FILE.exists())

    def test_image_validation_and_normalization(self):
        for upload in [None, io.BytesIO(b"not an image"), io.BytesIO(b"\x89PNG")]:
            with self.subTest(upload=upload), self.assertRaises(ValueError):
                read_image(upload)
        buffer = io.BytesIO()
        Image.new("RGBA", (9, 15), "red").save(buffer, format="PNG")
        self.assertEqual((read_image(buffer).size, read_image(buffer).mode), ((9, 15), "RGB"))

    def test_completed_legacy_migration_does_not_restore_cleared_products(self):
        path = self.root / "legacy.png"
        self.image.save(path)
        db.CATALOG_FILE.write_text(json.dumps([
            {"sku": "LEGACY", "name": "Old demo product", "image_path": str(path)}
        ]), encoding="utf-8")
        db.migrate_catalog_json(self.business["id"])
        with db.connect() as connection:
            connection.execute("DELETE FROM products WHERE business_id = ?", (self.business["id"],))
        db.init_db()
        auth.seed_demo_users()
        db.migrate_catalog_json(self.business["id"])
        self.assertEqual(self.owner.products(), [])
        self.assertEqual(self.employee.products(), [])
        self.assertEqual(self.owner.export_csv(self.location).decode("utf-8-sig"), "sku,name,quantity\r\n")


class AppTests(unittest.TestCase):
    setUp = WorkflowTests.setUp

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()

    def login(self, app, username, password):
        app.text_input(key="login_username").input(username)
        app.text_input(key="login_password").input(password)
        next(button for button in app.button if button.label == "Sign in").click().run()
        self.assertEqual(len(app.exception), 0)

    def test_zero_expected_counts_are_not_scored_without_negative_confirmation(self):
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        upload = io.BytesIO()
        self.image.save(upload, format="PNG")

        with patch("streamlit.file_uploader", return_value=upload), \
                patch("vision.load_model", return_value=object()), \
                patch("vision.detect_products", return_value=([], self.image, self.image, [])):
            app = self.app()
            self.login(app, "owner", "owner123")
            next(item for item in app.radio if item.label == "Workspace").set_value("Scanner").run()
            next(item for item in app.expander if item.label == "Test tracker").proto.expanded = True
            app.run()
            next(item for item in app.checkbox if item.label == "Record expected counts for this test").check().run()
            self.assertTrue(any("All expected quantities are zero" in item.value for item in app.warning))
            next(button for button in app.button if button.label == "Run real AI detection").click().run()

            snapshot = json.loads((self.root / "last_detection.json").read_text(encoding="utf-8"))
            self.assertIsNone(snapshot["expected_counts"])
            self.assertEqual(snapshot["comparison"], [])

    def test_confirmed_negative_scene_records_zero_expected_counts(self):
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        upload = io.BytesIO()
        self.image.save(upload, format="PNG")

        with patch("streamlit.file_uploader", return_value=upload), \
                patch("vision.load_model", return_value=object()), \
                patch("vision.detect_products", return_value=([], self.image, self.image, [])):
            app = self.app()
            self.login(app, "owner", "owner123")
            next(item for item in app.radio if item.label == "Workspace").set_value("Scanner").run()
            next(item for item in app.expander if item.label == "Test tracker").proto.expanded = True
            app.run()
            next(item for item in app.checkbox if item.label == "Record expected counts for this test").check().run()
            next(item for item in app.checkbox if item.label.startswith("Confirm this is an intentional test")).check().run()
            next(button for button in app.button if button.label == "Run real AI detection").click().run()

            snapshot = json.loads((self.root / "last_detection.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["expected_counts"], {"A": 0, "B": 0})
            self.assertTrue(all(row["correct"] for row in snapshot["comparison"]))

    def test_named_test_without_expected_counts_is_marked_unscored(self):
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        upload = io.BytesIO()
        self.image.save(upload, format="PNG")

        with patch("streamlit.file_uploader", return_value=upload), \
                patch("vision.load_model", return_value=object()), \
                patch("vision.detect_products", return_value=([], self.image, self.image, [])):
            app = self.app()
            self.login(app, "owner", "owner123")
            next(item for item in app.radio if item.label == "Workspace").set_value("Scanner").run()
            next(item for item in app.expander if item.label == "Test tracker").proto.expanded = True
            app.run()
            next(item for item in app.text_input if item.label == "Test name").input("Unscored example").run()
            self.assertTrue(any("without expected counts and will not be scored" in item.value for item in app.info))
            next(button for button in app.button if button.label == "Run real AI detection").click().run()

            snapshot = json.loads((self.root / "last_detection.json").read_text(encoding="utf-8"))
            self.assertIsNone(snapshot["expected_counts"])
            self.assertEqual(snapshot["comparison"], [])

    def test_owner_login_dashboard_and_logout(self):
        app = self.app()
        self.login(app, "owner", "owner123")
        self.assertEqual([tab.label for tab in app.tabs], ["Inventory", "Products", "Team"])
        self.assertTrue(any(button.label == "Save reference" for button in app.button))
        next(button for button in app.button if button.label == "Sign out").click().run()
        self.assertTrue(any(item.label == "Password" for item in app.text_input))
        self.assertFalse(any(button.label == "Save reference" for button in app.button))

    def test_wrong_password_does_not_open_dashboard(self):
        app = self.app()
        self.login(app, "owner", "incorrect")
        self.assertTrue(any("Incorrect" in item.value for item in app.error))
        self.assertEqual(len(app.tabs), 0)

    def test_owner_adds_reference_and_removes_it_through_gallery(self):
        buffer = io.BytesIO()
        Image.new("RGB", (50, 80), "blue").save(buffer, format="PNG")

        def uploads(label, **kwargs):
            return [buffer] if label == "Additional reference photographs" else None if label == "CSV with sku and quantity" else []

        with patch("streamlit.file_uploader", side_effect=uploads):
            app = self.app()
            self.login(app, "owner", "owner123")
            next(button for button in app.button if button.label == "Add references").click().run()
            self.assertEqual(len(app.exception), 0)
            references = self.owner.products()[0]["references"]
            self.assertEqual(len(references), 2)
            next(button for button in app.button if button.key == f'remove_reference_{references[1]["id"]}').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(self.owner.products()[0]["references"]), 1)
            self.assertTrue(next(button for button in app.button if button.label == "Remove reference").disabled)

    def test_owner_cannot_add_a_fifth_reference_through_gallery(self):
        views = [Image.new("RGB", (50, 80), (index, 0, 0)) for index in range(3)]
        self.owner.add_references("A", views)
        app = self.app()
        self.login(app, "owner", "owner123")
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(next(button for button in app.button if button.label == "Add references").disabled)
        self.assertEqual(len(self.owner.products()[0]["references"]), 4)

    def test_owner_registers_multiple_photos_in_one_form(self):
        uploads = []
        for color in ("blue", "green"):
            buffer = io.BytesIO()
            Image.new("RGB", (50, 80), color).save(buffer, format="PNG")
            uploads.append(buffer)

        def uploader(label, **kwargs):
            if label == "Reference photographs":
                return uploads
            return None if label == "CSV with sku and quantity" else []

        with patch("streamlit.file_uploader", side_effect=uploader):
            app = self.app()
            self.login(app, "owner", "owner123")
            next(item for item in app.text_input if item.label == "SKU").input("C")
            next(item for item in app.text_input if item.label == "Product name").input("New product")
            next(button for button in app.button if button.label == "Save reference").click().run()
            self.assertEqual(len(app.exception), 0)
            product = next(item for item in self.employee.products() if item["sku"] == "C")
            self.assertEqual(len(product["references"]), 2)

    def test_changed_reference_gallery_invalidates_pending_scan(self):
        image = self.image
        upload = io.BytesIO()
        image.save(upload, format="PNG")
        detection = {"sku": "A", "name": "A", "confidence": 0.94, "box": [10, 30, 50, 90]}
        with patch("streamlit.file_uploader", return_value=upload), \
                patch("vision.load_model", return_value=object()), \
                patch("vision.detect_products", return_value=([detection], image, image, [detection])):
            app = self.app()
            self.login(app, "employee", "employee123")
            next(button for button in app.button if button.label == "Run real AI detection").click().run()
            self.assertTrue(any(button.label == "Approve and save count" for button in app.button))
            self.owner.add_references("A", [Image.new("RGB", (50, 80), "blue")])
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("catalogue or threshold changed" in item.value for item in app.info))
            self.assertFalse(any(button.label == "Approve and save count" for button in app.button))
            quantities = {row["sku"]: row["quantity"] for row in self.owner.inventory(self.location)}
            self.assertEqual(quantities, {"A": None, "B": None})

    def test_scan_progress_and_timings_are_saved_without_approving_inventory(self):
        from types import SimpleNamespace
        import streamlit as st
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        upload = io.BytesIO()
        self.image.save(upload, format="PNG")
        timings = {"references_seconds": 0.2, "photo_segmentation_seconds": 2.0,
                   "matching_seconds": 0.1, "rendering_seconds": 0.3, "total_seconds": 2.6}
        model = SimpleNamespace(last_timings=timings)
        detection = {"sku": "A", "name": "A", "confidence": 0.94, "box": [10, 30, 50, 90]}

        def infer(model, photo, products, confidence, margin, progress=None):
            self.assertTrue(callable(progress))
            progress({"stage": "references", "completed": 0, "total": 2, "sku": "A", "view": 1})
            progress({"stage": "references", "completed": 2, "total": 2})
            for stage in ("photo", "matching", "rendering"):
                progress({"stage": stage})
            return [detection], self.image, self.image, [detection]

        with patch("streamlit.file_uploader", return_value=upload), \
                patch("vision.load_model", return_value=model), \
                patch("vision.detect_products", side_effect=infer):
            app = self.app()
            self.login(app, "employee", "employee123")
            next(button for button in app.button if button.label == "Run real AI detection").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Analysis: 2.6s" in item.value for item in app.caption))
            snapshot = json.loads((self.root / "last_detection.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["timings"], timings)
            self.assertEqual(len(app.get("progress")), 0)
            self.assertEqual(self.owner.history(), [])
            self.assertTrue(all(row["quantity"] is None for row in self.owner.inventory(self.location)))

    def test_matcher_update_invalidates_pending_approval_without_changing_references(self):
        import sam_vision
        upload = io.BytesIO()
        self.image.save(upload, format="PNG")
        detection = {"sku": "A", "name": "A", "confidence": 0.94, "box": [10, 30, 50, 90]}
        with patch("streamlit.file_uploader", return_value=upload), \
                patch("vision.load_model", return_value=object()), \
                patch("vision.detect_products", return_value=([detection], self.image, self.image, [detection])):
            app = self.app()
            self.login(app, "employee", "employee123")
            next(button for button in app.button if button.label == "Run real AI detection").click().run()
            self.assertTrue(any(button.label == "Approve and save count" for button in app.button))
            snapshot = json.loads((self.root / "last_detection.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["recognition_version"], sam_vision.RECOGNITION_VERSION)
            original_references = self.owner.products()
            with patch.object(sam_vision, "RECOGNITION_VERSION", sam_vision.RECOGNITION_VERSION + "-updated"):
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any("recognition was updated" in item.value for item in app.info))
                self.assertFalse(any(button.label == "Approve and save count" for button in app.button))
            self.assertEqual(self.owner.products(), original_references)
            self.assertEqual(self.owner.history(), [])

    def test_employee_review_save_and_owner_sync_with_stubbed_inference(self):
        # Inference is stubbed only to isolate the approval and permissions workflow.
        image = self.image
        upload = io.BytesIO()
        image.save(upload, format="PNG")
        detections = [{"sku": "A", "name": "Product A", "confidence": 0.7,
                       "box": [1.0, 1.0, 30.0, 90.0], "method": "test_fixture"},
                      {"sku": None, "name": "Uncertain product", "confidence": 0.4,
                       "box": [35.0, 5.0, 60.0, 90.0], "method": "test_fixture", "status": "uncertain"}]
        from sam_vision import encode_mask
        import numpy as np
        for detection in detections:
            mask = np.zeros((image.height, image.width), dtype=bool)
            left, top, right, bottom = [int(value) for value in detection["box"]]
            mask[top:bottom, left:right] = True
            detection["mask"] = encode_mask(mask)
        with patch("streamlit.file_uploader", return_value=upload), \
                patch("vision.load_model", return_value=object()), \
                patch("vision.detect_products", return_value=(detections, image, image, detections)):
            app = self.app()
            self.login(app, "employee", "employee123")
            self.assertFalse(any(button.label == "Save reference" for button in app.button))
            self.assertEqual(len(app.tabs), 0)
            next(button for button in app.button if button.label == "Run real AI detection").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("uncertain region" in item.value for item in app.warning))
            self.assertFalse(any(button.label.startswith("Download") for button in app.button))
            app.number_input(key="approved_A").set_value(2).run()
            next(item for item in app.checkbox if item.label.startswith("I reviewed")).check().run()
            next(button for button in app.button if button.label == "Approve and save count").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("recorded successfully" in item.value for item in app.success))
            quantities = {row["sku"]: row["quantity"] for row in self.owner.inventory(self.location)}
            self.assertEqual(quantities, {"A": 2, "B": None})
            self.assertEqual(self.owner.history()[0]["predicted_quantity"], 1)
            self.assertTrue(next(button for button in app.button if button.label == "Approve and save count").disabled)


if __name__ == "__main__":
    unittest.main()
