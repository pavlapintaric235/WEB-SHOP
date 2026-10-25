"""Authorized operations shared by the owner dashboard and scanner."""

import hashlib
import re
import sqlite3
from pathlib import Path
from uuid import uuid4

import auth
import db
from images import product_references
from PIL import Image, ImageOps


class InventoryService:
    """Resolve business and role from the server's authenticated user ID."""

    def __init__(self, user_id):
        self.user_id = user_id

    def user(self, owner=False):
        return auth.require_user(self.user_id, owner=owner)

    def products(self):
        return db.get_business_products(self.user()["business_id"])

    def locations(self):
        return db.get_locations(self.user()["business_id"])

    def _location(self, user, location_id):
        with db.connect() as connection:
            db.require_location(connection, user["business_id"], location_id)

    def register_product(self, sku, name, image):
        user = self.user(owner=True)
        sku, name = sku.strip().upper(), name.strip()
        if not re.fullmatch(r"[A-Z0-9_-]{1,40}", sku):
            raise ValueError("SKU must be 1-40 letters, numbers, '-' or '_'.")
        if not name:
            raise ValueError("Product name is required.")
        if db.get_product_by_sku(user["business_id"], sku):
            raise ValueError("That SKU is already registered.")
        images = self._reference_images(image)
        paths = self._save_references(user["business_id"], sku, images)
        try:
            db.add_product(user["business_id"], sku, name, paths[0], paths)
        except sqlite3.IntegrityError as error:
            self._discard_new_references(paths)
            raise ValueError("That SKU is already registered.") from error
        except Exception:
            self._discard_new_references(paths)
            raise

    @staticmethod
    def _reference_images(images, existing_paths=()):
        images = list(images) if isinstance(images, (list, tuple)) else [images]
        if not images or len(images) + len(existing_paths) > db.MAX_PRODUCT_REFERENCES:
            raise ValueError(f"Upload 1-{db.MAX_PRODUCT_REFERENCES} total reference photographs per product.")
        seen = set()

        def fingerprint(image):
            return hashlib.sha256(str(image.size).encode("ascii") + image.tobytes()).digest()

        for path in existing_paths:
            with Image.open(path) as image:
                seen.add(fingerprint(ImageOps.exif_transpose(image).convert("RGB")))
        normalized = []
        for image in images:
            if not isinstance(image, Image.Image):
                raise ValueError("Upload a readable reference photograph.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            digest = fingerprint(image)
            if digest in seen:
                raise ValueError("Duplicate reference photograph. Upload a different view.")
            seen.add(digest)
            normalized.append(image)
        return normalized

    @staticmethod
    def _discard_new_references(paths):
        for path in paths:
            Path(path).unlink(missing_ok=True)

    def _save_references(self, business_id, sku, images):
        folder = db.DATA_DIR / "references" / str(business_id)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        try:
            for image in images:
                path = folder / f"{sku}-{uuid4().hex}.png"
                paths.append(str(path))
                image.save(path)
        except Exception:
            self._discard_new_references(paths)
            raise
        return paths

    def add_references(self, sku, images):
        user = self.user(owner=True)
        sku = sku.strip().upper()
        product = next((item for item in self.products() if item["sku"] == sku), None)
        if not product:
            raise ValueError("Unknown SKU.")
        existing = [item["image_path"] for item in product_references(product)]
        normalized = self._reference_images(images, existing)
        paths = self._save_references(user["business_id"], sku, normalized)
        try:
            db.add_product_references(user["business_id"], sku, paths)
        except Exception:
            self._discard_new_references(paths)
            raise
        return len(paths)

    def remove_reference(self, sku, reference_id):
        user = self.user(owner=True)
        db.remove_product_reference(user["business_id"], sku.strip().upper(), reference_id)

    def update_product(self, sku, name):
        user = self.user(owner=True)
        if not name.strip():
            raise ValueError("Product name is required.")
        if not db.get_product_by_sku(user["business_id"], sku):
            raise ValueError("Unknown SKU.")
        db.update_product(user["business_id"], sku, name.strip())

    def add_location(self, name):
        user = self.user(owner=True)
        if not name.strip():
            raise ValueError("Location name is required.")
        db.add_location(user["business_id"], name.strip())

    def add_employee(self, username, password):
        user = self.user(owner=True)
        try:
            auth.create_user(user["business_id"], username, password, "employee")
        except sqlite3.IntegrityError as error:
            raise ValueError("Email or username is already taken.") from error

    def employees(self):
        user = self.user(owner=True)
        with db.connect() as connection:
            return db.rows(connection.execute(
                "SELECT username, role, created_at FROM users WHERE business_id = ? "
                "ORDER BY username", (user["business_id"],),
            ))

    def inventory(self, location_id):
        user = self.user(owner=True)
        self._location(user, location_id)
        return db.get_inventory(user["business_id"], location_id)

    def import_csv(self, location_id, filename, csv_text, confirmed=False):
        user = self.user(owner=True)
        self._location(user, location_id)
        if not confirmed:
            raise ValueError("Confirm replacing quantities at the selected location first.")
        return db.import_csv_quantities(user["business_id"], location_id, csv_text, filename)

    def export_csv(self, location_id):
        user = self.user(owner=True)
        self._location(user, location_id)
        return db.export_inventory_csv(user["business_id"], location_id)

    def history(self):
        return db.get_scan_history(self.user(owner=True)["business_id"])

    def approve(self, location_id, photo_hash, submission_key, approved_counts,
                predicted_counts=None, scope="partial", notes="", operation="audit"):
        user = self.user()
        self._location(user, location_id)
        return db.create_scan_session(
            user["business_id"], user["id"], location_id, photo_hash,
            approved_counts, predicted_counts, notes, submission_key, scope, operation,
        )

    def import_catalog(self, records, confirmed=False, opening=False, shelves=(), fictional=False):
        user = self.user(owner=True)
        from shop_catalog import import_catalog
        return import_catalog(user['business_id'], records, confirmed, opening, shelves, fictional)

    def preview_action(self, location_id, counts, operation):
        user = self.user()
        self._location(user, location_id)
        from scan_review import stock_preview
        return stock_preview(db.get_inventory(user['business_id'], location_id), counts, operation)

    def preview_catalog_action(self, counts, operation):
        from catalog_stock import preview
        return preview(self, counts, operation)

    def approve_catalog_action(self, photo_hash, submission_key, counts, predicted=None, notes='', operation='subtract', expected_preview=None):
        from catalog_stock import approve
        return approve(self,photo_hash,submission_key,counts,predicted,notes,operation,expected_preview)

    def request_registration(self, description, evidence_id):
        user = self.user()
        if not description.strip() or len(description) > 2000:
            raise ValueError('Describe the unknown product (at most 2000 characters).')
        with db.connect() as connection:
            connection.execute('INSERT INTO product_registration_requests (business_id,user_id,description,evidence_id,created_at) VALUES (?,?,?,?,?)',
                               (user['business_id'], user['id'], description.strip(), evidence_id, db.now()))

    def registration_requests(self):
        user = self.user(owner=True)
        with db.connect() as connection:
            return db.rows(connection.execute('SELECT * FROM product_registration_requests WHERE business_id=? ORDER BY id DESC', (user['business_id'],)))
