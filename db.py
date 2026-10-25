"""SQLite persistence for InventoryLens MVP."""

import csv
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.environ.get("WEBSHOP_DATA_DIR", str(Path(__file__).resolve().parent / "data"))).resolve()
DB_FILE = DATA_DIR / "inventorylens.sqlite3"
CATALOG_FILE = DATA_DIR / "catalog.json"
DEFAULT_BUSINESS_NAME = "Demo Business"
MAX_PRODUCT_REFERENCES = 4


@contextmanager
def connect():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def rows(cursor):
    return [dict(row) for row in cursor.fetchall()]


def init_db():
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS businesses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL REFERENCES businesses(id),
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('owner', 'employee')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL REFERENCES businesses(id),
                sku TEXT NOT NULL,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (business_id, sku)
            );

            CREATE TABLE IF NOT EXISTS product_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (product_id, image_path)
            );

            CREATE TABLE IF NOT EXISTS inventory_locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL REFERENCES businesses(id),
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (business_id, name)
            );

            CREATE TABLE IF NOT EXISTS inventory_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL REFERENCES businesses(id),
                location_id INTEGER NOT NULL REFERENCES inventory_locations(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                quantity INTEGER NOT NULL CHECK (quantity >= 0),
                updated_at TEXT NOT NULL,
                UNIQUE (business_id, location_id, product_id)
            );

            CREATE TABLE IF NOT EXISTS scan_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL REFERENCES businesses(id),
                user_id INTEGER NOT NULL REFERENCES users(id),
                location_id INTEGER NOT NULL REFERENCES inventory_locations(id),
                photo_hash TEXT NOT NULL,
                submission_key TEXT NOT NULL,
                scope TEXT NOT NULL CHECK (scope IN ('partial', 'complete')),
                status TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                UNIQUE (business_id, submission_key)
            );

            CREATE TABLE IF NOT EXISTS scan_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES scan_sessions(id),
                product_id INTEGER NOT NULL REFERENCES products(id),
                predicted_quantity INTEGER NOT NULL CHECK (predicted_quantity >= 0),
                approved_quantity INTEGER NOT NULL CHECK (approved_quantity >= 0)
            );

            CREATE TABLE IF NOT EXISTS csv_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                business_id INTEGER NOT NULL REFERENCES businesses(id),
                location_id INTEGER NOT NULL REFERENCES inventory_locations(id),
                filename TEXT NOT NULL,
                csv_text TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )
        db.execute(
            "INSERT OR IGNORE INTO product_references (product_id, image_path, created_at) "
            "SELECT id, image_path, created_at FROM products WHERE image_path != ''"
        )
        # Additive migrations preserve existing catalogues and count history.
        for table, column, declaration in (
            ('businesses', 'is_demo', 'INTEGER NOT NULL DEFAULT 0'),
            ('products', 'price', 'REAL'), ('products', 'unit', "TEXT DEFAULT 'pending'"),
            ('products', 'shelf_id', 'TEXT'), ('products', 'registration_status', "TEXT DEFAULT 'pending'"),
            ('products', 'brand_variant_size', 'TEXT'),
            ('products', 'opening_stock', 'INTEGER'),
            ('scan_sessions', 'operation', "TEXT NOT NULL DEFAULT 'audit'"),
            ('scan_items', 'before_quantity', 'INTEGER'), ('scan_items', 'after_quantity', 'INTEGER'),
        ):
            if column not in {r['name'] for r in db.execute(f'PRAGMA table_info({table})')}:
                db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {declaration}')
        db.executescript('''
            CREATE TABLE IF NOT EXISTS shop_shelves (
                business_id INTEGER NOT NULL REFERENCES businesses(id),
                shelf_id TEXT NOT NULL, zone TEXT, aisle TEXT, map_x INTEGER, map_y INTEGER,
                PRIMARY KEY (business_id, shelf_id));
            CREATE TABLE IF NOT EXISTS shop_layouts (
                business_id INTEGER PRIMARY KEY REFERENCES businesses(id), layout_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS product_registration_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT, business_id INTEGER NOT NULL REFERENCES businesses(id),
                user_id INTEGER NOT NULL REFERENCES users(id), description TEXT NOT NULL,
                evidence_id TEXT, created_at TEXT NOT NULL);
        ''')


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_or_create_business(name=DEFAULT_BUSINESS_NAME):
    with connect() as db:
        db.execute(
            "INSERT OR IGNORE INTO businesses (name, created_at) VALUES (?, ?)",
            (name, now()),
        )
        return dict(db.execute(
            "SELECT * FROM businesses WHERE name = ?", (name,)
        ).fetchone())


def get_business(business_id):
    with connect() as db:
        row = db.execute("SELECT * FROM businesses WHERE id = ?", (business_id,)).fetchone()
        return dict(row) if row else None


def get_business_products(business_id):
    with connect() as db:
        products = rows(
            db.execute(
                """
                SELECT *
                FROM products
                WHERE business_id = ?
                ORDER BY sku
                """,
                (business_id,),
            )
        )
        references = rows(db.execute(
            "SELECT r.id, r.product_id, r.image_path FROM product_references r "
            "JOIN products p ON p.id = r.product_id WHERE p.business_id = ? ORDER BY r.id",
            (business_id,),
        ))
        by_product = {}
        for reference in references:
            product_id = reference.pop("product_id")
            by_product.setdefault(product_id, []).append(reference)
        for product in products:
            product["references"] = by_product.get(product["id"], [])
        return products


def get_product_by_sku(business_id, sku):
    with connect() as db:
        row = db.execute(
            "SELECT * FROM products WHERE business_id = ? AND sku = ?",
            (business_id, sku),
        ).fetchone()
        return dict(row) if row else None


def add_product(business_id, sku, name, image_path, reference_paths=None):
    paths = list(reference_paths) if reference_paths is not None else [image_path]
    if not paths or len(paths) > MAX_PRODUCT_REFERENCES or paths[0] != image_path:
        raise ValueError(f"A product needs 1-{MAX_PRODUCT_REFERENCES} reference photographs.")
    with connect() as db:
        cursor = db.execute(
            """
            INSERT INTO products (business_id, sku, name, image_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (business_id, sku, name, image_path, now()),
        )
        db.executemany(
            "INSERT INTO product_references (product_id, image_path, created_at) VALUES (?, ?, ?)",
            [(cursor.lastrowid, path, now()) for path in paths],
        )


def add_product_references(business_id, sku, paths):
    paths = list(paths)
    if not paths:
        raise ValueError("Upload at least one reference photograph.")
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        product = db.execute(
            "SELECT id FROM products WHERE business_id = ? AND sku = ?", (business_id, sku)
        ).fetchone()
        if not product:
            raise ValueError("Unknown SKU.")
        count = db.execute(
            "SELECT COUNT(*) FROM product_references WHERE product_id = ?", (product["id"],)
        ).fetchone()[0]
        if count + len(paths) > MAX_PRODUCT_REFERENCES:
            raise ValueError(f"A product can have at most {MAX_PRODUCT_REFERENCES} reference photographs.")
        db.executemany(
            "INSERT INTO product_references (product_id, image_path, created_at) VALUES (?, ?, ?)",
            [(product["id"], path, now()) for path in paths],
        )
        db.execute("UPDATE products SET image_path=? WHERE id=? AND image_path=''", (paths[0], product['id']))


def remove_product_reference(business_id, sku, reference_id):
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        product = db.execute(
            "SELECT id, image_path FROM products WHERE business_id = ? AND sku = ?", (business_id, sku)
        ).fetchone()
        if not product:
            raise ValueError("Unknown SKU.")
        references = rows(db.execute(
            "SELECT id, image_path FROM product_references WHERE product_id = ? ORDER BY id",
            (product["id"],),
        ))
        selected = next((item for item in references if item["id"] == reference_id), None)
        if not selected:
            raise ValueError("Reference does not belong to this product.")
        if len(references) <= 1:
            raise ValueError("Keep at least one reference photograph per product.")
        db.execute("DELETE FROM product_references WHERE id = ?", (reference_id,))
        if selected["image_path"] == product["image_path"]:
            replacement = next(item for item in references if item["id"] != reference_id)
            db.execute("UPDATE products SET image_path = ? WHERE id = ?",
                       (replacement["image_path"], product["id"]))


def update_product(business_id, sku, name):
    with connect() as db:
        db.execute(
            "UPDATE products SET name = ? WHERE business_id = ? AND sku = ?",
            (name, business_id, sku),
        )


def get_locations(business_id):
    with connect() as db:
        return rows(
            db.execute(
                """
                SELECT *
                FROM inventory_locations
                WHERE business_id = ?
                ORDER BY name
                """,
                (business_id,),
            )
        )


def add_location(business_id, name):
    with connect() as db:
        db.execute(
            """
            INSERT OR IGNORE INTO inventory_locations (business_id, name, created_at)
            VALUES (?, ?, ?)
            """,
            (business_id, name, now()),
        )


def default_location(business_id):
    add_location(business_id, "Main Shelf")
    return next(location for location in get_locations(business_id)
                if location["name"] == "Main Shelf")


def latest_csv_template(business_id, location_id):
    with connect() as db:
        row = db.execute(
            """
            SELECT *
            FROM csv_templates
            WHERE business_id = ? AND location_id = ?
            ORDER BY uploaded_at DESC, id DESC
            LIMIT 1
            """,
            (business_id, location_id),
        ).fetchone()
        return dict(row) if row else None


def validate_csv_text(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")), strict=True)
    fields = reader.fieldnames or []
    if not {"sku", "quantity"}.issubset(fields):
        raise ValueError("CSV must contain 'sku' and 'quantity' columns.")
    if len(fields) != len(set(fields)) or any(not field for field in fields):
        raise ValueError("CSV column names must be unique and nonempty.")
    seen = set()
    parsed = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Each CSV row must have the same number of columns as the header.")
        sku = row["sku"].strip().upper()
        if not sku:
            raise ValueError("CSV rows need a nonempty SKU.")
        if sku in seen:
            raise ValueError(f"Duplicate SKU in CSV: {sku}")
        seen.add(sku)
        value = row["quantity"].strip()
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"Quantity for SKU {sku} must be a nonnegative integer.")
        validate_quantity(int(value))
        parsed.append(row)
    return fields, parsed


def validate_quantity(value):
    if type(value) is not int or not 0 <= value <= 2_147_483_647:
        raise ValueError("Quantities must be nonnegative whole numbers up to 2147483647.")
    return value


def require_location(db, business_id, location_id):
    if not db.execute(
        "SELECT id FROM inventory_locations WHERE business_id = ? AND id = ?",
        (business_id, location_id),
    ).fetchone():
        raise PermissionError("Inventory location is not part of your business.")


def import_csv_quantities(business_id, location_id, csv_text, filename="inventory.csv"):
    _, parsed = validate_csv_text(csv_text)
    products = {product["sku"]: product for product in get_business_products(business_id)}
    imported = 0
    with connect() as db:
        require_location(db, business_id, location_id)
        db.execute(
            "INSERT INTO csv_templates (business_id, location_id, filename, csv_text, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (business_id, location_id, filename, csv_text, now()),
        )
        for row in parsed:
            sku = row["sku"].strip().upper()
            if sku in products:
                _set_inventory_count(db, business_id, location_id,
                                     products[sku]["id"], int(row["quantity"]))
                imported += 1
    return imported


def set_inventory_count(business_id, location_id, product_id, quantity):
    validate_quantity(quantity)
    with connect() as db:
        require_location(db, business_id, location_id)
        if not db.execute("SELECT id FROM products WHERE id = ? AND business_id = ?",
                          (product_id, business_id)).fetchone():
            raise PermissionError("Product is not part of your business.")
        _set_inventory_count(db, business_id, location_id, product_id, quantity)


def _set_inventory_count(db, business_id, location_id, product_id, quantity):
    db.execute(
        """
            INSERT INTO inventory_counts (
                business_id, location_id, product_id, quantity, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (business_id, location_id, product_id)
            DO UPDATE SET quantity = excluded.quantity, updated_at = excluded.updated_at
        """,
        (business_id, location_id, product_id, quantity, now()),
    )


def get_inventory(business_id, location_id=None):
    params = [business_id]
    location_filter = ""
    if location_id is not None:
        location_filter = "AND l.id = ?"
        params.append(location_id)
    with connect() as db:
        return rows(
            db.execute(
                f"""
                SELECT
                    l.name AS location,
                    p.sku,
                    p.name,
                    c.quantity,
                    c.updated_at
                FROM inventory_locations l
                JOIN products p ON p.business_id = l.business_id
                LEFT JOIN inventory_counts c
                    ON c.location_id = l.id
                    AND c.product_id = p.id
                    AND c.business_id = l.business_id
                WHERE l.business_id = ?
                {location_filter}
                ORDER BY l.name, p.sku
                """,
                params,
            )
        )


def create_scan_session(
    business_id, user_id, location_id, photo_hash, approved_counts,
    predicted_counts=None, notes="", submission_key=None, scope="partial", operation="audit"
):
    predicted_counts = predicted_counts or {}
    products = {product["sku"]: product for product in get_business_products(business_id)}
    missing = [sku for sku in approved_counts if sku not in products]
    if missing:
        raise ValueError(f"Unknown SKU(s): {', '.join(missing)}")
    if not approved_counts:
        raise ValueError("Select at least one SKU to approve.")
    if scope not in {"partial", "complete"}:
        raise ValueError("Invalid count scope.")
    if operation not in {'audit', 'add', 'subtract'}:
        raise ValueError('Invalid inventory operation.')
    if scope == "complete" and set(approved_counts) != set(products):
        raise ValueError("A complete location count must include every registered SKU.")
    if not submission_key or not photo_hash:
        raise ValueError("A scan session and photograph are required.")
    for sku, value in approved_counts.items():
        validate_quantity(value)
        validate_quantity(predicted_counts.get(sku, 0))
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        require_location(db, business_id, location_id)
        if not db.execute("SELECT id FROM users WHERE id = ? AND business_id = ?",
                          (user_id, business_id)).fetchone():
            raise PermissionError("User is not part of your business.")
        previous = db.execute(
            "SELECT * FROM scan_sessions WHERE business_id = ? AND submission_key = ?",
            (business_id, submission_key),
        ).fetchone()
        if previous:
            old_items = rows(db.execute(
                "SELECT p.sku, i.approved_quantity FROM scan_items i "
                "JOIN products p ON p.id = i.product_id WHERE i.session_id = ?",
                (previous["id"],),
            ))
            if (previous["user_id"] == user_id and previous["location_id"] == location_id
                    and previous["photo_hash"] == photo_hash and previous["scope"] == scope
                    and previous['operation'] == operation
                    and {item["sku"]: item["approved_quantity"] for item in old_items}
                    == approved_counts):
                return previous["id"]
            raise ValueError("This scan was already approved with different quantities or scope.")
        cursor = db.execute(
                """
                INSERT INTO scan_sessions (
                    business_id, user_id, location_id, photo_hash, submission_key,
                    scope, status, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'approved', ?, ?)
                """,
                (business_id, user_id, location_id, photo_hash, submission_key, scope, notes, now()),
            )
        session_id = cursor.lastrowid
        db.execute('UPDATE scan_sessions SET operation=? WHERE id=?', (operation, session_id))
        for sku, approved_quantity in approved_counts.items():
            product = products[sku]
            prior = db.execute('SELECT quantity FROM inventory_counts WHERE business_id=? AND location_id=? AND product_id=?',
                               (business_id, location_id, product['id'])).fetchone()
            before = prior['quantity'] if prior else 0
            after = (approved_quantity if operation == 'audit' else
                     before + approved_quantity if operation == 'add' else before - approved_quantity)
            if after < 0:
                raise ValueError(f'Removal exceeds stock for {sku}; no changes saved.')
            predicted_quantity = int(predicted_counts.get(sku, 0))
            db.execute(
                """
                INSERT INTO scan_items (
                    session_id, product_id, predicted_quantity, approved_quantity
                )
                VALUES (?, ?, ?, ?)
                """,
                (session_id, product["id"], predicted_quantity, approved_quantity),
            )
            db.execute('UPDATE scan_items SET before_quantity=?, after_quantity=? WHERE session_id=? AND product_id=?',
                       (before, after, session_id, product['id']))
            _set_inventory_count(db, business_id, location_id, product["id"], after)
        return session_id


def get_scan_history(business_id, limit=50):
    with connect() as db:
        return rows(
            db.execute(
                """
                SELECT
                    s.id,
                    s.created_at,
                    s.scope,
                    s.operation,
                    s.photo_hash,
                    s.submission_key,
                    s.notes,
                    u.username,
                    l.name AS location,
                    p.sku,
                    p.name AS product,
                    i.predicted_quantity,
                    i.approved_quantity
                    ,i.before_quantity, i.after_quantity
                FROM scan_sessions s
                JOIN users u ON u.id = s.user_id
                JOIN inventory_locations l ON l.id = s.location_id
                JOIN scan_items i ON i.session_id = s.id
                JOIN products p ON p.id = i.product_id
                WHERE s.business_id = ?
                ORDER BY s.created_at DESC, s.id DESC
                LIMIT ?
                """,
                (business_id, limit),
            )
        )


def export_inventory_csv(business_id, location_id):
    template = latest_csv_template(business_id, location_id)
    inventory = {
        row["sku"]: row for row in get_inventory(business_id, location_id)
    }
    if template:
        fields, output_rows = validate_csv_text(template["csv_text"])
        seen = set()
        for row in output_rows:
            sku = row.get("sku", "").strip().upper()
            if sku in inventory:
                if inventory[sku]["quantity"] is not None:
                    row["quantity"] = str(inventory[sku]["quantity"])
                seen.add(sku)
        for sku, item in inventory.items():
            if sku in seen:
                continue
            row = {field: "" for field in fields}
            row["sku"] = sku
            row["quantity"] = "" if item["quantity"] is None else str(item["quantity"])
            if "name" in fields:
                row["name"] = item["name"]
            output_rows.append(row)
    else:
        fields = ["sku", "name", "quantity"]
        output_rows = [
            {
                "sku": item["sku"],
                "name": item["name"],
                "quantity": "" if item["quantity"] is None else str(item["quantity"]),
            }
            for item in inventory.values()
        ]

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output_rows)
    return buffer.getvalue().encode("utf-8-sig")


def migrate_catalog_json(business_id):
    if not CATALOG_FILE.exists():
        return
    migration = f"json_catalogue_{business_id}"
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT name FROM migrations WHERE name = ?", (migration,)).fetchone():
            return
        items = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
        for item in items:
            image_path = Path(item["image_path"])
            if not image_path.is_absolute():
                image_path = Path(__file__).resolve().parent / image_path
            if not image_path.is_file():
                raise ValueError(f"Missing reference for {item['sku']}: {image_path}")
            db.execute(
                "INSERT OR IGNORE INTO products (business_id, sku, name, image_path, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (business_id, item["sku"], item["name"], str(image_path), now()),
            )
        db.execute(
            "INSERT OR IGNORE INTO product_references (product_id, image_path, created_at) "
            "SELECT id, image_path, created_at FROM products WHERE business_id = ?", (business_id,)
        )
        db.execute("INSERT INTO migrations VALUES (?, ?)", (migration, now()))
