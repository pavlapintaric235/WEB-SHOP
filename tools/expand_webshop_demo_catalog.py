"""Add the rest of the existing registered SKU catalog to the isolated demo.

This is additive: existing demo records, quantities, prices, locations, and
reference images are left untouched. New products receive no invented shelf,
opening stock, or price values.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "webshop_demo"
SOURCE_DB = ROOT / "data" / "inventorylens.sqlite3"

os.environ["WEBSHOP_DATA_DIR"] = str(DEMO)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import db
from prepare_webshop_demo import write_workbook


def main() -> None:
    if not (DEMO / "inventorylens.sqlite3").is_file():
        raise FileNotFoundError(f"Isolated demo database not found: {DEMO}")
    if db.DB_FILE.resolve() != (DEMO / "inventorylens.sqlite3").resolve():
        raise RuntimeError("Refusing to modify anything except the isolated demo database.")

    db.DB_FILE = SOURCE_DB
    source = {item["sku"]: item for item in db.get_business_products(1)}
    db.DB_FILE = DEMO / "inventorylens.sqlite3"

    with db.connect() as connection:
        businesses = db.rows(connection.execute(
            "SELECT id, name FROM businesses WHERE name LIKE ?", ("Web&Shop Corner Market%",)
        ))
    if len(businesses) != 1:
        raise RuntimeError(f"Expected one isolated Corner Market demo, found {len(businesses)}.")
    business_id = businesses[0]["id"]
    fixture_path = DEMO / "demo_shop.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not fixture.get("demo_data") or fixture.get("shop") != "Web&Shop Corner Market":
        raise RuntimeError("Demo metadata does not identify the expected isolated shop.")

    existing = {item["sku"] for item in db.get_business_products(business_id)}
    additions = [source[sku] for sku in sorted(set(source) - existing)]
    for product in additions:
        references = product.get("references", [])
        if not references or len(references) > db.MAX_PRODUCT_REFERENCES:
            raise ValueError(f"{product['sku']} does not have a valid registered reference set.")
        for reference in references:
            if not Path(reference["image_path"]).is_file():
                raise FileNotFoundError(reference["image_path"])

    for product in additions:
        sku = product["sku"]
        folder = DEMO / "references" / sku
        if folder.exists():
            raise FileExistsError(f"Refusing to overwrite existing demo reference folder: {folder}")
        folder.mkdir(parents=True)
        copied = []
        for number, reference in enumerate(product["references"]):
            original = Path(reference["image_path"])
            destination = folder / f"{number}{original.suffix}"
            shutil.copy2(original, destination)
            copied.append(str(destination))
        db.add_product(business_id, sku, product["name"], copied[0], copied)
        fixture["products"].append({
            "sku": sku,
            "name": product["name"],
            "gtin": "",
            "unit": "one retail package",
            "price": "",
            "shelf_id": "",
            "initial_quantity": "",
            "identity_status": "existing registered catalog identity; owner to set price, shelf, and opening count",
        })

    all_skus = {item["sku"] for item in fixture["products"]}
    if all_skus != set(source):
        raise RuntimeError(f"Demo/source SKU mismatch: demo={len(all_skus)} source={len(source)}")
    fixture["note"] = (
        "All products and reference photos are copied from the existing registered catalog. "
        "Existing demo prices, shelf assignments, and opening quantities are illustrative; "
        "new products have blank owner-managed values. Verify product variants and package sizes."
    )
    fixture_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False), encoding="utf-8")

    with (DEMO / "demo_products.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fixture["products"][0]))
        writer.writeheader()
        writer.writerows(fixture["products"])
    workbook_sheets = [
        ("Products", [list(fixture["products"][0])] + [list(item.values()) for item in fixture["products"]]),
        ("Shelves", [list(fixture["shelves"][0])] + [list(item.values()) for item in fixture["shelves"]]),
    ]
    write_workbook(DEMO / "demo_catalog.xlsx", workbook_sheets)
    print(json.dumps({
        "added_products": [item["sku"] for item in additions],
        "registered_products": len(all_skus),
        "registered_reference_photos": sum(len(item.get("references", [])) for item in db.get_business_products(business_id)),
        "demo_catalog": str(fixture_path),
        "workbook": str(DEMO / "demo_catalog.xlsx"),
    }, indent=2))


if __name__ == "__main__":
    main()
