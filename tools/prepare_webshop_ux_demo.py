"""Create a separate fictional demo DB; never change the running demo database."""
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    source = ROOT / 'data/webshop_release_demo/inventorylens.sqlite3'
    target = ROOT / 'data/webshop_ux_demo/inventorylens.sqlite3'
    if target.exists():
        raise SystemExit('UX demo already exists; preserving it.')
    if not source.exists():
        raise SystemExit('Existing release demo DB is required; no source DB was found.')
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original, sqlite3.connect(target) as copy:
        original.backup(copy)
    os.environ['WEBSHOP_DATA_DIR'] = str(target.parent)
    import db
    db.init_db()
    with db.connect() as connection:
        owner = connection.execute("SELECT business_id FROM users WHERE username='owner'").fetchone()
        if not owner:
            raise SystemExit('Source demo has no owner login.')
        connection.execute('UPDATE businesses SET name=?,is_demo=1 WHERE id=?', ('WEB&SHOP UX - FICTIONAL DEMO', owner['business_id']))
        # Clear only the new copy so workbook opening balances can be explicitly initialized.
        for table in ('scan_items','scan_sessions','inventory_counts','csv_templates','product_registration_requests'):
            connection.execute(f'DELETE FROM {table}')
    print('Created isolated UX demo. Source DB and registered reference images unchanged.')
    print(target)


if __name__ == '__main__':
    main()
