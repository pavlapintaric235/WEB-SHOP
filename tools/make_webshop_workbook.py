"""Rebuild the shareable fictional catalogue workbook from its small CSV source."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_webshop_demo import write_workbook


def main():
    source = ROOT / 'docs/demo/webshop_catalog.csv'
    with source.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    target = ROOT / 'data/submission_demo_catalog.xlsx'
    target.parent.mkdir(parents=True, exist_ok=True)
    shelves = sorted({r['shelf_id'] for r in rows if r['shelf_id']})
    write_workbook(target, [('Products', [list(rows[0])] + [list(r.values()) for r in rows]),
                            ('Shelves', [['shelf_id', 'label']] + [[s, f'Fictional demo shelf {s}'] for s in shelves])])
    print(target)


if __name__ == '__main__':
    main()
