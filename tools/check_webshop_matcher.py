"""Real local SigLIP2 reference self-check; explicitly not an accuracy benchmark."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import db
from live_scan import ShopMatcher, SIGLIP_REVISION


def main():
    folder = ROOT / 'data/webshop_demo'
    db.DATA_DIR, db.DB_FILE = folder, folder / 'inventorylens.sqlite3'
    products = db.get_business_products(1)
    started = time.perf_counter()
    matcher = ShopMatcher(products)
    results = []
    for sku in [p['sku'] for p in products]:
        reference = next(row for row in matcher.records if row['sku'] == sku)
        crop = reference['crop']
        ranked = matcher.rank_boxes(crop, [{'box': [0, 0, crop.width, crop.height], 'confidence': 1.}])[0]
        results.append({'query_sku': sku, 'suggested_sku': ranked['sku'], 'top3': ranked['ranked_scores'][:3]})
    result = {'kind': 'reference_gallery_self_check_not_test_accuracy', 'siglip_revision': SIGLIP_REVISION,
              'reference_count': len(matcher.records), 'products': len(products),
              'seconds': round(time.perf_counter() - started, 2), 'results': results,
              'note': 'Queries are gallery references. This verifies local execution and catalog mapping, not unseen-photo recognition.'}
    output = folder / 'matcher_readiness.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
