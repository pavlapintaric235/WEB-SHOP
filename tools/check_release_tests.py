"""Run only application release regressions, preserving an explicit scenario matrix."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    folder = ROOT / 'data/webshop_release_checks'
    folder.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestLoader().loadTestsFromNames([
        'tests.test_live_scan', 'tests.test_workflow', 'tests.test_fresh_scan_ui',
        'tests.test_webshop_delivery', 'tests.test_practical_ux'])
    with (folder / 'tests.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = dict(tests_run=result.testsRun, failures=[str(t) for t, _ in result.failures],
                  errors=[str(t) for t, _ in result.errors], skipped=[str(t) for t, _ in result.skipped],
                  passed=result.wasSuccessful(),
                  note='Isolated SQLite fixtures. UI inference/HTTP failure tests use explicit mocks. Real replay is a separate command.')
    (folder / 'tests.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
