"""Real local SigLIP2 + archived DINO UI smoke. No stock approval or AWS calls."""
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FOLDER = ROOT / 'data/webshop_release_demo'


def main():
    os.environ['WEBSHOP_DATA_DIR'] = str(FOLDER)
    os.environ['WEBSHOP_DINO_URL'] = ''
    os.environ['WEBSHOP_DINO_TOKEN'] = ''
    from streamlit.testing.v1 import AppTest
    with sqlite3.connect(FOLDER / 'inventorylens.sqlite3') as connection:
        before = connection.execute('SELECT COUNT(*) FROM scan_sessions').fetchone()[0]
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=360).run()
    app.text_input(key='login_username').input('employee')
    app.text_input(key='login_password').input('employee123')
    next(b for b in app.button if b.label == 'Sign in').click().run()
    next(r for r in app.radio if r.label == 'Workspace').set_value('Fresh-photo SKU scan').run()
    app.radio(key='fresh_source').set_value('Archived photo · REPLAY MODE').run()
    next(b for b in app.button if b.label == 'Check this photo').click().run()
    next(b for b in app.button if b.label == 'Find products in this photo').click().run(timeout=360)
    if app.exception or app.error:
        raise RuntimeError(str([e.value for e in list(app.exception) + list(app.error)]))
    result = app.session_state['fresh_result']
    assert result['mode'] == 'REPLAY'
    assert result['detector']['kind'] == 'archived_detection_replay'
    assert any('Confirm or exclude' in i.value for i in app.info)
    with sqlite3.connect(FOLDER / 'inventorylens.sqlite3') as connection:
        after = connection.execute('SELECT COUNT(*) FROM scan_sessions').fetchone()[0]
    assert before == after
    evidence = dict(kind='REPLAY_DINO_REAL_CURRENT_SHOP_SIGLIP2_UI_SMOKE',
                    proposals=len(result['items']), elapsed_seconds=result['elapsed_seconds'],
                    checkpoint_sha256=result['detector']['checkpoint_sha256'],
                    source_sha256=result['source_sha256'], catalog_sha256=result['catalog_sha256'],
                    unknown=sum(not i['sku'] for i in result['items']),
                    scan_sessions_before=before, scan_sessions_after=after,
                    note='Historical photo and archived DINO boxes; actual current-shop SigLIP2 inference. Not fresh detection or accuracy.')
    (FOLDER / 'release_replay_smoke.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
