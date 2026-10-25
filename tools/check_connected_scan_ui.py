"""Real detector/matcher execution through Streamlit on a historical fixture.

Only file selection is automated. Inference and SKU models are not stubbed.
No review approval is clicked and no inventory count is posted.
"""
import io
import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'data/webshop_demo'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=FOLDER)
    parser.add_argument('--photo', type=Path, default=ROOT / 'data/segmentation_dataset/experiments/runpod_yolo11n_detection_20261017/package/final_test/images/IMG_4699.jpeg')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data_folder = args.data_dir.resolve(strict=True)
    config = json.loads((FOLDER / 'worker_connection.json').read_text())
    os.environ.update(WEBSHOP_DATA_DIR=str(data_folder), WEBSHOP_DINO_URL=config['url'], WEBSHOP_DINO_TOKEN=config['token'])
    from streamlit.testing.v1 import AppTest
    image = args.photo.resolve(strict=True)
    with sqlite3.connect(data_folder / 'inventorylens.sqlite3') as connection:
        before = connection.execute('SELECT COUNT(*) FROM scan_sessions').fetchone()[0]
        stock_before = connection.execute('SELECT * FROM inventory_counts ORDER BY id').fetchall()
    with patch('streamlit.file_uploader', return_value=io.BytesIO(image.read_bytes())):
        # Building the full shop-scoped SigLIP2 reference index on CPU can take
        # several minutes on a cold cache; allow the real, uncached path to finish.
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=360).run()
        print('STAGE app loaded', flush=True)
        app.text_input(key='login_username').input('employee')
        app.text_input(key='login_password').input('employee123')
        next(b for b in app.button if b.label == 'Sign in').click().run()
        next(r for r in app.radio if r.label == 'Workspace').set_value('Fresh-photo SKU scan').run()
        print('STAGE photo loaded', flush=True)
        next(b for b in app.button if b.label == 'Check this photo').click().run()
        print('STAGE capture check complete; requesting live detection and matching', flush=True)
        next(b for b in app.button if b.label == 'Find products in this photo').click().run(timeout=360)
        print('STAGE live detection and matching returned', flush=True)
        if app.exception or app.error:
            raise RuntimeError(str([str(e.value) for e in list(app.exception) + list(app.error)]))
        result = app.session_state['fresh_result']
        if result.get('mode') != 'LIVE' or result['detector'].get('kind') != 'fresh_tao_dino_inference':
            raise RuntimeError('Expected real live inference, not replay.')
        held = any('Confirm or exclude' in message.value for message in app.info)
        if not held:
            raise RuntimeError('Item approval boundary was not shown.')
    with sqlite3.connect(data_folder / 'inventorylens.sqlite3') as connection:
        after = connection.execute('SELECT COUNT(*) FROM scan_sessions').fetchone()[0]
        stock_after = connection.execute('SELECT * FROM inventory_counts ORDER BY id').fetchall()
    if before != after or stock_before != stock_after:
        raise RuntimeError('An unapproved test changed inventory scan history.')
    evidence = {'kind':'actual_inference_ui_smoke_on_historical_fixture',
                'detected_items':len(result['items']), 'needs_review':held,
                'scan_sessions_before':before,'scan_sessions_after':after,
                'elapsed_seconds':result['elapsed_seconds'],
                'checkpoint_sha256':result['detector']['checkpoint_sha256'],
                'detector_seconds':result['detector']['elapsed_seconds'],
                'source_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                'input_photo':str(image), 'stock_unchanged':stock_before == stock_after,
                'note':'File selection automated; real DINO and SigLIP2 ran. An existing historical photo is the default. This is not blind-test accuracy.'}
    (args.output or data_folder / 'connected_ui_smoke.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()
