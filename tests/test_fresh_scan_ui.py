"""Exercise new scan states with explicitly simulated worker and SKU results."""
import io
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from tests import test_workflow as workflow_helpers


class FreshScanUITests(unittest.TestCase):
    setUp = workflow_helpers.WorkflowTests.setUp

    def open_scan(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        app.text_input(key='login_username').input('employee')
        app.text_input(key='login_password').input('employee123')
        next(b for b in app.button if b.label == 'Sign in').click().run()
        next(r for r in app.radio if r.label == 'Workspace').set_value('Fresh-photo SKU scan').run()
        return app

    def photo(self):
        buffer = io.BytesIO()
        Image.fromarray(np.random.default_rng(10).integers(40, 210, (800, 800, 3), dtype=np.uint8)).save(buffer, format='PNG')
        return buffer

    def test_capture_works_with_no_worker_and_no_inventory_writes(self):
        with patch('streamlit.file_uploader', return_value=self.photo()), patch.dict(os.environ,
                {'WEBSHOP_DINO_URL': '', 'WEBSHOP_DINO_TOKEN': ''}):
            app = self.open_scan()
            self.assertEqual(len(app.exception), 0)
            next(b for b in app.button if b.label == 'Check this photo').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(next(b for b in app.button if b.label == 'Find products in this photo').disabled)
            self.assertTrue(any('not connected' in w.value for w in app.warning))
            self.assertEqual(self.owner.history(), [])

    def test_fresh_predictions_are_held_for_item_confirmation(self):
        detector = {'detections': [{'box': [1, 2, 300, 400], 'confidence': .9}]}
        item = {'sku': 'A', 'box': [1, 2, 300, 400],
                'ranked_scores': [{'sku': 'A', 'similarity': .8}]}
        matcher = Mock()
        matcher.rank_boxes.return_value = [item]
        with patch('streamlit.file_uploader', return_value=self.photo()), patch.dict(os.environ,
                {'WEBSHOP_DINO_URL': 'http://127.0.0.1:8765', 'WEBSHOP_DINO_TOKEN': 'simulated-only'}), \
                patch('scanner_ui.request_dino', return_value=detector) as worker, \
                patch('scanner_ui.shop_matcher', return_value=matcher):
            app = self.open_scan()
            next(b for b in app.button if b.label == 'Check this photo').click().run()
            next(b for b in app.button if b.label == 'Find products in this photo').click().run()
            self.assertEqual(len(app.exception), 0)
            worker.assert_called_once()
            self.assertTrue(any('Confirm or exclude' in message.value for message in app.info))
            self.assertFalse(any(b.label == 'Approve shelf count' for b in app.button))
            self.assertEqual(self.owner.history(), [])


if __name__ == '__main__':
    unittest.main()
