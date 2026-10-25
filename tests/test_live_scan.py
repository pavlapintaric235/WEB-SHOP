"""Scan boundaries: orientation, stale/mismatched inference, review and worker output."""
import hashlib
import io
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from live_scan import (COORDINATES, normalize_photo, validate_worker_result,
                       review_counts, ShopMatcher, request_dino)
from tools.tao_dino_worker import parse_labels


class FreshScanTests(unittest.TestCase):
    def setUp(self):
        self.data = b'upright image bytes'
        self.result = {'kind': 'fresh_tao_dino_inference', 'image_sha256': hashlib.sha256(self.data).hexdigest(),
                       'coordinate_space': COORDINATES, 'image_size': [100, 200],
                       'checkpoint_sha256': 'a' * 64,
                       'detections': [{'box': [1, 2, 30, 100], 'confidence': .9}]}

    def test_exif_is_applied_once_without_changing_source(self):
        original = Image.new('RGB', (60, 30), 'red')
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        original.save(stream, format='JPEG', exif=exif)
        source = stream.getvalue()
        image, upright = normalize_photo(source)
        self.assertEqual(image.size, (30, 60))
        twice, _ = normalize_photo(upright)
        self.assertEqual(twice.size, image.size)
        self.assertEqual(stream.getvalue(), source)
        self.assertEqual(np.asarray(image).tolist(), np.asarray(twice).tolist())

    def test_response_must_match_pixels_dimensions_model_and_fresh_kind(self):
        with patch.dict(os.environ, {'WEBSHOP_DINO_SHA256': 'a' * 64}):
            self.assertEqual(len(validate_worker_result(self.result, self.data, (100, 200))['detections']), 1)
            for key, bad in [('image_sha256', 'b' * 64), ('image_size', [200, 100]),
                             ('checkpoint_sha256', 'c' * 64), ('kind', 'archived_replay')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_worker_result({**self.result, key: bad}, self.data, (100, 200))

    def test_nonfinite_or_out_of_bounds_worker_values_are_rejected(self):
        with patch.dict(os.environ, {'WEBSHOP_DINO_SHA256': 'a' * 64}):
            for row in [{'box': [1, 2, 101, 100], 'confidence': .8},
                        {'box': [1, 2, 30, 100], 'confidence': math.nan}]:
                with self.subTest(row=row), self.assertRaises(ValueError):
                    validate_worker_result({**self.result, 'detections': [row]}, self.data, (100, 200))

    def test_no_config_means_no_replay_and_no_request(self):
        with patch.dict(os.environ, {'WEBSHOP_DINO_URL': '', 'WEBSHOP_DINO_TOKEN': ''}), patch('live_scan.urlopen') as http:
            with self.assertRaisesRegex(RuntimeError, 'not connected'):
                request_dino(self.data, (100, 200))
            http.assert_not_called()

    def test_remote_plaintext_worker_url_is_rejected(self):
        with patch.dict(os.environ, {'WEBSHOP_DINO_URL': 'http://example.com', 'WEBSHOP_DINO_TOKEN': 'secret'}):
            with self.assertRaisesRegex(ValueError, 'HTTPS'):
                request_dino(self.data, (100, 200))

    def row(self, **changes):
        return {'x1': 0, 'y1': 0, 'x2': 30, 'y2': 50, 'sku': 'A', 'quantity': 1,
                'confirmed': True, 'exclude': False, **changes}

    def test_review_requires_confirmation_and_registered_shop_sku(self):
        for changes in [{'confirmed': False}, {'confirmed': math.nan}, {'sku': 'OTHER-SHOP'},
                        {'quantity': 1.5}, {'quantity': math.nan}, {'x2': 101}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                review_counts([self.row(**changes)], {'A'}, (100, 200))

    def test_merged_box_can_be_excluded_and_replaced_with_two_items(self):
        rows = [self.row(confirmed=False, exclude=True), self.row(), self.row(x1=40, x2=70)]
        self.assertEqual(review_counts(rows, {'A'}, (100, 200)), {'A': 2})

    def test_repeated_sku_rows_collapse_to_one_sku_with_summed_units(self):
        rows = [self.row(quantity=3), self.row(x1=40, x2=70, quantity=2),
                self.row(sku='B', x1=75, x2=99, quantity=1)]
        counts = review_counts(rows, {'A', 'B'}, (100, 200))
        self.assertEqual(counts, {'A': 5, 'B': 1})
        self.assertEqual(len(counts), 2)

    def test_matcher_only_ranks_registered_shop_references(self):
        class Encoder:
            def encode(self, images, batch_size=1):
                return np.asarray([[1., 0.] for _ in images])
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'ref.png'
            Image.new('RGB', (30, 40), 'red').save(reference)
            matcher = ShopMatcher([{'sku': 'SHOP-A', 'name': 'Registered A', 'image_path': str(reference)}], encoder=Encoder())
            result = matcher.rank_boxes(Image.new('RGB', (100, 200)), [{'box': [0, 0, 30, 50], 'confidence': .8}])
            self.assertEqual(result[0]['sku'], 'SHOP-A')
            self.assertEqual([r['sku'] for r in result[0]['ranked_scores']], ['SHOP-A'])

    def test_kitti_parser_handles_blank_negative_and_clips_roundoff(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / 'query.txt'
            labels.write_text('')
            self.assertEqual(parse_labels(labels, (100, 200)), [])
            labels.write_text('product 0 0 0 -0.01 2 100.01 150 0 0 0 0 0 0 0 .9\n')
            self.assertEqual(parse_labels(labels, (100, 200))[0]['box'], [0, 2, 100, 150])
            labels.write_text('product 0 0 0 0 2 nan 150 0 0 0 0 0 0 0 .9\n')
            with self.assertRaises(ValueError):
                parse_labels(labels, (100, 200))


if __name__ == '__main__':
    unittest.main()
