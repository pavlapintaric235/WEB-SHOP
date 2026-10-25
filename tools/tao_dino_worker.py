"""Authenticated loopback HTTP worker for the existing TAO 5.5 DINO runtime.

Run on the GPU machine and reach it through SSH forwarding. No model download,
installation, provisioning or saved-prediction replay is performed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import json
import os
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, ImageOps
import yaml

COORDINATES = 'EXIF-corrected upright image pixels; xyxy'


def parse_labels(path, size):
    import math
    if not path.is_file():
        raise RuntimeError('TAO did not emit an item label file.')
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 16 or fields[0] != 'product':
            raise ValueError('Unexpected TAO KITTI label format.')
        coords = [float(v) for v in fields[4:8]]
        confidence = float(fields[-1])
        if not all(math.isfinite(v) for v in coords + [confidence]) or not 0 <= confidence <= 1:
            raise ValueError('Invalid TAO detection values.')
        # Numerical output can slightly exceed the image boundary.
        box = [max(0., min(float(size[i % 2]), value)) for i, value in enumerate(coords)]
        if box[0] < box[2] and box[1] < box[3]:
            rows.append({'box': box, 'confidence': confidence})
    return rows


def infer(payload, args, checkpoint_sha256):
    data = base64.b64decode(payload['image_b64'], validate=True)
    if not data or len(data) > 20 * 1024 * 1024:
        raise ValueError('Image exceeds the worker limit.')
    digest = hashlib.sha256(data).hexdigest()
    if digest != payload['image_sha256']:
        raise ValueError('Image hash mismatch.')
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > 30_000_000:
            raise ValueError('Image exceeds 30 megapixels.')
        image = ImageOps.exif_transpose(source).convert('RGB')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='webshop-dino-') as directory:
        root = Path(directory)
        images = root / 'images'
        images.mkdir()
        image.save(images / 'query.png')
        classmap = root / 'classmap.txt'
        classmap.write_text('product\n')
        config = yaml.safe_load(args.spec.read_text())
        config['results_dir'] = str(root / 'results')
        config.setdefault('inference', {}).update({'checkpoint': str(args.checkpoint),
            'results_dir': str(root / 'results'), 'conf_threshold': args.threshold})
        config['dataset']['infer_data_sources'] = {'image_dir': str(images), 'classmap': str(classmap)}
        config['dataset']['batch_size'] = 1
        config['dataset']['workers'] = 0
        if isinstance(config.get('wandb'), dict):
            config['wandb']['enable'] = False
        spec = root / 'scan.yaml'
        spec.write_text(yaml.safe_dump(config, sort_keys=False))
        proc = subprocess.run([args.dino, 'inference', '-e', str(spec)],
                              cwd=root, capture_output=True, text=True, timeout=120)
        if proc.returncode:
            # Keep local diagnostic output off the HTTP surface.
            print(proc.stdout[-5000:] + proc.stderr[-5000:], flush=True)
            raise RuntimeError('TAO inference failed; inspect the worker terminal.')
        candidates = list((root / 'results').rglob('labels/query.txt'))
        if len(candidates) != 1:
            raise RuntimeError('Expected one newly generated TAO label file.')
        detections = parse_labels(candidates[0], image.size)
    return {'schema_version': 1, 'kind': 'fresh_tao_dino_inference',
            'image_sha256': digest, 'image_size': list(image.size),
            'coordinate_space': COORDINATES, 'checkpoint_sha256': checkpoint_sha256,
            'confidence_threshold': args.threshold, 'runtime': 'TAO DINO subprocess',
            'elapsed_seconds': round(time.perf_counter() - started, 3), 'detections': detections}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dino', default='dino')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--threshold', type=float, default=0.25)
    args = parser.parse_args()
    args.spec = args.spec.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must be between zero and one')
    token = os.environ.get('WEBSHOP_DINO_TOKEN', '')
    if len(token) < 24:
        parser.error('Set WEBSHOP_DINO_TOKEN to a random secret of at least 24 characters')
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self):
            self.connection.settimeout(30)
            if self.path != '/detect':
                return self.reply(404, {'error': 'not_found'})
            if not hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token):
                return self.reply(401, {'error': 'unauthorized'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 29 * 1024 * 1024:
                    return self.reply(413, {'error': 'invalid_request_size'})
                payload = json.loads(self.rfile.read(length))
                result = infer(payload, args, digest)
            except (ValueError, KeyError, OSError):
                return self.reply(400, {'error': 'invalid_image_request'})
            except (RuntimeError, subprocess.TimeoutExpired):
                return self.reply(503, {'error': 'inference_failed'})
            self.reply(200, result)

    print(json.dumps({'listening': f'127.0.0.1:{args.port}', 'checkpoint_sha256': digest}), flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
