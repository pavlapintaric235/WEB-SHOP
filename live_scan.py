"""Fresh-photo DINO transport and shop-scoped SigLIP2 retrieval.

Archived detection replay is explicit and bound to the exact normalized image.
The worker receives normalized upright pixels; original photo bytes are preserved.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DEFAULT_MODEL = (ROOT / '.venv/huggingface_cache/hub/models--google--siglip2-base-patch16-224'
                 / 'snapshots' / SIGLIP_REVISION)
MAX_BYTES = 20 * 1024 * 1024
COORDINATES = 'EXIF-corrected upright image pixels; xyxy'


def normalize_photo(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Choose a readable photo of at most 20 MiB.')
    with Image.open(io.BytesIO(data)) as original:
        if original.width * original.height > 30_000_000:
            raise ValueError('Photo exceeds 30 megapixels.')
        image = ImageOps.exif_transpose(original).convert('RGB')
    # Serialize pixels without the source orientation tag: rotate exactly once.
    stream = io.BytesIO()
    clean = Image.fromarray(np.asarray(image))
    clean.save(stream, format='PNG')
    if len(stream.getvalue()) > MAX_BYTES:
        raise ValueError('The upright lossless image exceeds 20 MiB; choose a lower-resolution photo.')
    return clean, stream.getvalue()


def valid_box(box, size):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('Each item needs four box coordinates.')
    values = [float(v) for v in box]
    x1, y1, x2, y2 = values
    if not all(math.isfinite(v) for v in values) or not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
        raise ValueError('An item box is outside the upright photo.')
    return values


def validate_worker_result(result, image_bytes, size):
    if result.get('image_sha256') != hashlib.sha256(image_bytes).hexdigest():
        raise ValueError('Detector response belongs to a different photo.')
    if result.get('coordinate_space') != COORDINATES or result.get('image_size') != list(size):
        raise ValueError('Detector response uses different image coordinates.')
    if result.get('kind') != 'fresh_tao_dino_inference':
        raise ValueError('A fresh TAO DINO result is required.')
    checkpoint = result.get('checkpoint_sha256', '')
    if len(checkpoint) != 64 or any(c not in '0123456789abcdef' for c in checkpoint):
        raise ValueError('Detector response has no valid checkpoint hash.')
    expected = os.environ.get('WEBSHOP_DINO_SHA256', '8bb11d6d63f26edfd480bdf1264467bd5c080f190fbb83a034b252d13ee20d27')
    if checkpoint != expected:
        raise ValueError('Detector worker is running a different checkpoint.')
    detections = result.get('detections')
    if not isinstance(detections, list) or len(detections) > 900:
        raise ValueError('Invalid detector proposal list.')
    for row in detections:
        row['box'] = valid_box(row['box'], size)
        score = float(row['confidence'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid detector confidence.')
    return result


def request_dino(image_bytes, size):
    endpoint = os.environ.get('WEBSHOP_DINO_URL', '').rstrip('/')
    token = os.environ.get('WEBSHOP_DINO_TOKEN', '')
    if not endpoint or not token:
        raise RuntimeError('The DINO worker is not connected. Configure WEBSHOP_DINO_URL and WEBSHOP_DINO_TOKEN. No saved predictions have been substituted.')
    parsed = urlparse(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use a worker base URL without credentials or query parameters.')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost', '::1'}):
        raise ValueError('Use HTTPS for the worker, or HTTP through a local SSH tunnel.')
    payload = json.dumps({'image_b64': base64.b64encode(image_bytes).decode('ascii'),
                          'image_sha256': hashlib.sha256(image_bytes).hexdigest()}).encode()
    request = Request(endpoint + '/detect', data=payload,
                      headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=150) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError('Detector response exceeds its size limit.')
            result = json.loads(raw)
    except HTTPError as error:
        raise RuntimeError(f'DINO worker rejected the request (HTTP {error.code}); no predictions were used.') from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError('The OpenCV capture check completed, but product inference did not run: the DINO worker or its private SSH tunnel is unreachable or timed out. Reconnect both services and retry. No SKU predictions or inventory changes were made.') from error
    return validate_worker_result(result, image_bytes, size)


def replay_dino(archive, image_bytes, size):
    """Validate original live provenance, then label its reuse as replay."""
    recorded = json.loads(Path(archive).read_text(encoding='utf-8'))
    detector = validate_worker_result(recorded['detector'], image_bytes, size)
    return {**detector, 'kind': 'archived_detection_replay',
            'original_elapsed_seconds': detector.get('elapsed_seconds'),
            'replay_archive': str(archive)}


def reference_records(products):
    from images import product_references
    from sam_vision import REFERENCE_CACHE_VERSION
    records = []
    for product in products:
        for ref in product_references(product):
            path = Path(ref['image_path'])
            data = path.read_bytes()
            key = hashlib.sha256(REFERENCE_CACHE_VERSION.encode('ascii') + data).hexdigest()
            cached = ROOT / 'data/models/reference_cache' / (key + '.npz')
            if cached.is_file():
                with np.load(cached, allow_pickle=False) as archive:
                    crop = Image.fromarray(archive['crop'].astype(np.uint8)).convert('RGB')
                representation = 'existing_cached_product_crop'
            else:
                with Image.open(io.BytesIO(data)) as original:
                    crop = ImageOps.exif_transpose(original).convert('RGB')
                representation = 'full_reference_photo'
            records.append({'sku': product['sku'], 'name': product['name'], 'crop': crop,
                            'source_sha256': hashlib.sha256(data).hexdigest(),
                            'representation': representation})
    if not records:
        raise ValueError('This shop has no registered reference photographs.')
    return records


def reference_fingerprint(products):
    """Bind the app/cache to file contents, not just unchanged reference paths."""
    from images import product_references
    records = [(p['sku'], p['name'], ref['image_path'],
                hashlib.sha256(Path(ref['image_path']).read_bytes()).hexdigest())
               for p in products for ref in product_references(p)]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


class ShopMatcher:
    def __init__(self, products, encoder=None):
        self.records = reference_records(products)
        if encoder is None:
            from experiments.hybrid_pipeline.hf_encoders import HuggingFaceImageEncoder
            model = Path(os.environ.get('WEBSHOP_SIGLIP_PATH', str(DEFAULT_MODEL)))
            if not (model / 'model.safetensors').is_file():
                raise RuntimeError('The pinned SigLIP2 model is not cached locally.')
            encoder = HuggingFaceImageEncoder(str(model), 'siglip2', device='cpu',
                                             local_files_only=True, revision=SIGLIP_REVISION)
        self.encoder = encoder
        self.features = encoder.encode([r['crop'] for r in self.records], batch_size=1)
        self.by_sku = {}
        for index, row in enumerate(self.records):
            self.by_sku.setdefault(row['sku'], []).append(index)

    def rank_boxes(self, image, detections, minimum=0.55, margin=0.05):
        from experiments.hybrid_pipeline.hybrid_core import abstaining_decision
        if not detections:
            return []
        boxes = [valid_box(row['box'], image.size) for row in detections]
        crops = [image.crop((math.floor(b[0]), math.floor(b[1]), math.ceil(b[2]), math.ceil(b[3]))) for b in boxes]
        vectors = self.encoder.encode(crops, batch_size=1)
        output = []
        for number, (detection, vector, box) in enumerate(zip(detections, vectors, boxes), 1):
            sims = self.features @ vector
            scores = {sku: float(np.mean(sorted((float(sims[i]) for i in indices), reverse=True)[:2]))
                      for sku, indices in self.by_sku.items()}
            decision = abstaining_decision(scores, minimum, margin)
            output.append({'item_id': number, 'box': box, 'detector_confidence': detection.get('confidence'),
                           **decision})
        return output


def render_items(image, rows):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(14, image.width // 75))
    for index, row in enumerate(rows, 1):
        box = valid_box(row['box'], image.size)
        color = ('#15803d' if row.get('confirmed') else '#2563eb' if row.get('origin') == 'manual'
                 else '#d97706') if 'confirmed' in row else ('#15803d' if row.get('sku') else '#d97706')
        draw.rectangle(box, outline=color, width=max(2, image.width // 500))
        label = f"#{row.get('item_id', index)} {row.get('sku') or 'Unknown / review'}"
        draw.text((box[0] + 4, box[1] + 4), label, font=font, fill='white', stroke_width=2, stroke_fill='black')
    return canvas


def review_counts(rows, allowed_skus, size, allow_empty=False):
    """Validate explicit item decisions, including manual additions, before posting."""
    counts = {}
    for row in rows:
        valid_box([row['x1'], row['y1'], row['x2'], row['y2']], size)
        if row.get('exclude') is True:
            continue
        if row.get('confirmed') is not True:
            raise ValueError('Confirm or exclude every item, including uncertain proposals.')
        sku = row.get('sku')
        if sku not in allowed_skus:
            raise ValueError('Select a registered shop SKU for every counted item.')
        quantity = float(row.get('quantity', 1))
        if not math.isfinite(quantity) or quantity != int(quantity) or not 1 <= quantity <= 10000:
            raise ValueError('Use a positive whole-number quantity of at most 10000.')
        counts[sku] = counts.get(sku, 0) + int(quantity)
    if not counts and not allow_empty:
        raise ValueError('No reviewed products are included in this count.')
    return counts
