#!/usr/bin/env python3
"""NVIDIA Retail DINO + paired SigLIP2 gallery rankings for isolated scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
MANIFEST_PATH = INPUT / "manifest.json"
MEAN_BGR = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
SCALE = 0.01735207357279195
SIGLIP_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if len(data["scenes"]) != 30 or len(data["reference_gallery"]) != 34:
        raise RuntimeError("Input manifest does not match the audited 30-scene / 34-reference package.")
    if any(not all(key in row for key in ("box_sha256", "masked_sha256", "box_archive_path", "masked_archive_path"))
           for row in data["reference_gallery"]):
        raise RuntimeError("Paired original-box and masked references are required.")
    return data


def rank_gallery(query: np.ndarray, references: np.ndarray,
                 by_sku: dict[str, list[int]]) -> dict[str, float]:
    """Mean of each SKU's two highest cosine similarities; query is fixed across galleries."""
    sims = references @ query
    return {sku_id: float(np.mean(sorted((float(sims[idx]) for idx in indices), reverse=True)[:2]))
            for sku_id, indices in by_sku.items()}


def top_three(scores: dict[str, float]) -> list[tuple[float, str]]:
    return sorted(((score, sku_id) for sku_id, score in scores.items()),
                  key=lambda item: (-item[0], item[1]))[:3]


def detect(args: argparse.Namespace) -> int:
    import cv2
    import onnxruntime as ort
    import tensorrt as trt
    from PIL import Image, ImageOps

    data = load_manifest()
    if not 1 <= args.limit_scenes <= len(data["scenes"]):
        raise ValueError("--limit-scenes must be between 1 and the 30 packaged scenes")
    model = args.model.resolve(strict=True)
    if sha256(model) != args.expected_model_sha256:
        raise RuntimeError("NVIDIA Retail DINO model SHA-256 does not match the verified remapped model.")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"TensorRT EP unavailable: {ort.get_available_providers()}")
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    (output / "runtime").mkdir()
    engine_cache = args.engine_cache.resolve()
    engine_cache.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output / "runtime" / "dino_profile")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    providers = [
        ("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": "True",
            "trt_engine_cache_path": str(engine_cache),
            "trt_fp16_enable": "False",
        }),
        ("CUDAExecutionProvider", {"device_id": 0}),
    ]
    session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
    if not session.get_providers() or session.get_providers()[0] != "TensorrtExecutionProvider":
        raise RuntimeError(f"TensorRT did not become the primary provider: {session.get_providers()}")
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) != 4 or shape[1] != 3 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise RuntimeError(f"Unexpected static model input shape: {shape}")
    net_h, net_w = int(shape[2]), int(shape[3])
    output_names = [item.name for item in session.get_outputs()]
    if not {"pred_boxes", "pred_logits"}.issubset(output_names):
        raise RuntimeError(f"Unexpected DINO outputs: {output_names}")

    def prep(image: Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        scale = min(net_w / width, net_h / height)
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((net_h, net_w, 3), dtype=np.uint8)
        canvas[:rh, :rw] = resized
        bgr = canvas[:, :, ::-1].astype(np.float32)
        tensor = ((bgr - MEAN_BGR) * SCALE).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(tensor, dtype=np.float32), scale, width, height

    def parse(raw, scale, width, height):
        values = dict(zip(output_names, raw))
        logits = np.asarray(values["pred_logits"], dtype=np.float32).reshape(-1, 2)
        boxes = np.asarray(values["pred_boxes"], dtype=np.float32).reshape(-1, 4)
        logits = np.clip(logits, -80, 80)
        probs = np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))
        classes = probs.argmax(axis=1)
        scores = probs[np.arange(len(probs)), classes]
        detections = []
        for idx in np.argsort(scores)[::-1]:
            score = float(scores[idx])
            if classes[idx] != 1 or score < args.confidence:
                continue
            cx, cy, bw, bh = boxes[idx]
            box = [
                float(np.clip((cx - bw / 2) * net_w / scale, 0, width)),
                float(np.clip((cy - bh / 2) * net_h / scale, 0, height)),
                float(np.clip((cx + bw / 2) * net_w / scale, 0, width)),
                float(np.clip((cy + bh / 2) * net_h / scale, 0, height)),
            ]
            if box[2] > box[0] and box[3] > box[1]:
                detections.append({"box_xyxy": box, "detector_confidence": score})
            if len(detections) >= args.max_det:
                break
        return detections

    from PIL import ImageDraw
    scene_records = []
    flat = []
    warmup_seconds = 0.0
    for index, record in enumerate(data["scenes"][:args.limit_scenes]):
        path = INPUT / "scenes" / record["photo_filename"]
        if sha256(path) != record["source_sha256"]:
            raise RuntimeError(f"Scene image hash mismatch: {record['photo_filename']}")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        tensor, scale, width, height = prep(image)
        started = time.perf_counter()
        raw = session.run(None, {input_meta.name: tensor})
        elapsed = time.perf_counter() - started
        if index == 0:
            warmup_seconds = elapsed
        detections = parse(raw, scale, width, height)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for det_index, det in enumerate(detections, 1):
            x1, y1, x2, y2 = det["box_xyxy"]
            draw.rectangle((x1, y1, x2, y2), outline=(25, 210, 100), width=max(3, width // 500))
            draw.text((x1 + 3, max(0, y1 + 3)), f"{det_index} DINO {det['detector_confidence']:.2f}",
                      fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
            flat.append({
                "photo_filename": record["photo_filename"],
                "dataset_type": record["dataset_type"],
                "detection_id": f"det-{det_index}",
                "detector_confidence": det["detector_confidence"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        overlay_name = f"{index + 1:02d}_{Path(record['photo_filename']).stem}_dino.jpg"
        overlay.save(output / "overlays" / overlay_name, quality=92)
        scene_records.append({
            **record, "width": width, "height": height, "detections": detections,
            "detector_seconds": elapsed, "overlay": f"overlays/{overlay_name}",
        })
        print(f"DINO [{index + 1}/{args.limit_scenes}] {record['photo_filename']}: {len(detections)} boxes", flush=True)

    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text(encoding="utf-8"))
    node_events = [event for event in events if event.get("cat") == "Node" and event.get("ph") == "X"
                   and "provider" in event.get("args", {})]
    counts = {}
    for event in node_events:
        provider = event["args"]["provider"]
        counts[provider] = counts.get(provider, 0) + 1
    if counts.get("TensorrtExecutionProvider", 0) < 1 or any(
            provider != "TensorrtExecutionProvider" for provider in counts):
        raise RuntimeError(f"DINO node provider audit failed: {counts}")
    with (output / "detections.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]) if flat else [
            "photo_filename", "dataset_type", "detection_id", "detector_confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(flat)
    report = {
        "status": "PASS", "stage": "NVIDIA Retail DINO detection",
        "images": len(scene_records), "total_boxes": len(flat),
        "runtime_versions": {"opencv": cv2.__version__, "onnxruntime": ort.__version__,
                             "tensorrt": trt.__version__},
        "confidence_threshold": args.confidence, "max_detections": args.max_det,
        "model_sha256": sha256(model), "input_shape": shape,
        "providers": session.get_providers(), "profile_node_provider_counts": counts,
        "profile_path": str(profile), "first_image_seconds_including_engine_warmup": warmup_seconds,
        "per_image_inference_seconds_sum": sum(row["detector_seconds"] for row in scene_records),
        "records": scene_records,
    }
    (output / "detections.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2), flush=True)
    return 0


def recognize(args: argparse.Namespace) -> int:
    import torch
    from PIL import Image, ImageDraw, ImageOps
    from transformers import AutoImageProcessor, AutoModel

    data = load_manifest()
    detector_dir = args.detector_output.resolve(strict=True)
    detections = json.loads((detector_dir / "detections.json").read_text(encoding="utf-8"))
    if detections.get("status") != "PASS" or len(detections.get("records", [])) != 30:
        raise RuntimeError("DINO results are incomplete; refusing recognition stage.")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable; refusing CPU SigLIP2 inference.")
    device = torch.device("cuda:0")
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0),
                      "visible_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)}),
          flush=True)

    cache_dir = args.cache_dir
    load_kwargs = {"revision": SIGLIP_REVISION, "local_files_only": not args.allow_model_download}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    model_started = time.perf_counter()
    processor = AutoImageProcessor.from_pretrained(args.siglip_model, use_fast=False, **load_kwargs)
    model = AutoModel.from_pretrained(args.siglip_model, **load_kwargs).eval().requires_grad_(False).to(device)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_started

    def encode_images(images, batch_size: int):
        vectors = []
        total_seconds = 0.0
        with torch.inference_mode():
            for offset in range(0, len(images), batch_size):
                batch = images[offset:offset + batch_size]
                inputs = processor(images=[image.convert("RGB") for image in batch], return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                features = model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features, dim=1)
                end.record()
                end.synchronize()
                total_seconds += start.elapsed_time(end) / 1000.0
                vectors.append(features.float().cpu().numpy())
        return np.concatenate(vectors, axis=0), total_seconds

    ref_rows = data["reference_gallery"]
    ref_features_by_gallery = {}
    reference_gpu_seconds = {}
    for representation in ("box", "masked"):
        ref_images = []
        for record in ref_rows:
            path = ROOT / record[f"{representation}_archive_path"]
            if sha256(path) != record[f"{representation}_sha256"]:
                raise RuntimeError(f"Reference crop hash mismatch: {path.name}")
            with Image.open(path) as source:
                ref_images.append(ImageOps.exif_transpose(source).convert("RGB"))
        ref_features_by_gallery[representation], reference_gpu_seconds[representation] = encode_images(ref_images, args.batch_size)
    by_sku: dict[str, list[int]] = {}
    for idx, row in enumerate(ref_rows):
        by_sku.setdefault(row["sku_id"], []).append(idx)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlays").mkdir()
    prediction_rows, scene_rows = [], []
    all_query_features = []
    all_query_ids = []
    recognition_gpu_seconds = 0.0
    for image_index, scene in enumerate(detections["records"]):
        path = INPUT / "scenes" / scene["photo_filename"]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        records = scene["detections"]
        crops = []
        for det in records:
            x1, y1, x2, y2 = det["box_xyxy"]
            crop = image.crop((max(0, int(x1)), max(0, int(y1)), min(image.width, int(np.ceil(x2))), min(image.height, int(np.ceil(y2)))))
            if crop.width < 2 or crop.height < 2:
                crop = Image.new("RGB", (2, 2), "black")
            crops.append(crop)
        if crops:
            crop_features, gpu_seconds = encode_images(crops, args.batch_size)
            recognition_gpu_seconds += gpu_seconds
            all_query_features.append(crop_features)
            all_query_ids.extend(f"{scene['photo_filename']}:det-{index}" for index in range(1, len(records) + 1))
        else:
            crop_features = np.empty((0, ref_features_by_gallery["box"].shape[1]), dtype=np.float32)

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for det_index, (det, query) in enumerate(zip(records, crop_features), 1):
            box_scores = rank_gallery(query, ref_features_by_gallery["box"], by_sku)
            masked_scores = rank_gallery(query, ref_features_by_gallery["masked"], by_sku)
            combined_scores = {sku_id: (box_scores[sku_id] + masked_scores[sku_id]) / 2
                               for sku_id in by_sku}
            rankings = {"original": top_three(box_scores),
                        "masked": top_three(masked_scores),
                        "combined": top_three(combined_scores)}
            best_score, best_sku = rankings["original"][0]
            sku_names = {item["sku_id"]: item["proposed_name"] for item in data["candidate_skus"]}
            x1, y1, x2, y2 = det["box_xyxy"]
            draw.rectangle((x1, y1, x2, y2), outline=(255, 165, 0), width=max(3, image.width // 500))
            label = f"{det_index} O:{best_sku} M:{rankings['masked'][0][1]}"
            draw.text((x1 + 3, max(0, y1 + 3)), label, fill=(0, 0, 0), stroke_width=2, stroke_fill="white")
            row = {
                "photo_filename": scene["photo_filename"],
                "dataset_type": scene["dataset_type"],
                "detection_id": f"det-{det_index}",
                "detector_confidence": det["detector_confidence"],
                "top1_sku_id": best_sku, "top1_name": sku_names[best_sku], "top1_similarity": best_score,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "reference_count_for_top1": len(by_sku[best_sku]),
                "query_crop_sha256": hashlib.sha256(np.asarray(crops[det_index - 1]).tobytes()).hexdigest(),
                "recognition_status": "unaccepted_provisional_rank_no_unknown_threshold",
            }
            for gallery_name, top in rankings.items():
                for rank, (score, sku_id) in enumerate(top, 1):
                    row[f"{gallery_name}_top{rank}_sku_id"] = sku_id
                    row[f"{gallery_name}_top{rank}_name"] = sku_names[sku_id]
                    row[f"{gallery_name}_top{rank}_similarity"] = score
            for rank, (score, sku_id) in enumerate(rankings["original"][1:], 2):
                row[f"top{rank}_sku_id"] = sku_id
                row[f"top{rank}_name"] = sku_names[sku_id]
                row[f"top{rank}_similarity"] = score
            prediction_rows.append(row)
        overlay_name = f"{image_index + 1:02d}_{Path(scene['photo_filename']).stem}_dino_siglip2.jpg"
        overlay.save(output / "overlays" / overlay_name, quality=92)
        scene_rows.append({
            "photo_filename": scene["photo_filename"], "dataset_type": scene["dataset_type"],
            "detection_count": len(records), "overlay": f"overlays/{overlay_name}",
            "detector_seconds": scene["detector_seconds"],
        })
        print(f"Recognized [{image_index + 1}/30] {scene['photo_filename']}: {len(records)} boxes", flush=True)

    if not prediction_rows:
        raise RuntimeError("DINO returned no boxes across the full scene set; no recognition predictions were produced.")
    np.savez_compressed(output / "embeddings.npz",
                        original_references=ref_features_by_gallery["box"],
                        masked_references=ref_features_by_gallery["masked"],
                        queries=np.concatenate(all_query_features, axis=0),
                        query_ids=np.asarray(all_query_ids),
                        reference_sku_ids=np.asarray([row["sku_id"] for row in ref_rows]),
                        reference_source_filenames=np.asarray([row["source_photo_filename"] for row in ref_rows]))
    fields = list(prediction_rows[0])
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction_rows)
    report = {
        "status": "PASS", "experiment": "new_product_scanning_20261020",
        "scope": "30 held-out employee/customer scenes; inference only; human review pending",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "detector": {"name": "NVIDIA Retail DINO", "model_sha256": detections["model_sha256"],
                     "threshold": detections["confidence_threshold"],
                     "provider_nodes": detections["profile_node_provider_counts"]},
        "recognizer": {"name": "SigLIP2 base patch16 224", "revision": SIGLIP_REVISION,
                       "gallery_images": len(ref_rows), "gallery_skus": len(by_sku),
                       "gallery_ids": {key: len(value) for key, value in by_sku.items()},
                       "gallery_conditions": ["original", "masked", "combined"],
                       "original_definition": "unmasked approved owner box crops",
                       "masked_definition": "same owner photos and crop windows, background replaced by gray",
                       "combined_definition": "mean of original and masked per-SKU scores",
                       "per_sku_score": "mean of the two highest cosine similarities per SKU in each representation",
                       "same_detector_proposals_and_query_embeddings_across_galleries": True,
                       "similarities_are_calibrated_confidence": False,
                       "unknown_rejection_threshold": None},
        "scene_images": len(scene_rows), "detected_boxes": len(prediction_rows),
        "embeddings_file": "embeddings.npz",
        "engine_build_and_first_detector_warmup_seconds": detections["first_image_seconds_including_engine_warmup"],
        "detector_inference_seconds_sum": sum(row["detector_seconds"] for row in scene_rows),
        "siglip_model_load_seconds": model_load_seconds,
        "reference_embedding_gpu_seconds": reference_gpu_seconds,
        "scene_embedding_gpu_seconds": recognition_gpu_seconds,
        "accuracy_metrics": None,
        "limitations": [
            "Scene predictions are provisional ranks, not verified labels or accuracy scores.",
            "No unknown rejection is applied because no acceptance threshold was calibrated.",
            "The current IMG_4857 human annotation has duplicate boxes and is not used for scoring.",
            "Approved MobileSAM masks change registration crops only; displayed scene regions are DINO boxes.",
            "Counts are per photograph only; do not add counts across views of one arrangement.",
        ],
        "scenes": scene_rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "scenes"}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    det = sub.add_parser("detect")
    det.add_argument("--model", type=Path, required=True)
    det.add_argument("--expected-model-sha256", default="c6185e3b1cd01abfe9d83e7f6f13083a8485b6cd0ca409642f237a5c65c36f41")
    det.add_argument("--output", type=Path, required=True)
    det.add_argument("--engine-cache", type=Path, required=True)
    det.add_argument("--confidence", type=float, default=0.50)
    det.add_argument("--max-det", type=int, default=200)
    det.add_argument("--limit-scenes", type=int, default=30)
    det.set_defaults(func=detect)
    rec = sub.add_parser("recognize")
    rec.add_argument("--detector-output", type=Path, required=True)
    rec.add_argument("--output", type=Path, required=True)
    rec.add_argument("--siglip-model", default=SIGLIP_ID)
    rec.add_argument("--cache-dir", default=os.environ.get("WEBSHOP_HF_CACHE"))
    rec.add_argument("--allow-model-download", action="store_true")
    rec.add_argument("--batch-size", type=int, default=16)
    rec.set_defaults(func=recognize)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
