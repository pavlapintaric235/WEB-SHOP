#!/usr/bin/env python3
"""Isolated NVIDIA Retail DINO + InventoryLens recognition pass for 25 scenes.

Run ``detect`` in the TensorRT 10.14 / ONNX Runtime 1.24 environment, then
``recognize`` in the existing CUDA PyTorch benchmark environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def source_rows(root: Path) -> list[dict]:
    rows = [row for row in rows_csv(root / "data/source/photos.csv") if row["role"] == "scene"]
    if len(rows) != 25:
        raise RuntimeError(f"Expected the 25 scene photos; found {len(rows)}")
    for index, row in enumerate(rows):
        path = root / "data/source/images" / row["relative_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"Missing or changed source photo at manifest row {index}: {path.name}")
    return rows


def detect(args: argparse.Namespace) -> int:
    import cv2
    import onnxruntime as ort

    root = args.root.resolve(strict=True)
    sys.path.insert(0, str(root))
    from nvidia_onnx import DETECTOR_MEAN_BGR, DETECTOR_SCALE, decode_dino_detections

    model_path = args.model.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    photos = source_rows(root)
    output.mkdir(parents=True)
    (output / "overlays").mkdir()
    profile_dir = output / "runtime"
    profile_dir.mkdir()
    engine_cache = args.engine_cache.resolve() if args.engine_cache else output / "engine_cache"
    engine_cache.mkdir(parents=True, exist_ok=True)

    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"TensorRT EP unavailable: {ort.get_available_providers()}")
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_dir / "nvidia_dino_scene")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(engine_cache),
        "trt_fp16_enable": "False",
    }
    session = ort.InferenceSession(
        str(model_path), sess_options=options,
        providers=[("TensorrtExecutionProvider", provider_options), "CUDAExecutionProvider"],
    )
    providers = session.get_providers()
    if not providers or providers[0] != "TensorrtExecutionProvider":
        raise RuntimeError(f"TensorRT is not the selected provider: {providers}")
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) != 4 or shape[1] != 3 or not isinstance(shape[2], int) or not isinstance(shape[3], int):
        raise RuntimeError(f"Unexpected DINO input shape: {shape}")
    net_h, net_w = int(shape[2]), int(shape[3])
    output_names = {item.name for item in session.get_outputs()}
    if not {"pred_logits", "pred_boxes"}.issubset(output_names):
        raise RuntimeError(f"Unexpected DINO outputs: {output_names}")

    def prepare(image_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        height, width = image_bgr.shape[:2]
        scale = min(net_w / width, net_h / height)
        rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(image_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((net_h, net_w, 3), dtype=np.uint8)
        canvas[:rh, :rw] = resized
        tensor = ((canvas.astype(np.float32) - DETECTOR_MEAN_BGR) * DETECTOR_SCALE).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(tensor, dtype=np.float32), scale, (0, 0)

    # Build or load the TensorRT engine before collecting per-image timings.
    warm_image = cv2.imread(str(root / "data/source/images" / photos[0]["relative_path"]), cv2.IMREAD_COLOR)
    if warm_image is None:
        raise ValueError("Unable to read the first scene image for TensorRT warm-up.")
    warm_tensor, _, _ = prepare(warm_image)
    warm_started = time.perf_counter()
    session.run(None, {input_meta.name: warm_tensor})
    warmup_seconds = time.perf_counter() - warm_started

    records, flat_rows = [], []
    for photo_index, photo in enumerate(photos):
        image_path = root / "data/source/images" / photo["relative_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read scene image: {image_path}")
        image_h, image_w = image.shape[:2]
        tensor, scale, pad = prepare(image)
        started = time.perf_counter()
        raw = session.run(None, {input_meta.name: tensor})
        inference_seconds = time.perf_counter() - started
        outputs = dict(zip([item.name for item in session.get_outputs()], raw))
        detections = decode_dino_detections(
            outputs["pred_logits"], outputs["pred_boxes"], (image_w, image_h),
            (net_w, net_h), scale, pad, threshold=args.confidence, max_det=args.max_det,
        )
        line_width = max(3, image_w // 500)
        font_scale = max(0.7, image_w / 1600)
        font_thickness = max(2, image_w // 1200)
        for index, item in enumerate(detections, 1):
            x1, y1, x2, y2 = item["box"]
            box = (round(x1), round(y1), round(x2), round(y2))
            cv2.rectangle(image, box[:2], box[2:], (35, 230, 90), line_width)
            label = f"{index} {item['detector_confidence']:.2f}"
            cv2.putText(image, label, (box[0], max(24, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (35, 230, 90), font_thickness, cv2.LINE_AA)
            flat_rows.append({"photo_index": photo_index, "image": photo["original_filename"],
                              "scene_id": photo["scene_id"], "instance_id": index,
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                              "detector_confidence": item["detector_confidence"]})
        overlay_path = output / "overlays" / f"{photo_index + 1:02d}_{Path(photo['original_filename']).stem}_nvidia_dino.jpg"
        if not cv2.imwrite(str(overlay_path), image, [cv2.IMWRITE_JPEG_QUALITY, 93]):
            raise OSError(f"Unable to write detector overlay: {overlay_path}")
        records.append({
            "photo_index": photo_index, "image": photo["original_filename"],
            "relative_path": photo["relative_path"], "scene_id": photo["scene_id"],
            "source_sha256": photo["sha256"], "width": image_w, "height": image_h,
            "detections": detections, "detector_seconds": inference_seconds,
            "overlay": str(overlay_path.relative_to(output)),
        })
        if photo_index % 5 == 0:
            print(f"DINO [{photo_index + 1}/25] {photo['original_filename']}: {len(detections)} boxes", flush=True)

    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    node_events = [event for event in events if event.get("cat") == "Node" and
                   event.get("ph") == "X" and "provider" in event.get("args", {})]
    provider_counts: dict[str, int] = {}
    for event in node_events:
        provider = event["args"]["provider"]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
    trt_nodes = provider_counts.get("TensorrtExecutionProvider", 0)
    other_nodes = sum(value for key, value in provider_counts.items() if key != "TensorrtExecutionProvider")
    if trt_nodes < 1 or other_nodes:
        raise RuntimeError(f"DINO did not execute exclusively on TensorRT: {provider_counts}")

    (output / "detections.json").write_text(json.dumps({
        "status": "PASS", "images": len(records), "confidence_threshold": args.confidence,
        "max_detections": args.max_det, "model": str(model_path),
        "model_sha256": sha256(model_path), "source_manifest": "data/source/photos.csv",
        "provider_policy": "TensorRT required; CUDA listed as fallback provider; CPU fallback disabled",
        "providers": providers, "input_shape": shape, "warmup_engine_build_seconds": warmup_seconds,
        "profile": str(profile_path), "profile_node_provider_counts": provider_counts,
        "records": records,
    }, indent=2), encoding="utf-8")
    write_csv(output / "detections.csv", flat_rows)
    print(json.dumps({"status": "PASS", "images": len(records),
                      "total_boxes": sum(len(row["detections"]) for row in records),
                      "warmup_seconds": warmup_seconds, "provider_nodes": provider_counts,
                      "output": str(output)}, indent=2), flush=True)
    return 0


def recognize(args: argparse.Namespace) -> int:
    from PIL import Image, ImageOps

    root = args.root.resolve(strict=True)
    sys.path.insert(0, str(root))
    import run_comparative_benchmark as bench

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    photo_rows = bench.read_csv(root / "data/source/photos.csv")
    photo_rows = [row for row in photo_rows if row["role"] == "scene"]
    bench.validate_photos(photo_rows)
    detector_data = json.loads(args.detections.resolve(strict=True).read_text(encoding="utf-8"))
    if detector_data.get("status") != "PASS" or len(detector_data.get("records", [])) != len(photo_rows):
        raise RuntimeError("DINO detection manifest is not a complete PASS for the 25 source scenes.")
    for index, (photo, record) in enumerate(zip(photo_rows, detector_data["records"])):
        if record.get("photo_index") != index or record.get("relative_path") != photo["relative_path"]:
            raise RuntimeError(f"DINO detection/source mapping mismatch at photo index {index}.")
        if record.get("source_sha256") != photo["sha256"]:
            raise RuntimeError(f"DINO detection source hash mismatch at photo index {index}.")

    gpu_name, torch = bench.require_torch_cuda()
    output.mkdir(parents=True)
    (output / "runtime").mkdir()
    (output / "overlays").mkdir()
    products_all = json.loads((root / "data/source/product_registry_experiment.json").read_text(encoding="utf-8"))
    products = [row for row in products_all if row.get("references")]
    encoders, indexes, reference_rows, reference_seconds, encoder_errors = bench.load_encoder_gallery(
        products, output / "runtime")
    required = {"mobilenetv2", "dinov2", "siglip2", "nvidia_fan_base_hybrid"}
    if required - set(encoders):
        raise RuntimeError(f"Required recognizer(s) failed to load: {encoder_errors}")
    if len(reference_rows) != 55 or len({row["sku"] for row in reference_rows}) != 11:
        raise RuntimeError(f"Unexpected reference gallery: {len(reference_rows)} images / "
                           f"{len({row['sku'] for row in reference_rows})} SKUs; expected 55 / 11.")
    sam = bench.MobileSAMBoxRefiner()

    prediction_rows, image_rows = [], []
    visual_mode = args.overlay_mode
    if visual_mode not in bench.MODES:
        raise ValueError(f"Unknown overlay mode {visual_mode}; choose from {bench.MODES}")
    for photo_index, (photo, detector_record) in enumerate(zip(photo_rows, detector_data["records"])):
        with Image.open(root / "data/source/images" / photo["relative_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        boxes = detector_record["detections"]
        masked, sam_times = sam.refine(image, boxes, "nvidia_retail_dino_plus_mobile_sam")
        variants = [("boxes", boxes, {"wall_seconds": 0.0, "gpu_forward_seconds": 0.0}),
                    ("mobile_sam_masks", masked, sam_times)]
        for variant, detections, mask_times in variants:
            crops, crop_seconds = bench.crops_for(image, detections)
            recognitions, encoder_times = bench.recognize_all(crops, encoders, indexes)
            overlay = bench.render_overlay(image, detections, recognitions, visual_mode)
            overlay_name = f"{photo_index + 1:02d}_{Path(photo['original_filename']).stem}__{variant}__{visual_mode.replace('+', '_')}.jpg"
            overlay_path = output / "overlays" / overlay_name
            overlay.save(overlay_path, quality=92)
            for instance_index, (detection, per_mode, crop) in enumerate(zip(detections, recognitions, crops), 1):
                for mode in bench.MODES:
                    if mode not in per_mode:
                        continue
                    prediction_rows.append({
                        "photo_index": photo_index, "scene_id": photo["scene_id"],
                        "image": photo["original_filename"], "crop_variant": variant,
                        "instance_id": instance_index, "box_xyxy": detection["box"],
                        "mask_polygons": detection.get("mask_polygons", []),
                        "detector_confidence": detection.get("detector_confidence"),
                        "crop_width": crop.width, "crop_height": crop.height,
                        "recognizer": mode, **per_mode[mode],
                    })
            recognized_counts = {}
            for mode in bench.MODES:
                mode_items = [item.get(mode, {}) for item in recognitions]
                counts = {}
                for item in mode_items:
                    if item.get("status") == "recognized" and item.get("sku"):
                        counts[item["sku"]] = counts.get(item["sku"], 0) + 1
                recognized_counts[mode] = {
                    "recognized": sum(item.get("status") == "recognized" for item in mode_items),
                    "unknown_or_abstained": sum(item.get("status") != "recognized" for item in mode_items),
                    "counts_by_sku": counts,
                }
            image_rows.append({
                "photo_index": photo_index, "scene_id": photo["scene_id"],
                "image": photo["original_filename"], "crop_variant": variant,
                "detection_count": len(detections),
                "nvidia_detector_seconds": detector_record["detector_seconds"],
                "mobile_sam_wall_seconds": mask_times["wall_seconds"],
                "mobile_sam_gpu_seconds": mask_times["gpu_forward_seconds"],
                "crop_seconds": crop_seconds, "recognizer_timing": encoder_times,
                "per_recognizer_counts": recognized_counts,
                "overlay": str(overlay_path.relative_to(output)),
            })
        if photo_index % 5 == 0:
            print(f"Recognition [{photo_index + 1}/25] {photo['original_filename']}: "
                  f"{len(boxes)} DINO boxes, {len(masked)} SAM masks", flush=True)

    runtime_audits = {}
    for name, encoder in encoders.items():
        runtime = getattr(encoder, "runtime", None)
        if runtime is not None:
            runtime_audits[name] = runtime.finish_and_audit()
    report = {
        "status": "completed",
        "experiment": "NVIDIA Retail DINO detector + InventoryLens few-shot recognition on 25 development scenes",
        "device": gpu_name,
        "detector": {"model_sha256": detector_data["model_sha256"],
                     "confidence_threshold": detector_data["confidence_threshold"],
                     "detections_json": str(args.detections),
                     "runtime_profile_node_provider_counts": detector_data["profile_node_provider_counts"]},
        "images": 25,
        "reference_gallery_images": len(reference_rows),
        "reference_gallery_skus": sorted({row["sku"] for row in reference_rows}),
        "reference_setup_seconds": reference_seconds,
        "recognizers": list(encoders), "recognizer_load_errors": encoder_errors,
        "fixed_thresholds": {name: {"values": values, "source": bench.THRESHOLD_SOURCE}
                             for name, values in bench.THRESHOLDS.items()},
        "nvidia_fan_threshold": "not calibrated; ranking reported, accepted SKU remains unknown",
        "runtime_audits": runtime_audits,
        "output_counts": {"detection_variants": 2, "per_image_rows": len(image_rows),
                          "per_detection_recognizer_rows": len(prediction_rows)},
        "scoring_limit": "No per-object ground truth for these 25 scenes; SKU correctness, detection precision/recall, mask IoU, duplicate rate, and unknown-rejection accuracy are not scored.",
        "counting_limit": "Photos from the same arrangement are retained separately; counts are not summed across views. Only source-manifest verified totals can be compared, not treated as per-object recognition accuracy.",
        "reference_coverage_limit": "This packaged experimental gallery contains 55 images for 11 mapped experiment SKUs, not every catalogue registration.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps({"report": report, "images": image_rows,
                                                        "predictions": prediction_rows}, indent=2), encoding="utf-8")
    write_csv(output / "per_image_results.csv", image_rows)
    write_csv(output / "per_detection_predictions.csv", prediction_rows)
    lines = ["# NVIDIA DINO 25-Scene End-to-End Run", "",
             f"Status: **{report['status']}** on **{gpu_name}**.", "",
             "NVIDIA Retail DINO boxes were passed to the existing MobileNetV2, DINOv2-small, SigLIP2, NVIDIA FAN, and the two existing fused matchers. Each box was also refined by the existing MobileSAM model; both box crops and masked crops were recognized.", "",
             f"Reference gallery used: **{len(reference_rows)} photos across {len(report['reference_gallery_skus'])} mapped SKUs**.", "",
             "This is an inference/prediction report, not an accuracy benchmark: the 25 scenes have no object-level SKU labels. Top-1/top-5 rankings, abstentions, candidate counts, and runtimes are provided, but they cannot establish recognition accuracy. NVIDIA FAN has no calibrated acceptance threshold and therefore abstains.", "",
             "The source package's fixed thresholds were reused unchanged. Cross-view counts are never added. The report does not cover all catalogue registrations; the packaged gallery is limited to 11 mapped experiment SKUs.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "images": len(photo_rows),
                      "gallery_images": len(reference_rows), "gallery_skus": len(report["reference_gallery_skus"]),
                      "prediction_rows": len(prediction_rows), "output": str(output)}, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    detect_parser = sub.add_parser("detect", help="TensorRT-only NVIDIA detector stage")
    detect_parser.add_argument("--root", type=Path, required=True)
    detect_parser.add_argument("--model", type=Path, required=True, help="Working domain-remapped NVIDIA DINO ONNX")
    detect_parser.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    detect_parser.add_argument("--engine-cache", type=Path)
    detect_parser.add_argument("--confidence", type=float, default=0.50)
    detect_parser.add_argument("--max-det", type=int, default=200)
    detect_parser.set_defaults(func=detect)
    recog_parser = sub.add_parser("recognize", help="Existing PyTorch encoders + MobileSAM stage")
    recog_parser.add_argument("--root", type=Path, required=True)
    recog_parser.add_argument("--detections", type=Path, required=True)
    recog_parser.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    recog_parser.add_argument("--overlay-mode", default="siglip2+mobilenetv2")
    recog_parser.set_defaults(func=recognize)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
