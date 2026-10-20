#!/usr/bin/env bash
set -euo pipefail

BASE=/workspace/new_product_scanning
INPUT_ZIP=${1:-/workspace/runpod_scene_inference_input_20261020_v3.zip}
TRT_PY=${TRT_PY:-/workspace/trt1014-probe-20261019/bin/python}
TORCH_PY=${TORCH_PY:-/workspace/webshop-venv/bin/python}
MODEL=${DINO_MODEL:-/workspace/ngc_detector_probe_20261019/retail_detector_domain_remap.onnx}
MODEL_SHA256=${DINO_MODEL_SHA256:-c6185e3b1cd01abfe9d83e7f6f13083a8485b6cd0ca409642f237a5c65c36f41}
SIGLIP_MODEL=${SIGLIP_MODEL:-/workspace/webshop_benchmark/models/siglip2}
LIMIT_SCENES=${LIMIT_SCENES:-30}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=$BASE/results/$RUN_ID
DET=$OUT/detector
REC=$OUT/recognition
CACHE=$BASE/runtime/tensorrt_engine_cache

command -v unzip >/dev/null
command -v zip >/dev/null
command -v sha256sum >/dev/null
test -x "$TRT_PY"
test -x "$TORCH_PY"
test -f "$MODEL"
test -d "$SIGLIP_MODEL"
test -f "$SIGLIP_MODEL/config.json"
test -f "$INPUT_ZIP"
mkdir -p "$BASE"
unzip -q -o "$INPUT_ZIP" -d "$BASE"
test -f "$BASE/input/manifest.json"

TRT_SITE=$("$TRT_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
TRT_LIBS=$(find "$TRT_SITE" -type d \( -name tensorrt_libs -o -path '*/nvidia/*/lib' \) -print 2>/dev/null | paste -sd: -)
export LD_LIBRARY_PATH="${TRT_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$TRT_PY" -c 'import cv2, onnxruntime as ort, tensorrt as trt; print("DINO_RUNTIME", cv2.__version__, ort.__version__, trt.__version__, ort.get_available_providers()); assert "TensorrtExecutionProvider" in ort.get_available_providers()'
"$TORCH_PY" -c 'import torch, transformers; print("SIGLIP_RUNTIME", torch.__version__, transformers.__version__, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'

"$TRT_PY" "$BASE/scripts/run_scene_inference.py" detect \
  --model "$MODEL" --expected-model-sha256 "$MODEL_SHA256" \
  --output "$DET" --engine-cache "$CACHE" --confidence 0.50 --limit-scenes "$LIMIT_SCENES"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  printf 'PREFLIGHT_DETECTOR_REPORT=%s\n' "$DET/report.json"
  exit 0
fi

export YOLO_CONFIG_DIR="$BASE/runtime/ultralytics"
export MPLCONFIGDIR="$BASE/runtime/matplotlib"
"$TORCH_PY" "$BASE/scripts/run_scene_inference.py" recognize \
  --detector-output "$DET" --output "$REC" --siglip-model "$SIGLIP_MODEL" --batch-size 16

cp "$DET/detections.json" "$OUT/detections.json"
cp "$DET/detections.csv" "$OUT/detections.csv"
cp "$REC/report.json" "$OUT/report.json"
cp "$REC/predictions.csv" "$OUT/predictions.csv"
cp "$REC/embeddings.npz" "$OUT/embeddings.npz"
cp -a "$DET/overlays" "$OUT/detector_overlays"
cp -a "$REC/overlays" "$OUT/overlays"
ARCHIVE=/workspace/new_product_scanning_results_${RUN_ID}.zip
cd "$OUT"
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
zip -qr "$ARCHIVE" .
unzip -tqq "$ARCHIVE"
sha256sum "$ARCHIVE"
ls -lh "$ARCHIVE"
printf 'RESULTS_DIR=%s\nARCHIVE=%s\n' "$OUT" "$ARCHIVE"
