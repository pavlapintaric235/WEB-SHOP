# Web&Shop RunPod YOLO11n Detection Experiment

This isolated package fine-tunes one generic class, `product`. SKU identity remains a separate few-shot recognition task. It does not change the production pipeline, catalog, database, references, or historical results.

## Contents and data boundary

- `images/train` and `labels/train`: 21 images, 130 product instances.
- `images/validation` and `labels/validation`: 5 images, 33 instances.
- All 26 images are from the approved seed plus the owner-approved 23-photo CVAT batch. The scene-grouped split follows `dataset_manifest.csv`; no arrangement group crosses train and validation.
- `final_test/images/IMG_4699.jpeg` is test-only and excluded from `data.yaml`. Its confirmed total is six products (Jana Antiox x2, Fentimans Ginger Ale x2, Energia Power x1, Dukat milk x1). The confirmation export contains model proposals, not verified boxes, so the supplied evaluator can compare counts and visualize predictions but cannot report IoU, precision, or recall for this image.
- The October 17 diagnostic photos, including the existing 01:42/01:44 held-out arrangement, are not copied into train or validation.
- `data.yaml` has one class, `product`. The YOLO detection labels are `class cx cy width height`, normalized to image dimensions. They were derived from each approved polygon's coordinate extrema; segmentation polygon text was not interpreted as a box.

## Data limitations and review notes

This is a small experiment, not a generalization-quality dataset: 21 training images, 5 validation images, and 163 instances total. The 5 validation images provide weak evidence and are not a substitute for a fresh untouched evaluation batch.

The source CVAT geometry audit found zero invalid/self-intersecting polygons, zero out-of-bounds masks, and zero duplicate-mask candidates. It flagged four tiny masks (three small candy masks in `IMG_4571.jpeg`, one small wrapped-item mask in `IMG_4588.jpeg`) and one small partial overlap in `IMG_4571.jpeg`; the owner approved preserving the individual candies and the reviewed overlap. Polygon geometry cannot prove every photograph is completely annotated. The earlier visual review found no obvious missing fully visible item, but crowded scenes remain inherently uncertain.

The output contains axis-aligned boxes computed tightly around visible-region polygons. This loses contour detail by design, as required for YOLO detection. It does not change source polygons.

## RunPod setup

Upload and unzip `inventorylens_runpod_training.zip` in JupyterLab. Select a CUDA-enabled PyTorch image, open a terminal in the extracted package folder, then verify the GPU before installing anything:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "CUDA build", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
PY
```

Install Ultralytics into the active environment, then repeat the CUDA check. Do not replace the RunPod image's CUDA-enabled PyTorch build with a CPU-only build:

```bash
python -m pip install -r requirements-runpod.txt
python train.py --data ./data.yaml --weights ./weights/sku110k-yolo11-n640.pt --mode smoke
```

The smoke command checks CUDA, the actual checkpoint, all image/label pairs, class ids, normalized boxes, and scene grouping, then runs a one-epoch CUDA smoke fit. It saves output under `runs/webshop_yolo11n/` and does not alter the packaged data or checkpoint.

After the smoke run completes successfully, launch the capped experiment:

```bash
python train.py --data ./data.yaml --weights ./weights/sku110k-yolo11-n640.pt --mode train --epochs 40 --patience 8 --imgsz 640 --batch 8
```

`--mode train` repeats the one-epoch smoke check first, then starts a fresh run from the supplied retail checkpoint (not from the smoke checkpoint). Early stopping uses validation patience 8; 40 is a hard maximum. If CUDA memory is insufficient, lower `--batch` (e.g. 4) and retry. Keep the `runs/webshop_yolo11n/` output until the best checkpoint and logs have been downloaded.

Download `runs/webshop_yolo11n/fine_tune/weights/best.pt` (preferred) and the matching run's results/plots from JupyterLab before stopping the Pod. Keep the baseline weights so detector comparisons use the same baseline.

## Compare the untouched test image

The evaluator never trains or changes either checkpoint. It creates side-by-side prediction overlays and JSON output in a new directory:

```bash
python evaluate_detectors.py \
  --baseline ./weights/sku110k-yolo11-n640.pt \
  --fine-tuned ./runs/webshop_yolo11n/fine_tune/weights/best.pt \
  --source ./final_test/images \
  --expected-counts ./final_test/ground_truth_counts.json \
  --output ./evaluation/img4699_first_run
```

Use the exact output path matching the actual `project`/`name` reported by training (default is `runs/webshop_yolo11n/fine_tune/weights/best.pt`). The default inference settings are the same for both models. The final image has count truth but no verified box labels; do not interpret count error as object-level precision/recall. For IoU metrics, add independently reviewed YOLO detection labels for a separate evaluation set using `--labels-dir`, and report that set's provenance.

## Checkpoint and license

`weights/sku110k-yolo11-n640.pt` is the retail-specific PyTorch YOLO11n checkpoint. It is checksum-matched to the public model card; the project also has its ONNX inference export. The model card marks the weights `other` and ties downstream use to the SKU-110K source-data terms. Treat them as research-only unless you obtain rights clearance. Ultralytics is AGPL-3.0 absent a separate commercial license; review obligations before any proprietary Web&Shop deployment. This bundle does not authorize commercial use.

No training is performed during package creation. Do not use the final-test image or October 17 held-out diagnostic images for training, validation, threshold selection, or model selection.
