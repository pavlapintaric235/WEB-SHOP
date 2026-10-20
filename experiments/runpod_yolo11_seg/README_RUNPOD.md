# InventoryLens Instance-Segmentation Experiment

This is an isolated, generic `product` instance-segmentation experiment. It contains the reviewed YOLO polygon labels unchanged; it does not contain SKU classes or use the October 17 diagnostic photos or `IMG_4699` for training/validation.

## Dataset

- 26 reviewed photographs and 163 visible-product polygons.
- Train: 21 images / 130 instances across 17 arrangement groups.
- Validation: 5 images / 33 instances across 5 arrangement groups.
- Class `0`: `product`.
- Original pixel resolution is retained; YOLO segmentation coordinates remain normalized polygon coordinates.
- `dataset_manifest.csv` stores exact source filenames, groups, dimensions and SHA-256 values.
- The 5 October 17 recognition diagnostics, all other existing held-out scenes and `IMG_4699` are not included in `dataset/`.

The validation set is very small. This run is an engineering experiment, not evidence of generalization. Collect a fresh scene-grouped evaluation set after the model and pipeline are frozen.

## RunPod setup

Use a RunPod image that already supplies CUDA-enabled PyTorch. Upload `inventorylens_yolo11_seg_training.zip`, then in JupyterLab terminal:

```bash
mkdir -p /workspace/inventorylens-seg
unzip -q inventorylens_yolo11_seg_training.zip -d /workspace/inventorylens-seg
cd /workspace/inventorylens-seg/inventorylens_yolo11_seg
python -m pip install -r requirements-runpod.txt
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Download and confirm the correct pretrained *segmentation* checkpoint (not the old detection-only retail checkpoint):

```bash
python -c "from ultralytics import YOLO; m=YOLO('yolo11n-seg.pt'); print('task:', m.task)"
```

Run the CUDA/data preflight and one-epoch smoke test:

```bash
python train_yolo11_seg.py --data dataset/data.yaml --weights yolo11n-seg.pt --mode smoke --device 0 --imgsz 960 --batch 4
```

If the smoke test completes and the GPU memory/runtime are acceptable, run the capped fine-tuning experiment. The script re-runs the smoke test first, then trains up to 40 epochs with patience 8:

```bash
python train_yolo11_seg.py --data dataset/data.yaml --weights yolo11n-seg.pt --mode train --epochs 40 --patience 8 --device 0 --imgsz 960 --batch 4 --workers 2
```

If the GPU runs out of memory, repeat with `--batch 2`; do not lower image resolution before inspecting small-product recall. The script refuses to overwrite existing run names. Change `--project` for a fresh attempt.

## Validation

The archive intentionally contains no protected final-test photographs. After training is finished, prepare a separately reviewed dataset YAML with test: images/test and corresponding labels/test/ polygons, then evaluate the untouched images without training or tuning on them:

    python evaluate_segmentation.py --data /workspace/heldout/data.yaml --split test --baseline yolo11n-seg.pt --fine-tuned runs/inventorylens_yolo11n_seg/fine_tune/weights/best.pt --output runs/final_test_comparison --device 0 --imgsz 960

Compare pretrained and fine-tuned segmentation on the same five grouped validation images. This is validation, not a final test:

```bash
python evaluate_segmentation.py --data dataset/data.yaml --baseline yolo11n-seg.pt --fine-tuned runs/inventorylens_yolo11n_seg/fine_tune/weights/best.pt --output runs/validation_comparison --device 0 --imgsz 960
```

The report includes Ultralytics mask mAP, one-to-one mask IoU@0.50 precision/recall, matched-mask IoU, false positives, misses, per-image counts and overlays. Keep the protected diagnostic images and `IMG_4699` out of this workspace until training is finished. Their evaluation must be a separate, later step; the diagnostic scenes are development evidence and are not an untouched final test.

Before terminating a non-persistent pod, download the entire `runs/inventorylens_yolo11n_seg/` directory (especially `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`, validation JSON and plots) to durable storage. Verify the downloaded checkpoint's SHA-256. Then stop and destroy the pod in RunPod to stop compute charges; stopping alone may leave storage charges.

## Local experimental demo

From the project root, use the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run experiments/hybrid_pipeline/segmentation_demo.py --server.port 8504
```

The demo defaults to the existing SKU-110K YOLO box detector plus MobileSAM box-prompt refinement. It offers MobileNetV2, SigLIP2 and equal-score fusion; it shows masks and top-five SKU proposals and supports employee removal, SKU correction and manually added rectangle regions. Exported counts are review-only and never written to SQLite.

## Licensing and scope

Ultralytics states its YOLO code, models, training pipelines and fine-tuned weights are AGPL-3.0 by default. A proprietary product must either meet AGPL obligations or obtain a suitable commercial license; review the current terms at https://www.ultralytics.com/license before deployment. The existing SKU-110K ONNX detector and MobileSAM checkpoint have separate provenance/licensing questions and are not redistributed in this training archive. The SigLIP2 model card lists Apache-2.0, but that does not clear its upstream training data or every dependency. MobileNetV2 is an experimental feature extractor only; no SKU classifier is trained.

This package makes no production changes and does not start training by itself.
