# CPU detector fine-tune report — 2026-10-20

## Run

- Base: existing Retail YOLO11n detection checkpoint, `sku110k-yolo11-n640.pt`.
- Goal: adapt its generic `object` class to the generic `product` class. It does not learn the ten new SKU identities.
- Data: 25 train images / 144 instances and 5 validation images / 33 instances. The reserved 25 employee and 5 customer scenes were not used.
- Runtime: local Intel Core i5-8250U, 8 logical CPUs; PyTorch `2.14.1+cpu`; Ultralytics `8.4.177`; CUDA unavailable.
- Settings: CPU, 640 px, batch 2, up to 20 epochs, patience 5, seed `20261020`, workers 0. Early stopping ended at epoch 7; the best weights are from epoch 2. The recorded training epoch time sums to about 196 seconds; validation and setup add several minutes overall.
- No RunPod GPU was used for training. The previously named Pod was read back as `EXITED`, with its 50 GB persistent `/workspace` volume attached. No AWS or production changes were made.

## Same-split validation

Both checkpoints were validated on the exact same 5 photos and 33 boxes at 640 px on CPU.

| Checkpoint | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Pretrained retail checkpoint | 0.259 | 0.303 | 0.161 | 0.040 |
| Fine-tuned best checkpoint (epoch 2) | 0.729 | 0.489 | 0.531 | 0.211 |

This is a favorable signal on this tiny validation split, not a reliable claim of real-world improvement. There are only five validation scenes, and the measures vary sharply by epoch: the metrics peaked at epoch 2, dropped in epochs 4–6, then partially recovered at epoch 7. Early stopping retained the best checkpoint. More reviewed, diverse scenes are needed to decide whether this training recipe generalizes.

## Artifacts

- Best checkpoint: `runs_cpu/retail_yolo11n_product_finetune_20261020-2/weights/best.pt` (local artifact: `runs_cpu/retail_yolo11n_product_finetune_20261020-2/weights/best.pt`)
- Last checkpoint: `runs_cpu/retail_yolo11n_product_finetune_20261020-2/weights/last.pt`
- Epoch metrics: `runs_cpu/retail_yolo11n_product_finetune_20261020-2/results.csv`
- Same-split comparison: `runtime/validation_comparison.json`
- Best checkpoint SHA-256: `F1E4CA73018ACB75D575FDF842FA7D09B27B79F4A5BE4B2B079757718DE48E7E`

## Held-out newest scenes

The 30 scene rows are listed in `03_newest_10_products_owner_employee_customer/manifest_this_part.csv`, but their original image bytes are not available in the current workspace. A recursive search found no `IMG_4857.jpeg` or `IMG_4897.jpeg`; that handoff folder contains the manifest, registration material and scene-mask folder metadata only. No third-set inference or visual review has been run. The next step is to add the original employee and customer scene photos (or the complete part-3 ZIP), verify their SHA-256 values against the manifest, then produce paired pretrained-versus-fine-tuned predictions for human review. Do not compute newest-scene accuracy until identities and physical-instance annotations are verified.

## Integrity and scope

The original dataset-preparation script generated an isolated YOLO dataset under `training_data/`. Ultralytics interpreted its relative `path: .` against the repository root, so training used a separate corrected YAML at `runtime/dataset_cpu.yaml`; source labels and original photographs were not rewritten. No DINO or SigLIP2 component was trained or changed. This experiment is not deployed in WEB&SHOP and does not adjust live stock.
