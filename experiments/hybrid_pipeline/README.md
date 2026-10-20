# Web&Shop Hybrid Pipeline (Experimental)

This is an employee-facing experiment isolated from the live recognizer, SQLite, registered references, approved segmentation set, and historical results. It reads the existing catalogue and references only. It does not add business-facing Web&Shop features or persist SKU registrations.

## What Runs Here

- **Detector:** a checksum-verified YOLO11n ONNX checkpoint pretrained on SKU-110K (single generic `object` class), for local research benchmarking. It provides boxes, not SKU labels or masks. Its use is restricted by the source-data terms, so it is not cleared for commercial deployment.
- **Fallback detector:** the existing local `yoloe-26s-seg.pt`, with one visual reference prompt per registered SKU. This is not SKU-110K and must not be presented as a generic retail detector. It is included as an offline fallback.
- **Few-shot matcher:** ImageNet-pretrained TorchVision MobileNetV2 pooled features compared to normalized per-SKU reference centroids; supports explicit unknown/ambiguous abstention. It does not train a SKU classifier.
- **Optional feature-only baselines:** public `facebook/dinov2-small` (CLS-token embedding) and `google/siglip2-base-patch16-224` image embeddings. Neither trains a classifier. They load local Hugging Face cache files only; no download or fallback happens implicitly.
- **DINOv3 adapter:** supported when Meta's gated local repository and selected checkpoint are supplied through `DINOV3_REPO` and `DINOV3_WEIGHTS`. No DINOv3 weights are bundled or silently substituted.
- **Retail detector adapter:** uses `weights/sku110k-yolo11-n640.onnx`; set `INVENTORYLENS_SKU110K_WEIGHTS` to override it. A COCO checkpoint is not equivalent.

The upstream [sku-vision-pipeline](https://github.com/Sanket758/sku-vision-pipeline) describes SKU110K/YOLO retail proposals followed by normalized DINOv3 and MobileNetV2 embeddings, per-SKU exemplar/centroid matching, and human review. I inspected its README plus the curated-pipeline detection, embedding, and matching modules; its README documents the orchestration as detect → embed → match → review. This prototype borrows the retrieval shape, but uses SigLIP2 instead of gated DINOv3, equal-weight branch cosine fusion, existing InventoryLens references, and a read-only registry; it is not a line-for-line reproduction and does not reuse the upstream fusion weights. The upstream README reports thesis metrics but links to a placeholder dataset, so those results are not independently reproducible from that repository alone. Its MIT code license does not establish the license of every checkpoint or dataset it references.

## Install and Weights (PowerShell)

Run these from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r experiments\hybrid_pipeline\requirements.txt
New-Item -ItemType Directory -Force experiments\hybrid_pipeline\weights | Out-Null
```

MobileNetV2 uses the official TorchVision `IMAGENET1K_V2` checkpoint (about 14 MB). Download it explicitly into the experiment, not the global Torch cache:

```powershell
$weights = (Resolve-Path experiments\hybrid_pipeline\weights).Path + '\mobilenet_v2-7ebf99e0.pth'
Invoke-WebRequest 'https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth' -OutFile $weights
Get-FileHash $weights -Algorithm SHA256
```

The benchmark's YOLO11n ONNX file and official published checksum manifest are already in `experiments/hybrid_pipeline/weights/`. The ONNX runtime is an optional dependency listed above. The checkpoint is surfaced in the demo as **research-only**.

TorchVision applies its documented V2 resize, center crop, and ImageNet normalization. The demo does not fetch weights by itself. Keep the existing `yoloe-26s-seg.pt` in the project root for the fallback.

## Run the Local Instance-Mask Review Demo

```powershell
$env:YOLO_CONFIG_DIR = (Resolve-Path experiments\hybrid_pipeline\.cache\ultralytics -ErrorAction SilentlyContinue).Path
if (-not $env:YOLO_CONFIG_DIR) { New-Item -ItemType Directory -Force experiments\hybrid_pipeline\.cache\ultralytics | Out-Null; $env:YOLO_CONFIG_DIR = (Resolve-Path experiments\hybrid_pipeline\.cache\ultralytics).Path }
.\.venv\Scripts\python.exe -m streamlit run experiments\hybrid_pipeline\segmentation_demo.py --server.port 8504
```

The page accepts an uploaded scene or one of the five saved October 17 photos. It runs either retail YOLO boxes refined by MobileSAM or an explicitly supplied YOLO11-seg checkpoint, then shows colored instance masks, detector confidence, proposed SKU, top-five candidates, and stage timings. Employees can remove false masks, correct SKUs, or add missed products as manual rectangular regions, then export confirmed counts to JSON/CSV. Corrections exist only in Streamlit session state and are never written to inventory.

The app defaults to SigLIP2 + MobileNetV2 features with per-encoder-set abstention cutoffs calibrated on the three calibration scenes in the polygon-masked crop experiment. These are experimental thresholds, not production settings; the demo shows them and includes them in exports. The demo never trains a SKU classifier.

## Segmentation Benchmarks

The completed five-photo mask comparison is at data/recognition_checks/oct17_diagnostic_benchmark_20261017/mask_backend_comparison_20261017_224806/. It compares automatic MobileSAM proposals with retail YOLO boxes refined by MobileSAM against the same ambiguity-filtered human masks. The output contains per-image overlays and report.json; these five photos are development diagnostics, not final evaluation data.

The repeatable mask comparison is implemented in experiments.hybrid_pipeline.evaluate_mask_backends. Crop representation comparisons across MobileNetV2, SigLIP2, and equal-score fusion are implemented in experiments.hybrid_pipeline.crop_ablation. Both write to a new output directory and keep the predefined calibration/held-out split; the two held-out photographs are views of one arrangement, so this sample cannot support a generalization claim.

## DINOv2 / SigLIP2 Recognition Comparison

The end-to-end mask-to-SKU comparison is at `data/recognition_checks/oct17_diagnostic_benchmark_20261017/hybrid_mask_recognition_20261017_234206/`. It evaluates automatic MobileSAM proposals, SKU110K YOLO boxes refined by MobileSAM, and human masks with MobileNetV2, SigLIP2, and their fusion. On these five development photos, YOLO+MobileSAM reduced the proposal count from 238 to 34 and improved mask precision from 0.151 to 0.676, but recall was only 0.469 (23/49 objects matched at IoU 0.50). With human masks, SigLIP2 recognized 32/44 registered objects correctly, abstained on 12, and falsely assigned 1/5 undefined objects. These small, related scenes are diagnostic only; they do not establish generalization. See `report.md`, `report.json`, and the per-object CSVs for detailed results.

The repeatable comparison uses the same 49 ambiguity-filtered Inventory6 human masks for every encoder (44 registered-SKU instances and 5 explicitly undefined instances). Crops, product-reference crops, and per-SKU best-reference aggregation are aligned across feature encoders. It reports top-1/top-5 ranking before abstention, accepted correct/wrong/unknown outcomes, and top-5 recall after abstention (only accepted objects contribute; both accepted-only and all-object denominators are reported), unknown-product rejection, per-SKU rows, reference filenames, crop geometry, SIFT evidence for the unchanged ResNet18 matcher, and tight versus 10%-neutral-padding crops. The fixed calibration scenes are 01:40, 01:45 and 01:47; 01:42 and 01:44 remain together as the held-out related-arrangement group. New encoder cutoffs are selected from calibration only. ResNet18/SIFT keeps the per-run historical thresholds.

Install the optional Transformers adapter only after confirming free disk space. DINOv2 Small's safetensors file is about 88 MB; reserve at least 1 GB for model, Python dependencies, cache and temporary overhead. SigLIP2 Base's checkpoint is about 1.5 GB; reserve at least 2.5 GB. Both pinned checkpoints and the optional dependencies are now cached under `.venv\huggingface_cache`; the last verified free space was about 24 GB. Recheck before downloading or running another model.

```powershell
.\.venv\Scripts\python.exe -m pip install -r experiments\hybrid_pipeline\requirements-encoders.txt
$env:HF_HOME = (Resolve-Path .venv).Path + '\huggingface_cache'
New-Item -ItemType Directory -Force $env:HF_HOME | Out-Null
.\.venv\Scripts\hf.exe download facebook/dinov2-small config.json preprocessor_config.json model.safetensors --revision ed25f3a31f01632728cabb09d1542f84ab7b0056
# Optional, much larger CPU baseline:
.\.venv\Scripts\hf.exe download google/siglip2-base-patch16-224 config.json preprocessor_config.json model.safetensors --revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
```

Run only models already present locally (the default is offline and does not fetch weights):

```powershell
.\.venv\Scripts\python.exe -m experiments.hybrid_pipeline.fewshot_compare --models dinov2 mobilenetv2 siglip2 dinov2+mobilenetv2 --output experiments\hybrid_pipeline\runs\fewshot_review_01
```

Choose a new output directory each time. To permit downloading by this evaluator, pass `--allow-download`; it performs a disk-space preflight first. `--no-padding-ablation` skips the secondary crop comparison. The MobileNetV2 and original ResNet18/SIFT arms can run without Transformers. SigLIP2 is CPU-feasible here but slow; it embeds references and queries one at a time to keep memory use bounded.

The DINOv2 Small Hugging Face model card identifies it as Apache-2.0 and a raw feature encoder without a task-specific head. SigLIP2 Base is also listed Apache-2.0 by its model card. Checkpoint license metadata does not replace review of upstream training-data terms or other deployment obligations. SigLIP2 is included only as a CPU-feasibility-dependent independent baseline, not a promised improvement.

For repeatable local inference on image files (writes overlays, `results.json`, and `counts.csv` to a new run folder):

```powershell
.\.venv\Scripts\python.exe -m experiments.hybrid_pipeline.run_photos C:\path\to\photo1.jpeg C:\path\to\photo2.jpeg
```

The CLI defaults use the calibration-only cutoffs from the five development photos. Arbitrary photos have no ground truth, so these outputs are exploratory and must not be reported as accuracy measurements.

## Reproduce the Five-Photo Benchmark

The benchmark reads `C:\Users\U938\Downloads\inventory6.zip`, saved original images and historical results read-only. It writes a unique folder beneath `experiments/hybrid_pipeline/runs/` and refuses to reuse an existing timestamp folder.

```powershell
.\.venv\Scripts\python.exe -m experiments.hybrid_pipeline.run_benchmark
```

The calibration scenes are 01:40, 01:45, 01:47. The 01:42 and 01:44 images stay together as held-out views of one arrangement. Known Inventory6 count/boundary discrepancies are flagged and ignored in scored comparisons; all raw objects remain visible in the CSV. This is a tiny development split, not a valid estimate of generalization. Historical MobileSAM/ResNet18/SIFT values are copied into a new report only as context; they were previously calculated with all provisional Inventory6 annotations and are not directly comparable to the filtered experimental scores.

The primary benchmark is **SKU-110K YOLO11n + MobileNetV2**. YOLOE remains a clearly named fallback option in the demo. DINOv3 arms remain unavailable until the gated model assets are supplied. The human-mask MobileNetV2 diagnostic is additional; the earlier Inventory6 report contains the existing ResNet18/SIFT human-mask arm.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s experiments\hybrid_pipeline\tests -v
```

## License and Deployment Notes

- The reference repository says its code is MIT, but that does not settle checkpoint or training-data rights. Its referenced YOLOv5 checkpoint is not present here.
- The downloaded community SKU-110K YOLO11n checkpoint lists its license as `other`; its model card explicitly says it does not claim a more permissive license than SKU-110K's upstream terms. Treat it as academic/non-commercial research only; do not ship it in Web&Shop or use it for commercial service without written rights clearance.
- Ultralytics states its YOLO code, models, training pipelines and fine-tuned models are AGPL-3.0 by default; a proprietary product must either meet AGPL obligations or obtain a suitable commercial license. Review the current Ultralytics licensing terms at https://www.ultralytics.com/license before deployment; this experiment is not legal advice.
- The SigLIP2 Base model card lists Apache-2.0, and the DINOv2 Small model card lists Apache-2.0. That does not independently clear their training data or every dependency; review upstream terms before commercial use. See https://huggingface.co/google/siglip2-base-patch16-224.
- DINOv3 code and weights require accepting Meta's DINOv3 license and obtaining gated access. Review its redistribution and use terms before shipping. Do not equate it to Apache or MIT.
- The reference repository's README labels DINOv3 as CC-BY-NC; Meta's official repository instead publishes a separate DINOv3 License. The upstream README's shorthand is not authoritative for those weights.
- YOLOX is an Apache-2.0 codebase candidate, but every pretrained retail checkpoint and its training data still need independent license verification. A permissible code license does not automatically clear weights.
- A more permissive detector alternative for commercial research would be an Apache-2.0 detector implementation/checkpoint, for example YOLOX, with a compatible, explicitly licensed retail checkpoint. Its accuracy and runtime may differ from Ultralytics and it needs an adapter; this prototype does not replace YOLOE or claim that YOLOX weights are already cleared.
- MobileNetV2's TorchVision implementation is BSD-3-Clause; verify the exact pretrained-weight and source-dataset terms before commercial distribution.

## Non-Goals

No model training, cloud/GPU rental, writes to production database/reference images, changes to recognition thresholds, or promotion into the production pipeline occur here.
