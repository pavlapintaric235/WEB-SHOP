# October 17 Diagnostic Benchmark

This is a development-only benchmark for the five runs between 01:40 and 01:47 Zagreb time. The photos must remain outside all training splits. The 01:42 and 01:44 views show the same cupboard arrangement and belong to one scene group.

## Provisional Visible Counts

These are a separate visual recount, not approved ground truth. Annotate visible pixels only; wholly hidden stock is not a visible instance.

| Run | Recorded expected | Provisional visible | Items not visually verified |
|---|---:|---:|---|
| 01:40 | 10 | 10 | None identified |
| 01:42 | 20 | 17 | 2 Energia, 1 Jana |
| 01:44 | 20 | 17 | 2 Energia, 1 Jana |
| 01:45 | 16 | 13 | 1 Eintopf, 1 Energia, 1 Freshona |
| 01:47 | 9 | 9 | None identified |

The 01:42 and 01:44 expected SKU dictionaries are exactly identical, while their image hashes differ. The photographs appear to show the same arrangement with different framing, so matching counts are plausible; intent is not recorded. Do not count a fully occluded extra as detected from these images. The 01:45 frame visibly supports one Eintopf, three Energia and two Freshona, not the recorded two, four and three. The extras could be occluded, so the table remains provisional until human review.

## CVAT Review Package

Open `data/recognition_checks/oct17_diagnostic_benchmark_20261017/cvat_review/README.md`. The folder contains copies of all five originals, an annotation-only COCO 1.0 ZIP with 30 unverified MobileSAM suggestions, numbered overlays of all 121 retained proposals, and proposal metadata. The preferred import has one `product` class and each proposed instance's SKU in `attributes.sku`; a legacy alternate ZIP uses SKU-specific categories. The suggestions are not ground truth. Correct/add polygons in a separate CVAT task; do not import them into the training dataset.

In CVAT, create a single `product` label and a non-mutable **Select** attribute named `sku`, with all registered SKU codes and CVAT's special `__undefined__` option. CVAT displays that option as blank; for this benchmark it means a real, visible product that is not registered in the catalogue. Keep its polygon as positive segmentation ground truth, and select it only after confirming the item is unregistered. For every other object, select the exact registered SKU. CVAT stores custom values in COCO annotation `attributes`; the evaluator accepts the literal `attributes.sku: "__undefined__"` and rejects a missing or empty SKU attribute. After export, spot-check both a registered item and an undefined item in `instances_default.json`. See the [CVAT COCO format documentation](https://docs.cvat.ai/docs/manual/advanced/formats/format-coco/) and [task label attributes guide](https://docs.cvat.ai/docs/workspace/tasks-page/).

The current visual recount is 66 target instances across the five photo views. It is not a final label count. The historic expected total is 75 and the historic system counted 30, but subtracting those values is not a validated miss count until the object-level labels and 01:45 expectations are reviewed.

The 30 COCO suggestion polygons were rasterized back at the model's working resolution and compared with their source MobileSAM masks: mean IoU **0.9957**, minimum **0.9644**. This checks only polygon-conversion fidelity; it does not validate whether a proposed mask follows a real product boundary.

## Recognition Evaluation

After exporting the reviewed CVAT task to COCO 1.0 with the five original images, run from the repository root:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_oct17_benchmark.py `
  --coco path\to\reviewed_task_export.zip `
  --images data\recognition_checks\oct17_diagnostic_benchmark_20261017\cvat_review\images
```

The evaluator accepts either the whole CVAT ZIP (it reads the COCO manifest without extracting) or an `instances_default.json` file.

This runs:

- **A: MobileSAM + current matcher.** Fresh MobileSAM masks on the same photographs, then current recognition/filtering with each run's saved similarity and margin settings.
- **C: reviewed-mask oracle + current matcher.** Each human polygon is used as a crop; detector size rejection and geometric proposal pruning are bypassed in this isolated process. ResNet18/SIFT, product references, and per-run match thresholds remain the current ones. This estimates SKU matching independent of proposal quality.
- **B: YOLO + current matcher.** Omitted until trained weights are supplied. Later, add `--yolo-weights path\to\best.pt --device 0` (or `--device cpu`) to run all arms. YOLO is expected to be a single `product` segmentation class; SKU recognition remains in the unchanged matcher.

The evaluator requires a complete five-image COCO mapping, a SKU identity on every annotation, an unchanged catalogue/reference set, and existing reference embedding caches. It accepts either COCO's per-instance `attributes.sku` or SKU-specific category names. The `__undefined__` identity remains a positive product instance for detection metrics: an unmatched object is a segmentation/detection miss; on a matched object, a matcher abstention is a correct unknown outcome and a registered-SKU assignment is a wrong identity. Registered-SKU identity accuracy is reported separately. Missing/empty SKU attributes and RLE masks are rejected rather than silently interpreted. It writes a new `evaluation/report.json` and refuses to overwrite that directory.

## Metrics and Timing

Detection matches are one-to-one, class-agnostic instance matches at mask IoU >= 0.50. Reported detection metrics are precision, recall, absolute count error and mean matched-mask IoU. SKU metrics are evaluated on matched instances, with correct, incorrect and unknown assignments broken down by true SKU; unregistered-product abstentions and false catalogue assignments are reported separately. The oracle arm reports SKU metrics only; its perfect input masks are not a detector result.

Detector time is measured separately from matcher/filter/prune time. The process also instruments ResNet18 encoding, SIFT verification and geometry pruning without editing production modules. Total recognition time includes the current recognition call and rendering. Historical A timings can differ because A is rerun. Record the device and GPU/CPU for every run: a GPU YOLO latency must not be presented as directly comparable to CPU MobileSAM latency.

## What a Detector Can Fix

A stronger instance detector can reduce missed, fragmented or merged object proposals and background regions. It cannot resolve a product crop that the ResNet18/SIFT matcher assigns to the wrong SKU or rejects as unknown. The oracle-crop arm isolates those matcher limits. Only after all three arms are measured should batching/cache/profile optimization be considered. Current ResNet18 crop encoding is already batched in groups of 16; reference embeddings are cached, and SIFT reference features are cached by the verifier. The saved history's `matching_seconds` combines neural encoding, similarity, SIFT verification and geometry pruning, so earlier stage bottlenecks inside that 15.35-second average cannot be attributed precisely. This evaluator adds separate timings for those operations.

These five photographs remain development diagnostics, not final test evidence. A fresh untouched evaluation batch is still required after the pipeline is frozen.
