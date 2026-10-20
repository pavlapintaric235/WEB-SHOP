# Benchmark Audit and Provisional MVP Architecture Decision

Date: 2026-10-19

## Decision

**Provisional MVP:** NVIDIA Retail DINO detector boxes + SigLIP2 reference-image retrieval + mandatory human confirmation/editing. MobileSAM may be enabled as an optional visual overlay/refinement, but must not define product counts or gate scanning. Keep this experimental until recognition is scored on a verified, gallery-covered, single-product set and the detector comparison is rerun under matched settings.

**Polygons are not required for the initial MVP.** Rectangular boxes support selection, crop-based SKU suggestions, count review, adding missed items, and splitting merged detections. Show masks only as an optional visual aid. Promote polygons to a core requirement only if the product requirement depends on visible-surface/partial-occlusion measurement or if instance boundaries must be represented precisely for downstream geometry.

This is an architecture recommendation, not authorization to change production code. No production changes, training, annotations, or new GPU rental were made.

## Detector audit

The local YOLO reports and NVIDIA holdout run refer to the same five CVAT validation arrangements and 33 instances. Both use a 0.50 IoU matching threshold. They are **not an identical-settings head-to-head**:

| Detector | TP / FP / FN | Precision | Recall | F1 | Count MAE | Evaluation details |
|---|---:|---:|---:|---:|---:|---|
| NVIDIA Retail DINO | 27 / 0 / 6 | 1.000 | 0.818 | 0.900 | 1.2 | 5 images, 33 GT; confidence 0.50; box IoU 0.50; input 960x540; TensorRT. Engine build/warmup 227.87 s excluded; summed detector inference 0.217 s. |
| COCO YOLO11n-Seg | 13 / 5 / 20 | 0.722 | 0.394 | 0.510 | 3.0 | Same 5 images/33 GT; confidence 0.25; `imgsz=960`; IoU 0.50. Primary metric was mask IoU. The report's box-IoU aggregate also gives 13/5/20 and the same P/R/F1. Summed inference 3.739 s. |
| Retail SKU110K YOLO11n detector | 1 / 29 / 32 | 0.033 | 0.030 | 0.032 | 3.8 | Same 5 images/33 GT; confidence 0.25; box IoU 0.50; summed inference 3.763 s. This extremely poor result needs output/coordinate/class-agnostic evaluator audit before it is treated as a stable model ranking. |

On the available numbers, DINO has the strongest detection result. But its higher confidence threshold, different network input/preprocessing, and TensorRT implementation mean the advantage is descriptive, not a controlled causal comparison. YOLO's segmentation primary score is mask-IoU; DINO has only boxes. For the fairest existing comparison, compare DINO box metrics with YOLO-Seg's separate box metrics, while still noting threshold/input differences. The validation set is only five arrangements, so uncertainty is large and not captured by conventional asymptotic confidence intervals here.

The human-reviewed 25-scene DINO overlays independently show missed objects, multi-product boxes, and mask spill/incompleteness. Those observations are consistent with instance separation remaining a major practical issue; they are qualitative and are not added to the five-image metric table.

## Recognition audit

The six recognition decision modes in the experiment are:

1. MobileNetV2
2. DINOv2-small
3. DINOv2-small + MobileNetV2
4. SigLIP2 Base
5. SigLIP2 + MobileNetV2
6. NVIDIA FAN-Base-Hybrid

The NVIDIA+DINO 25-scene run applied these modes to the same detector-derived box crops and MobileSAM-mask crops, so candidate rankings can be compared for the *same crop input*. It used fixed, previously sourced thresholds for the first five modes and did not tune them on these scenes. FAN had no calibrated acceptance threshold, so it reported rankings but abstained; do not score those abstentions as recognition failures or successes.

This does **not** satisfy the requested accuracy audit on verified identical single-product crops:

- The full retry5 comparative archive is not locally available, so its `recognition_comparison.csv`, `pipeline_comparison.csv`, and exact six-mode tables could not be recovered or rechecked.
- The available 25-scene prediction report explicitly says there are no object-level labels and does not score SKU correctness, wrong assignments, or unknown rejection.
- The gallery had 55 reference photographs across 11 mapped experiment SKUs: ILX-001..007 and ILX-009..012. **ILX-008 / Croco was omitted.** Croco predictions therefore are not eligible for closed-gallery accuracy, even though the product is registered in the broader InventoryLens catalog.
- Several crops are not single products: reviewed overlays include same-SKU pairs merged together and a cross-SKU box spanning Fentimans, curry, and 7Days. These must be excluded from ordinary single-instance recognition scoring or explicitly labeled as detector boundary errors.
- Reference coverage for a mapped SKU does not prove every catalog registration or every useful view is covered. Gallery identity/viewpoint audit is still needed.
- Prior human-crop tallies were provisional and conditional on manually mapped detections; they are not a substitute for the missing full result tables and a locked, instance-level label file.

**Recognition conclusion:** SigLIP2 is a reasonable first candidate for the MVP suggestion layer, not yet a validated winner. The benchmark does not establish that it beats all five other modes on a sufficiently large, verified, gallery-covered, identical crop set. Keep the recognition choice behind an experimental adapter and human confirmation until the archive and labels are audited.

The NVIDIA FAN graph did execute partly on CUDA and partly on CPU (62.1% of profiled node events were CUDA, 37.9% CPU in the available audit). It was not all-GPU execution, and its no-threshold abstention policy means there is no accepted-SKU accuracy to compare.

## Artifact recovery and RunPod boundary

I searched the workspace and the local Downloads, Desktop, Documents, and OneDrive directories for `inventorylens_gpu_benchmark_retry5_20261019.zip`; it was not found. The local NVIDIA DINO E2E archive is present and SHA-256 verified as `20b98a4b3a8bfe5ce0d1fb84e2dc78353fd695563b8e13cce4ae6d3a1b88bff3`.

The live RunPod list read shows the existing benchmark Pod `1ohwvb6g1jia9f` is `EXITED`, with a persistent 50 GB `/workspace` volume. The RunPod MCP connection can inspect Pod metadata but cannot run shell commands or transfer files. I did not start the Pod because that resumes GPU charges and no rental was authorized. To complete archive recovery without creating a Pod, upload the already-downloaded retry5 ZIP from your laptop into the workspace. Alternatively, you may explicitly authorize resuming this existing Pod for the short retrieval; that is a billable action, not performed here.

## Minimum integration plan (after audit)

1. **Capture:** guided photo capture with framing/blur feedback and a clear instruction to include each visible product. Customer and employee workflows use the same image inference API, with role-specific inventory destination and confirmation copy.
2. **Detect:** return stable per-image detection IDs, bounding boxes, detector confidence, and crop previews. Never merge counts across different camera views automatically.
3. **Suggest:** run SigLIP2 retrieval against a verified registry gallery and show top candidates plus similarity/margin. Abstain to `unknown` when evidence is weak; unknown remains editable and is never forced to a registered SKU.
4. **Review:** user confirms or corrects SKU and quantity, can add missed products, remove a false detection, or split a box containing multiple products. Keep a pending count until confirmation.
5. **Optional mask view:** provide MobileSAM polygons for visual inspection only. The user can hide this layer. A missing mask must not erase a detected box or change its proposed count.
6. **Commit/export:** write only confirmed SKU/count rows to the chosen inventory session; retain image, box ID, selected SKU, user correction, and model/version provenance for audit.
7. **Acceptance gate:** before production integration, recover retry5 outputs; produce a reviewed single-product crop set with complete gallery coverage; report per-method correct/wrong/unknown, top-1/top-5, false acceptance on unknown products, and latency. Evaluate detection separately on the same images with matched input/confidence/matching settings and untouched grouped arrangements.

## Required next action to finish the audit

Provide `inventorylens_gpu_benchmark_retry5_20261019.zip` locally. Then inspect all three comparison CSVs and `report.json`, verify exact run/image/threshold fields, join recognition predictions by stable crop ID, and score only manually verified single-product crops whose actual SKU is in the gallery. Keep Croco and other uncovered SKUs unscored until their references are included in a future isolated evaluation package; do not rerun inference or tune against the final held-out set merely to fill this gap.

## Recovery correction (2026-10-19)

The requested ZIP has since been recovered locally and verified against SHA-256 `309160573265d74f07786ece29e488fa64480f68243a35a68cc04d549ba1cafd`. The earlier archive-missing statements and “provide the ZIP” next action above are superseded. Its complete reports and all three comparison CSVs were inspected. They confirm that recognition results are acceptance coverage, not accuracy, and that no verified single-product labels exist to score all six recognizers. See [complete_final_gpu_benchmark_audit_20261019.md](complete_final_gpu_benchmark_audit_20261019.md) for the corrected, full audit, tables, and final provisional MVP recommendation.

The separate DINO CVAT result and YOLO CVAT reports use the same five images and 33 instances but different confidence/input settings. DINO leads numerically; the comparison is not identical-settings. The 25-scene DINO inference and human overlay review remain qualitative, not a scored end-to-end recognition benchmark. No Pod was resumed and no production code was changed.

