# Complete InventoryLens GPU Benchmark Audit

Date: 2026-10-19  
Scope: artifact recovery, report audit, detector and recognition evidence, and provisional MVP decision. No model training, CVAT annotation, production change, or Pod start was performed.

## Decision

For a guided, human-confirmed MVP, the best-supported experimental path is **NVIDIA Retail DINO boxes + SigLIP2 few-shot SKU suggestions + explicit human review**. Keep MobileSAM as an optional visualization/refinement layer. Do not auto-commit counts, and do not describe this combination as production-validated: DINO has promising results on only five held-out arrangements, SigLIP's reported coverage is not correctness, and human review found missed/merged instances and incomplete masks.

**Rectangular boxes are enough for the initial workflow.** They let a reviewer select a detected item, inspect its crop, confirm a SKU, split a multi-product box, delete a false detection, or add a missed product. Polygons can remain an optional overlay. Make polygons mandatory only if the workflow needs precise visible boundaries or quantitative mask/area evaluation.

## Recovered artifacts

The full retry-5 archive is present locally and its SHA-256 matches the previously recorded digest:

- [Full GPU benchmark archive](../runpod_b300_20261019/inventorylens_gpu_benchmark_retry5_20261019.zip), 3,943,354,882 bytes; SHA-256 `309160573265d74f07786ece29e488fa64480f68243a35a68cc04d549ba1cafd`.
- It contains `report.json`, `report.md`, `detector_comparison.csv`, `recognition_comparison.csv`, `pipeline_comparison.csv`, `scene_metrics.csv`, `per_image_results.csv`, `per_detection_predictions.csv`, overlays, runtime profiles, model failures, and experiment artifacts.
- The separate CVAT holdout package is [here](../runpod_b300_20261019/inventorylens_cvat_holdout_eval_20261019.zip).
- The separate 25-scene DINO end-to-end output archive was downloaded and SHA-256 verified as `20b98a4b3a8bfe5ce0d1fb84e2dc78353fd695563b8e13cce4ae6d3a1b88bff3` (local Downloads folder).

The retry-5 report says `completed_with_model_failures`. It used 25 scene photos across 19 scene groups, 16 supplied reference photos, and 55 gallery reference images for 11 mapped experiment IDs (ILX-001..007 and ILX-009..012). These are not all registered catalogue SKUs; ILX-008/Croco was excluded as an unmapped variant mismatch. The report says no per-instance scene boxes or masks were available. It therefore does not score object-level detection precision/recall, mask IoU, per-object SKU top-1/top-5 accuracy, or unknown rejection.

The existing Pod was EXITED when checked. No reason to resume it: the archive is local. No new GPU rental was started.

## Detector results

### Separate five-image CVAT holdout

Both reports use the same five image arrangements and 33 ground-truth instances. Both use one-to-one matching at IoU 0.50, but they are not an identical-settings comparison:

| Detector | TP / FP / FN | Precision | Recall | F1 | Count MAE | Evaluation settings |
|---|---:|---:|---:|---:|---:|---|
| NVIDIA Retail DINO | 27 / 0 / 6 | 1.000 | 0.818 | 0.900 | 1.2 | confidence 0.50; box IoU 0.50; fixed input 960x540; TensorRT |
| COCO YOLO11n-Seg | 13 / 5 / 20 | 0.722 | 0.394 | 0.510 | 3.0 | confidence 0.25; Ultralytics `imgsz=960`; IoU 0.50; box aggregate shown for comparison, segmentation report also evaluates masks |
| Retail SKU110K YOLO11n | 1 / 29 / 32 | 0.033 | 0.030 | 0.032 | 3.8 | confidence 0.25; `imgsz=960`; box IoU 0.50 |

DINO's summed detector inference time was 0.217 s after a 227.87 s engine build/warm-up, excluded from image totals. YOLO11n-Seg's summed inference was 3.739 s; SKU110K YOLO's was 3.763 s. These timing figures come from the respective evaluation reports and are not a controlled throughput comparison across identical runtimes/input shapes.

**Finding:** DINO is the clear numerical leader on this small holdout at the settings used. It still missed six instances and undercounted by six in aggregate. The confidence thresholds and input/preprocessing differ, so this is descriptive evidence, not a controlled claim that the model alone caused the entire difference. The YOLO-Seg primary mask score must not be compared to DINO's box score; use YOLO's separate box metrics for the table above.

SKU110K's near-zero result is genuinely poor as reported. The available T05 overlay and aggregate results show substantial localization/matching failures. The evaluator's polygon-to-box conversion and class-agnostic matching are caveats worth auditing, but the current evidence does not establish a global coordinate bug that would explain away the result. Do not call the SKU110K result a successful retail detector.

### 25-photo comparative run and separate DINO run

In retry-5, the detectors that actually ran were SKU110K YOLO11n + MobileSAM (89 proposals; mean 3.56/photo) and COCO YOLO11n-Seg (92 proposals; mean 3.68/photo). NVIDIA Retail DINO **failed in that run** because ONNX Runtime could not load its `MultiscaleDeformableAttnPlugin_TRT` custom operation. No 25-photo DINO-versus-YOLO accuracy comparison exists in retry-5.

A later, separate TensorRT/domain-remapped DINO run did execute on all 25 scene photos and emitted 102 boxes. MobileSAM and the recognition encoders were run on its box/mask crops. This run had no object-level labels, so 102 boxes is output volume, not recall or accuracy. User overlay review found missed items, merged boxes, cross-SKU boxes, and partial/spilling masks; see [human review](human_review_nvidia_dino25_20261019.md).

## Recognition audit

The six modes were MobileNetV2, DINOv2-small, DINOv2+MobileNetV2, SigLIP2 Base, SigLIP2+MobileNetV2, and NVIDIA FAN-Base-Hybrid. Within each detector arm, every recognizer received the same proposal-crop set and gallery. This is a same-crop **prediction/acceptance-coverage** comparison, not a verified single-product accuracy comparison: proposal crops were not labeled, can contain multiple products, and scene ground truth was absent.

| Detector proposal source | Recognizer | Accepted / proposals | Acceptance coverage | Accuracy scored? |
|---|---|---:|---:|---|
| SKU110K + MobileSAM (89) | MobileNetV2 | 10 / 89 | 11.24% | No |
|  | DINOv2-small | 2 / 89 | 2.25% | No |
|  | DINOv2 + MobileNetV2 | 1 / 89 | 1.12% | No |
|  | SigLIP2 | 27 / 89 | 30.34% | No |
|  | SigLIP2 + MobileNetV2 | 2 / 89 | 2.25% | No |
|  | NVIDIA FAN-Base-Hybrid | 0 / 89 | 0% | No; threshold not calibrated |
| COCO YOLO11n-Seg (92) | MobileNetV2 | 49 / 92 | 53.26% | No |
|  | DINOv2-small | 43 / 92 | 46.74% | No |
|  | DINOv2 + MobileNetV2 | 41 / 92 | 44.57% | No |
|  | SigLIP2 | 53 / 92 | 57.61% | No |
|  | SigLIP2 + MobileNetV2 | 50 / 92 | 54.35% | No |
|  | NVIDIA FAN-Base-Hybrid | 0 / 92 | 0% | No; threshold not calibrated |

SigLIP2 has the highest accepted coverage in both detector arms under the pre-existing thresholds. Coverage means only “passed this encoder's fixed acceptance rule”; it does **not** mean the SKU was correct. The different encoders' similarity values are in different spaces and must not be ranked by raw score. The first five modes used prior Inventory6 calibration thresholds unchanged; FAN had no calibrated acceptance threshold, so its all-abstain policy is neither a measured failure nor a measured success.

The run does not provide a fair accuracy leaderboard on verified, identical, single-product crops with adequate SKU gallery coverage. Human review of the separate DINO run is useful qualitative evidence but is not a normalized per-instance label set for scoring all six methods. Products lacking verified gallery coverage and mixed/multi-product boxes must remain unscored.

## Full-pipeline timing

Retry-5's mean estimated per-photo pipeline wall times, by selected recognizer, were:

| Detector arm | SigLIP2 | DINOv2 | MobileNetV2 | FAN | All loaded recognizers together |
|---|---:|---:|---:|---:|---:|
| SKU110K + MobileSAM | 0.350 s | 0.357 s | 2.111 s | 0.540 s | 2.528 s measured |
| COCO YOLO11n-Seg | 0.322 s | 0.350 s | 2.129 s | 0.554 s | 2.609 s measured |

Fused DINOv2+MobileNet and SigLIP2+MobileNet estimated means were about 2.16-2.21 s/photo. These are experiment-specific wall-time estimates from the report, not user-facing latency guarantees. The FAN ONNX model ran on mixed providers: 59,173 of 95,222 profiled node events (62.14%) were on CUDA and 36,049 (37.86%) on CPU. Therefore FAN was not fully GPU-executed.

## Gallery and unknown-product limits

The 55-image, 11-ID gallery was the experiment gallery, not the whole registered database. Exact catalog/reference correspondence was not established for every product. Croco was intentionally not scored because the available reference mapping was a different variant. A SKU absent from this gallery cannot be used to judge the embedder's ability to recognize it; a genuinely unregistered product also needs explicit known-unknown labels to evaluate rejection.

## Guided scanning integration plan

1. Capture one well-framed image or one shelf section at a time; prompt for a second angle only as a separate view. Never add counts across views unless instances are explicitly linked.
2. Run the experimental DINO detector and display stable detection IDs, boxes, confidence, crop previews, and timing. Preserve missed/uncertain items as user-addable rather than implying the detector saw everything.
3. Run SigLIP2 retrieval against an audited per-business SKU gallery. Show top candidates and scores, but use an `unknown`/needs-review state when the existing acceptance rule fails.
4. Require confirmation before counts are committed. Allow changing SKU, removing false detections, adding missed items, and splitting a box that contains two or more products.
5. Keep MobileSAM hidden by default or available as an optional overlay. A missing or bad mask must not remove a box or alter a count.
6. Store confirmed image-level product instances and corrections with provenance. Export counts only from confirmed instances; keep photos from different angles separate without identity linking.
7. Run a small acceptance evaluation before production: identical detector preprocessing/settings, reviewed single-product crops, per-SKU gallery audit, and correct/wrong/unknown plus false-acceptance metrics on untouched arrangements.

Customer-facing scanning should favor a short guided capture and review flow; employee shelf scanning should guide users through bounded shelf sections and make omissions/splits easy to correct. Both should use the same explicit confirmation principle. The current benchmark does not establish separate customer-versus-employee accuracy.

## Next improvements

- Complete and audit the reference gallery for every SKU the MVP promises to support; record exact variant and package size.
- Create a compact, reviewed set of distinct product instances with SKU labels, plus explicitly unregistered examples; preserve held-out arrangements.
- Normalize the DINO and YOLO holdout input size, preprocessing, confidence policy, and box-IoU evaluation before claiming a final detector ranking. Keep a separate mask-IoU metric for segmentation.
- Use human corrections to measure accepted-correct, accepted-wrong, abstained, missed, merged, count error, and correction time. Do not lower thresholds merely to increase coverage.
- Defer any new paid GPU run until the evaluation inputs and required outcomes are fixed; the local archive is already recovered, and the previous Pod is exited.

