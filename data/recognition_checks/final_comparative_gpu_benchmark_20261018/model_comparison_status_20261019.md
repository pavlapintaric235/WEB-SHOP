# InventoryLens Model Comparison Status (2026-10-19)

## Executive summary

The strongest numerical detector result available in the conversation is NVIDIA Retail DINO on the separate five-image CVAT holdout: 27/33 instances matched, precision 1.00, recall 0.818, F1 0.900, and mean absolute count error 1.2 products/image. It missed six objects and undercounted by six overall. This is promising for that tiny holdout, not evidence of broad generalization.

The 25-photo NVIDIA DINO + MobileSAM run is end-to-end in the sense that detector boxes were passed to MobileSAM and then to the recognition encoders. It is not an accuracy benchmark: those 25 photographs did not have reconciled object-level ground truth when the run was scored. The user's later review found material missed, merged, and partial masks, so those outputs cannot be described as accurate full-scene SKU counting.

Across the human-reviewed scenes, the recurring bottleneck is product-instance detection and separation. Mask failures compound it. Recognition is only meaningful on a correctly detected, single-product crop and when that product has verified references in the gallery.

## Detector comparison

| Detector path | Evidence available | Interpretation |
|---|---|---|
| NVIDIA Retail DINO (TensorRT) | Separate CVAT holdout: 5 images, 33 GT instances, 27 predictions, TP 27, FP 0, FN 6; precision 1.000, recall 0.818, F1 0.900; MAE 1.2; total predicted-minus-GT = -6. Engine build/warmup 227.87 s excluded; measured detector inference total 0.217 s. | Best exact held-out detector metrics currently in the record. Very small sample; high precision came with misses/undercount. This is box detection, not segmentation or SKU recognition. |
| Retail SKU110K YOLO11n + MobileSAM | Ran as a detector/refiner arm in the 25-photo comparative inference run. No usable object-level labels for those scenes were available in that run. | No valid accuracy comparison against DINO on the 25 scenes. Human overlay review found missed and merged products in the overall review, but does not yield a complete numeric score for this detector arm. |
| COCO-pretrained YOLO11n-Seg | Ran as a separate detector arm in the comparative inference run. No matched reviewed labels/results bundle is available locally. | No comparable numerical score available here. COCO segmentation is a generic baseline, not retail-specific. |
| NVIDIA Retail DINO + MobileSAM | 25-scene prediction run; 102 detector boxes across 25 images. No object-level SKU/count labels at run time. | End-to-end predictions and overlays exist, but 102 is output volume, not recall/accuracy. Human review identified single boxes spanning multiple products, missed products, and partial/spilling masks. |

The locally present YOLO11n-Seg report confirms the same five CVAT scene arrangements and 33 GT instances, but the runs were **not configured identically**. YOLO11n-Seg used confidence 0.25, IoU matching 0.50, and `imgsz=960`; its primary metric was mask IoU. Aggregated box matching from the same report is TP=13, FP=5, FN=20 (precision 0.722, recall 0.394, F1 0.510), MAE=3 and total count bias=-15. NVIDIA DINO used confidence 0.50 and box IoU matching at 0.50; its fixed input shape was 960x540. Thus both report box scores use the same five arrangements, instance count, and IoU threshold, but confidence thresholds and input/preprocessing differ. The primary YOLO segmentation score is mask-based, whereas DINO has no masks; compare the YOLO **box_metrics** row to DINO boxes, not YOLO's primary mask score. The results favor DINO descriptively, but are not a controlled head-to-head because threshold and input settings differ. No threshold tuning is proposed from this held-out set.

An earlier YOLO holdout attempt reported near-zero precision/recall, but it was followed by overlay review and a later result of 0.722/0.394/0.510. The near-zero attempt should be treated as a failed evaluation/configuration, not model evidence.

## Recognition model comparison

The comparative pipeline evaluated these crop-embedding arms against the packaged few-shot reference gallery:

- MobileNetV2
- DINOv2-small
- DINOv2-small + MobileNetV2
- SigLIP2 Base
- SigLIP2 + MobileNetV2
- NVIDIA FAN-Base-Hybrid retail recognition model

The gallery used in the 25-scene NVIDIA DINO end-to-end run was 55 reference photos across 11 mapped SKUs. This is not the entire registered catalogue. The run emitted rankings, similarities, abstentions, and timings, but without object-level labels those are not recognition accuracy.

The human review addendum contains provisional box-crop tallies only for reviewed, single-product detections whose SKU was considered gallery-covered. It suggests SigLIP2 had the best accepted coverage in that subset, with no observed wrong accepted assignments in the provisional tally; other encoders either accepted fewer or had some wrong accepted outputs. These counts are conditional on reviewed crops and user-supplied identity mappings, not a formal held-out test. Most SAM crops were not separately validated, so do not use their apparent recognition results as a fair mask-crop comparison. NVIDIA FAN abstained on the reported run because it had no calibrated acceptance threshold; its outputs must not be interpreted as zero recognition capability.

No trustworthy comparison of rejection of genuinely unregistered products is available: the reviewed set did not include a curated, labeled unknown-product cohort.

## Human review findings affecting interpretation

The user's review of NVIDIA DINO + MobileSAM overlays reported:

- Missed Energia and Fentimans instances in crowded/repeated arrangements.
- A DINO box covering two Energia cans as one detection, with another can not detected.
- Croco packages merged in one detection/mask in at least one image.
- A cross-SKU box spanning Fentimans, curry, and 7Days in one image.
- Partial masks, spill onto adjacent bottles, and pairs of same-SKU bottles masked together rather than separately.
- Many other listed masks judged good by the user.

The detailed per-image evidence is in [human_review_nvidia_dino25_20261019.md](human_review_nvidia_dino25_20261019.md). That review is qualitative and has not been converted into CVAT polygons or a complete per-instance metric dataset.

## Bottom line

1. **Detector:** NVIDIA Retail DINO is the current leader by the exact small held-out numbers available, but it still undercounts, and human review on the separate crowded scenes confirms instance separation failures.
2. **Segmentation:** MobileSAM can produce useful masks for some detections, but it cannot repair missed detector proposals. Human review found boundary spill, incomplete masks, and merged instances. A good mask on a subset is not complete scene segmentation.
3. **SKU recognition:** SigLIP2 appears most promising for accepted predictions on the provisionally reviewed, gallery-covered box crops. Do not claim a definitive winner until corrected instance labels and reference coverage are fixed and scores are recomputed on held-out arrangements.
4. **Most justified improvement:** Improve retail instance detection/separation and evaluate masks per object; then improve recognition with a complete, audited reference gallery and a properly calibrated unknown/rejection set.

## Evidence gaps before a final all-model leaderboard

The full comparative archive `inventorylens_gpu_benchmark_retry5_20261019.zip` was not found in the checked local workspace, Downloads, Desktop, Documents, or OneDrive locations. The RunPod Pod still exists but is EXITED and its `/workspace` volume is persistent; reading files there requires the user to resume the Pod (which incurs GPU billing) and use SSH/SFTP or transfer the archive. I did not resume it. Therefore the archive's `detector_comparison.csv`, `recognition_comparison.csv`, `pipeline_comparison.csv`, and full `report.json` remain unaudited. The NVIDIA 25-scene archive is present and SHA-256 verified, but its report is prediction-focused, not scored accuracy. The original RunPod outputs and this review have not been altered.

## MVP architecture recommendation (provisional)

- Use NVIDIA Retail DINO for candidate product boxes and SigLIP2 few-shot retrieval as the leading experimental pair, with confidence-based abstention and mandatory human confirmation before committing SKU counts. This is not yet approval to modify production: DINO's tiny holdout is promising, but its confidence/input settings were not matched to YOLO and the crowd-scene review shows missed/merged instances.
- Keep MobileSAM optional for visual refinement and QA display. It cannot create missing detections; the review found incomplete, spilling, and merged masks.
- For the first guided scanning MVP, boxes are sufficient for product selection, crops, SKU suggestions, and counting. Polygons are not a prerequisite unless the workflow requires visible-portion accounting, precise overlap separation, or auditable area/IoU measurement. Show boxes as the core interaction; offer masks as an optional overlay/refinement, not as the source of the count.
- Recognition leaderboard remains pending: the retry5 archive is required to recover the six recognition/pipeline result tables, and a fair accuracy comparison additionally needs confirmed single-product labels and verified gallery coverage. FAN's abstentions are not evidence of failure because no acceptance threshold was calibrated.

### Minimum integration sequence after the audit

1. Build an isolated adapter that accepts an image and returns per-box detection IDs, coordinates, detector score, crop, and runtime metadata.
2. Pass each crop through SigLIP2 retrieval against an audited SKU reference gallery; return top candidates, scores/margin, and `unknown` when acceptance criteria are not met. Keep FAN disabled for automatic decisions until calibrated on separate calibration data.
3. Present boxes and candidate SKU choices to the user. Require confirmation or correction of both SKU and count; support adding a missed item and splitting a merged box. Do not silently auto-increment inventory from uncertain predictions.
4. Export a reviewable inventory summary and retain per-image evidence. Add optional MobileSAM polygons for visualization/refinement, while keeping confirmed object instances and SKU/count corrections as the authoritative record.
5. Run a small offline acceptance test on held-out arrangements with the exact production-like preprocessing and separately report detection, recognition (correct/wrong/unknown), counting, correction rate, and latency. Only then consider a production integration proposal.

To produce a rigorous leaderboard, use the full comparative archive plus a reconciled object-level label file, verify identical five-image holdout IDs/configuration for detector arms, audit the exact reference-gallery SKU coverage, and report recognized-correct / recognized-wrong / unknown separately. Keep the 25-scene human review out of threshold tuning and reserve genuinely untouched arrangements for final evaluation.

## Recovery correction (2026-10-19)

The retry-5 archive **was recovered locally** at `data/recognition_checks/runpod_b300_20261019/inventorylens_gpu_benchmark_retry5_20261019.zip`; the earlier “not found” statements above are superseded. Its SHA-256 is `309160573265d74f07786ece29e488fa64480f68243a35a68cc04d549ba1cafd`. I inspected its `report.json`, `report.md`, detector, recognition and pipeline CSVs, model failures, and scene metrics. The exact audit and corrected architecture decision are in [complete_final_gpu_benchmark_audit_20261019.md](complete_final_gpu_benchmark_audit_20261019.md).

Correction to the earlier detector narrative: the later 0.722 precision / 0.394 recall / 0.510 F1 result is **COCO YOLO11n-Seg**, not SKU110K YOLO. SKU110K YOLO's same-five-image report is 1 TP / 29 FP / 32 FN (P=.033, R=.030, F1=.032). NVIDIA DINO's separate five-image result is 27/0/6 (P=1.000, R=.818, F1=.900). Same images/GT/IoU, but different confidence and input/preprocessing settings; this is not an identical-settings comparison.

The six recognizers were compared on the same unverified proposal crops within each detector arm. The recovered CSVs report acceptance coverage only, not accuracy. SigLIP2 had the highest acceptance coverage in both arms; this does not demonstrate that those accepted SKUs were correct. The 25-photo DINO+MobileSAM run is also prediction-only, with qualitative human review but no scored instance-level recognition set. FAN remains uncalibrated for automatic acceptance and its abstentions must not be scored as model errors.

