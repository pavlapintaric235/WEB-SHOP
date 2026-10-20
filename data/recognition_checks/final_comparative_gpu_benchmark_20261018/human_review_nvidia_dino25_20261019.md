# Human Review: NVIDIA DINO + MobileSAM (2026-10-19)

## Scope and evidence

This addendum records the user's visual review of the NVIDIA Retail DINO detector and MobileSAM mask overlays in `inventorylens_nvidia_dino25_e2e_results_20261019.zip` (SHA-256: `20b98a4b3a8bfe5ce0d1fb84e2dc78353fd695563b8e13cce4ae6d3a1b88bff3`). The archive contains 25 detector overlays and 25 recognition/mask overlays. Human comments below are review evidence, not CVAT annotations and not a replacement for the original benchmark reports.

The packaged `scene_counts_DRAFT.csv` labels itself provisional and explicitly says not to score it as ground truth. The 25 images include repeated views/arrangements; do not sum them as distinct inventory. No annotations, model outputs, thresholds, or historical results were changed by this review.

## Mask review

Status means the user's qualitative review of the displayed overlay. “Good” does not imply pixel-level mask annotation or quantitative IoU validation.

| Image | Human review |
|---|---|
| IMG_4703 (1).jpeg | Most visible products are masked acceptably. Of the three Energia cans, the middle is masked; the frontmost and farthest cans are not correctly/individually masked. The detector grouped two cans in one box, and the third was missed. |
| IMG_4707.jpeg | Good. |
| IMG_4710.jpeg | Fentimans mask partly spills onto a neighboring unmasked item. Jana mask does not cover the full bottle; cap is partly uncovered. |
| IMG_4711.jpeg | First Dukat mask spills onto the second Dukat bottle, which has no separate mask. |
| IMG_4713.jpeg | Fripsy mask follows its package accurately. Croco packages are covered together by one mask (merged instances). Additional Fripsy packages were missed by detection. |
| IMG_4716.jpeg | Good. |
| IMG_4717.jpeg | No detector boxes were produced; mask quality is not applicable to missed instances. |
| IMG_4754.jpeg | Good. |
| IMG_4755.jpeg | Good. |
| IMG_4757.jpeg | Good for detected objects; detection missed a 7Days product per the user's count review. |
| IMG_4758.jpeg | Good. |
| IMG_4759.jpeg | Good. |
| IMG_4760.jpeg | Good. |
| IMG_4761.jpeg | Good. |
| IMG_4763.jpeg | Good for the visible masks except the Oreo has a detection box but no usable mask. |
| IMG_4764.jpeg | Good for the reviewed visible masks. One detector box spans Fentimans, curry, and 7Days; this is a cross-SKU merged detection and cannot be treated as one correctly segmented instance. |
| IMG_4768.jpeg | Good for detected objects; one of two Oreos was missed by detection. |
| IMG_4769.jpeg | Good for the detected Croco; other visible products were missed by detection. |
| IMG_4770.jpeg | Good for the detections the user identified; several visible products are absent from detections. |
| IMG_4771.jpeg | Only the bottom portion of a Fentimans bottle is masked. |
| IMG_4774.jpeg | Same partial Fentimans mask issue: only the bottom portion is masked. |
| IMG_4775.jpeg | Good. |
| IMG_4779.jpeg | First Dukat mask spills slightly onto the second Dukat bottle, which is unmasked. Both Croco packages are masked together. |
| IMG_4781.jpeg | Dukat pair and Croco pair are each masked together instead of separately. First Energia is masked well; two Energia cans behind it are unmasked. |
| IMG_4782.jpeg | First Dukat mask spills partly onto the second Dukat bottle, which is unmasked. |

## Detection and instance-boundary findings

The human count and overlay review identifies recurring **instance detection/counting** failures, independently of SKU classification:

- `IMG_4703 (1).jpeg`: three Energia cans are present; a box spans two cans and the third is not detected. Only the middle can has a good mask.
- `IMG_4710.jpeg`: user reports four missed products, including two Energia and two Fentimans.
- `IMG_4711.jpeg`: user reports only 2 Dukat, 1 Energia, no Fentimans, and 1 Jana detected despite a larger visible count.
- `IMG_4713.jpeg`: two Croco packages share one detection/mask; additional Fripsy packages are missed.
- `IMG_4716.jpeg`: user reports only one Fripsy and one Stobi detected; the other visible products are missed.
- `IMG_4717.jpeg`: no boxes were produced for the visible products.
- `IMG_4757.jpeg`: 7Days missed, according to the user's image-by-image count review.
- `IMG_4763.jpeg`: Oreo box exists but its mask is unusable/missing.
- `IMG_4764.jpeg`: one box spans three different SKUs, so this is not a valid single-product instance.
- `IMG_4768.jpeg`: one of two Oreos is missed.
- `IMG_4769.jpeg`: only Croco was detected; other visible products were missed.
- `IMG_4770.jpeg`: user identifies only a subset of visible instances as boxed.
- `IMG_4771.jpeg` and `IMG_4774.jpeg`: Fentimans masks are partial.
- `IMG_4779.jpeg`, `IMG_4781.jpeg`, and `IMG_4782.jpeg`: same-SKU instances are merged or mask spill leaves a neighboring instance unmasked.

The primary system limitation indicated by this review is product-instance detection and separation, with further MobileSAM mask-boundary failures. Recognition scores on detector crops cannot recover products that were never independently detected, and merged crops may not be valid recognition inputs.

## Human count notes and unresolved reconciliation

The user supplied image-by-image count/identity observations during the review. These are not yet normalized to a structured annotation file. Some earlier quick summaries conflict with later per-detection mappings (notably counts for `IMG_4782.jpeg` and whether a Croco or other box was present on `IMG_4770.jpeg`). Prefer the detailed latest user-provided ID mapping only after reconciling it against `detections.csv` and the original overlay. Do not invent missing identities or force an ambiguous mixed crop to one SKU.

The user also stated that Mivolis packages/graphics in `IMG_4707.jpeg` are a distraction, not a product requiring recognition. This is a task-scope decision, not a general rule to omit real registered products.

Before computing corrected precision/recall, create reviewed object-level labels containing one row per visible physical product, a stable image ID, instance polygon/box, SKU or `__undefined__`, and explicit ambiguity/exclusion reason. Preserve every distinct visible instance; do not collapse same-SKU pairs. Keep fully hidden products out. Keep repeated scene views separate for per-photo scores and group them by arrangement when reporting uncertainty.

## Recognition interpretation

The earlier 25-photo run is a prediction/inference report, not an independently labeled accuracy benchmark. The reference gallery was reported as 55 photos across 11 mapped SKUs; this is not proof that every registered SKU/reference photo in the full catalogue was included. The benchmark's source package had a limited mapped gallery, and reference coverage needs explicit audit before claiming catalogue-wide few-shot performance.

Do not count an unknown or wrong-SKU prediction as a model recognition error until the detection crop is confirmed to contain exactly one product and the actual SKU is confirmed to be represented by the reference gallery. Do not score mixed-SKU detections as ordinary single-instance recognition. The human mask review is qualitative; only overlays explicitly reviewed above have human mask-quality evidence.

## Recommended next steps

1. Reconcile the user's per-image and per-detection notes with `detector_retry2/detections.csv` and `recognition_retry1/per_detection_predictions.csv`; flag contradictions instead of silently choosing.
2. Produce an object-level reviewed CSV/JSON for the eligible instances, excluding unresolved mixed boundaries and labeling excluded scope items explicitly.
3. Recompute detector metrics from the reviewed instance labels. Report detection recall/precision and count error separately from mask quality.
4. Recompute recognition metrics only on confirmed single-product detections whose SKU has verified reference-gallery coverage. Report accepted-correct, accepted-wrong, and abstained separately; do not tune thresholds on this review set.
5. If these labels will be used for model selection or training, keep this manual review set distinct from any later held-out evaluation.

