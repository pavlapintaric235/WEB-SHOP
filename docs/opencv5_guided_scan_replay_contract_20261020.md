# OpenCV 5 guided-scan replay contract

Date: 2026-10-20  
Status: historical local replay specification and read-only baseline; the later AWS capture pilot is documented separately.

This contract records the original local replay boundary. Web&Shop is the competition entry and InventoryLens is its guided-scan module. A later, separately approved [OpenCV 5 AWS capture pilot](opencv5_aws_capture_pilot_result_20261020.md) was deployed in `eu-north-1` for the same Web&Shop demonstration; the historical no-AWS statements below describe this offline replay only. The [submission architecture](webshop_opencv_2026_submission_20261020.md) distinguishes that live capture component from archived DINO/SigLIP2 predictions and planned full integration.

## Purpose and boundary

Demonstrate a small guided capture loop in which OpenCV 5 image evidence changes the next action: request a clearer photo or proceed to review NVIDIA Retail DINO boxes and SigLIP2 SKU suggestions. Every proposed count remains pending until a person confirms or corrects it. This reuses InventoryLens and its existing benchmark artifacts; it does not create another project.

This replay reads the original JPEGs, `photos.csv`, `scenes.csv`, and the recovered DINO output archive. It must not write into `data/`, the SQLite inventory, references, Owner_test, employee_test, customer_test, product-registration experiments, or the benchmark ZIPs. The source scenes are marked `holdout_unmodified`; the replay is diagnostic and cannot be used to tune detector or recognition thresholds or claim generalization. The eight photos below are seven physical arrangements, because both B10 views show the same arrangement.

The current Streamlit production path remains MobileSAM/ResNet18. DINO/SigLIP2 are experimental adapters. The owner registration flow remains the source of business SKU identities. Employee and customer demonstrations may share an inference contract, but neither may update inventory during this experiment.

## Frozen inputs

Source root: `data/recognition_checks/final_comparative_gpu_benchmark_20261018/inventorylens_final_comparative_gpu_benchmark/data/source/`  
Manifest: `photos.csv` (full SHA-256, dimensions, scene group and role); grouping: `scenes.csv`  
Saved inference archive: `C:/Users/U938/Downloads/inventorylens_nvidia_dino25_e2e_results_20261019.zip` (previously audited SHA-256 `20b98a4b3a8bfe5ce0d1fb84e2dc78353fd695563b8e13cce4ae6d3a1b88bff3`)

All eight source hashes were checked against `photos.csv` on 2026-10-20 and matched. Relative paths below are under `images/` in the source root.

| Scene | Workflow | Original photo path | SHA-256 | Why included |
|---|---|---|---|---|
| E01 | employee | `scenes/E01/IMG_4703 (1).jpeg` | `b0b797f715fa4fc42310f818e4e301fd7a39797012c1be3d67a37fef0fce7939` | Repeated Energia cans; merged and missed instances. |
| E02 | employee | `scenes/E02/IMG_4707.jpeg` | `a032caa5beefd214dd10efafd9106ba7e726cf333051074341d3b7ffc45c22d9` | Human-reviewed acceptable masks; dark products are a false-alarm check for exposure rules. |
| E06 | employee | `scenes/E06/IMG_4717.jpeg` | `d5338ce665094ba4abdec781a0dbcca7f6c2f8cf5d012e87c06eae1682cf8850` | User-confirmed six wrapped items; DINO produced zero boxes. |
| B01 | customer basket | `scenes/B01/IMG_4754.jpeg` | `9f8f47fcaf840bd63cb8a56a5e00be5b0d384c96246f32545d05d1679b232cb0` | Simple four-product scene and source for in-memory capture-quality controls. |
| B07 | customer basket | `scenes/B07/IMG_4764.jpeg` | `16f3fbf55a44395c79dc530b6f2ee149f3ef24e574385f9e2fdf06e2b1cbda57` | One box spans several SKUs; needs split/review rather than one accepted count. |
| B08 | customer basket | `scenes/B08/IMG_4768.jpeg` | `4a54d28bf74b50957385a32e52a3a974b95014c1801b9ffd7005d98bc01691a9` | One of two Oreos was missed; test manual addition. |
| B10 | customer basket | `scenes/B10/IMG_4770.jpeg` | `78201d59483f66687e3c566a67649b2b16247f5e5c447ef447302613aeaf7379` | First view of one user-confirmed 10-unit arrangement. |
| B10 | customer basket | `scenes/B10/IMG_4771.jpeg` | `a98ae0dde7df2202b9d2a9e0d3843c6de2e89e2d3912f759376b47d391fece38` | Second view of that *same* arrangement; must not be summed with the first. |

Never use a reference photograph as a replay scene. Do not treat B07 as an unchanged extra view of B06. For B10, retain two image records with one `arrangement_id=B10`; show counts per view and ask a reviewer to reconcile identities across views before producing one arrangement count.

## OpenCV 5 capture gate

The 2026-10-20 local environment reports `cv2.__version__ == 5.0.0` and `opencv-python == 5.0.0.93`. Record both values in a future replay trace. The main `requirements.txt` currently allows `opencv-python>=4.7`, so that file alone does not prove the submitted runtime is OpenCV 5. A later isolated demo package should pin a tested OpenCV 5 wheel without changing the production dependency file.

For each source photo, decode with OpenCV, retain its original width/height, resize to at most 1024 pixels on the long side using `INTER_AREA`, then convert BGR to grayscale. Record:

- `lap_var_1024`: variance of `cv2.Laplacian(gray, cv2.CV_64F)` as a focus cue;
- `dark_pct`: percentage of resized gray pixels with value <= 20;
- `bright_pct`: percentage with value >= 235;
- `decode_ok`, original dimensions, source SHA-256, and OpenCV version.

Provisional *demo branch*, frozen for any later implementation of this contract: unreadable image or short side below 720 pixels -> `retake`; otherwise `dark_pct > 35` -> `retake: more light`; otherwise `bright_pct > 35` -> `retake: reduce glare`; otherwise `lap_var_1024 < 100` -> `retake: hold camera steady`; otherwise `proceed`. These illustrative values were chosen after a read-only look at the eight images and are **not calibrated capture-quality thresholds**. Uniform backgrounds and dark packaging can fool whole-image exposure measures, so a human can override every prompt. Do not use these eight held-out images to adjust the numbers further or report sensitivity/specificity; obtain independent capture-quality labels and calibration photos first.

Read-only baseline measured with OpenCV 5 on the frozen original images:

| Photo | Laplacian variance | Dark % | Bright % | Provisional branch |
|---|---:|---:|---:|---|
| E01 `IMG_4703 (1)` | 430.6 | 1.4 | 0.4 | proceed |
| E02 `IMG_4707` | 223.8 | 18.5 | 0.0 | proceed |
| E06 `IMG_4717` | 290.4 | 0.0 | 0.1 | proceed |
| B01 `IMG_4754` | 774.2 | 0.8 | 0.1 | proceed |
| B07 `IMG_4764` | 519.2 | 4.5 | 1.4 | proceed |
| B08 `IMG_4768` | 387.1 | 0.8 | 0.6 | proceed |
| B10 `IMG_4770` | 1036.4 | 1.3 | 0.3 | proceed |
| B10 `IMG_4771` | 897.1 | 2.5 | 0.2 | proceed |

Functional negative controls are derived from B01 **in memory only**, with no saved image or new dataset row. `cv2.GaussianBlur(original, (0, 0), sigmaX=6)` gave `lap_var_1024=2.24` and should request a steadier photo. `cv2.convertScaleAbs(original, alpha=0.07, beta=0)` gave `dark_pct=100.0` and should request more light. These controls test branching, not real-world quality accuracy. Do not attach the original DINO predictions to a transformed control image.

## Saved DINO and SigLIP2 handoff

Use the archive members `detector_retry2/detections.csv` and `recognition_retry1/per_detection_predictions.csv` without rerunning models. Join on `(scene_id, image, instance_id)` after verifying each selected image and SHA-256. Filter the prediction table to `crop_variant == "boxes"` and `recognizer == "siglip2"`; do not mix mask crops or other recognizers into the demo count. Retain `detector_confidence`, `box_xyxy`, `top5`, `top1_similarity`, `margin`, `status`, `reason`, model/threshold provenance, and a stable replay ID (`photo_sha256:instance_id`). Treat an absent detection as zero proposals, not a missing image.

For these eight images the saved archive contains **29 DINO boxes** and **29 SigLIP2 box-crop prediction rows**: E01 6, E02 6, E06 0, B01 4, B07 1, B08 3, B10 view 1 4, B10 view 2 5. The 29 join keys were checked on 2026-10-20: none were duplicate or unmatched. SigLIP2 marked 18 of those rows `recognized` and 11 `unknown`. Those are *acceptance counts only*, not correct-SKU counts.

The review interaction must let a person confirm/change a SKU, retain `unknown`, remove a false box, add a missed physical product, split a merged box, and choose one reconciled count for B10's two views. No prediction commits automatically. E06 must display its zero-box result and allow all six units to be added; B07 must flag its mixed-SKU box for splitting; B08 must permit the missed Oreo to be added. The gallery omitted ILX-008/Croco because of a variant mismatch, so Croco cannot be scored as closed-gallery recognition or silently mapped to the registered pizza variant.

## Trace and evaluation contract

One trace record per photo should contain: source path/hash, scene/arrangement ID, workflow, OpenCV/version and quality metrics, branch and human override, DINO detector/runtime identifiers, every box ID/coordinates, SigLIP2 candidate list and acceptance status, correction actions, and final pending/confirmed state. A separate arrangement record may link B10's two view IDs; it must not add their raw box counts. Export a review-only JSON/CSV under a **new experiment output directory** if implementation is later authorized. It must not call the production inventory service or write to SQLite.

Functional acceptance checks for a later implementation:

1. All eight file hashes match before processing; any mismatch aborts that image.
2. Every original yields one quality trace; B01's in-memory blur and dark controls select their respective retake prompts; the eight originals select `proceed` under the provisional rule.
3. The archive join yields exactly 29 boxes and 29 SigLIP2 box rows, with zero boxes for E06. A duplicate or unmatched `(scene_id, image, instance_id)` fails the replay rather than being silently discarded.
4. A `retake` control never consumes predictions from the unmodified original. Every accepted prediction remains editable and pending until explicit confirmation.
5. The B10 views remain one 10-unit arrangement, never 20; the E06 human-confirmed total remains six. Existing human review describes B07's mixed box and B08's miss; a reviewer must resolve them rather than count predictions as ground truth.
6. No source file, reference, test history, registration, or inventory record changes. No network or AWS call occurs in the local replay.

Report separately: capture-gate branches and overrides; raw proposals per image; accepted/unknown suggestions; manual additions, splits, removals and corrections; confirmed count versus verified total only where available; correction time; and per-stage wall time. Do **not** report detector precision/recall, SKU accuracy, false-acceptance rate, or capture-gate accuracy from these eight scenes: reconciled instance labels, a complete verified gallery, and independent capture-quality labels are missing. Keep any later final evaluation on untouched arrangement groups with thresholds locked beforehand.

## Historical AWS mapping, later approved and deployed

The smallest cloud demonstration is a private S3 Standard input area and an on-demand Python Lambda Event Function in the confirmed `eu-north-1` selected Region. An upload triggers the OpenCV 5 capture gate; it writes a quality/decision JSON under a distinct output prefix and exposes the instruction to the review UI. The DINO TensorRT engine stays on a separately chosen compatible runner; the current Lambda CPU architectures cannot run that GPU engine unchanged. SigLIP2 CPU packaging and latency need measurement before any proposal to place it in Lambda. Use S3 prefix filtering to avoid recursive invocation, encryption in transit and at rest, narrow service permissions, a short retention policy for consented demo photos, and explicit cost limits. S3 requests/storage, Lambda requests/duration, and logs can incur charges. No AWS plan, service access, deployment cost or production migration was verified or authorized by this local contract.

## Evidence used

- `complete_final_gpu_benchmark_audit_20261019.md` and `human_review_nvidia_dino25_20261019.md` in the parent benchmark folder;
- `data/source/photos.csv`, `scenes.csv`, `catalog_crosswalk.csv`, `review_required.csv`, and `reference_audit.json` in the packaged benchmark;
- the saved DINO archive's CSV headers, 29 selected detection rows, and 29 selected SigLIP2 box prediction rows;
- read-only local OpenCV 5 measurements on the eight original JPEGs and two in-memory B01 controls.
