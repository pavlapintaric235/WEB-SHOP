# WEB&SHOP live inventory tracking — InventoryLens visual scanning module

**WEB&SHOP is the application and OpenCV AI Competition 2026 entry. InventoryLens is the product-registration, visual scanning, SKU recognition, and inventory-intelligence module being integrated into it.** This workspace contains the existing local prototype and its supporting datasets and benchmarks. The AWS guided-scan pilot is part of the same WEB&SHOP demonstration, not a second project.

## What the project should do

The intended workflow is **register a SKU → photograph products → locate and count each physical item → suggest a registered SKU → let a person correct the result → approve an inventory action → update the location's stock**. A photo supplies an observation and a proposed count. The user or a linked business event must specify whether approved units are added or subtracted; the image alone cannot determine that. Repeated views of the same arrangement must not double-count items.

- **Business owners** register distinct sellable SKUs with clear reference photos, review stock and scan history, and control who can approve changes.
- **Employees** photograph receiving or removal scenes, resolve missed or merged products, confirm SKUs and quantities, and approve an `ADD` or `SUBTRACT` movement for the chosen location.
- **Customers** use a guided photo scan to review products for a cart or purchase. A customer photo does not reduce stock; a finalized sale would be the separate authorized stock event.

OpenCV 5 currently provides live photo-quality guidance: it decodes the image, measures size, focus and exposure, and returns `retake` or `proceed`. The intended recognition path uses NVIDIA Retail DINO to propose product boxes and SigLIP2 to rank verified registry references for each crop. Approved reference masks can reduce background distraction, but they do not create missing detections; their benefit on real scenes remains to be measured. Optional barcode reading can help verify the identity of an already counted item without counting it twice. Human approval remains the boundary before any stock change.

## Current status — checked 2026-10-20 19:00 Europe/Zagreb

| Area | What is working or prepared | What remains |
| --- | --- | --- |
| Local InventoryLens MVP | Streamlit owner/employee views, SQLite SKU registry, MobileSAM/ResNet18 photo recognition, selective OpenCV SIFT/RANSAC label verification, reviewed count approval and scan history. | Approval currently **replaces** selected SKU totals at a location. The requested `ADD`/`SUBTRACT` movement ledger and customer-specific flow are designed but not implemented. |
| Live AWS OpenCV 5 pilot | The existing private S3 bucket and active Lambda in the WEB&SHOP project, selected Region `eu-north-1`, process explicitly submitted photos. The frozen audit verified matching hashes, actions and metrics for **176 distinct originals**. A later B07 guided-review check returned live OpenCV **5.0.0**, `proceed`, and the exact selected-photo SHA-256; the interface displayed the result alongside clearly labelled archived proposals. [Pilot evidence](docs/opencv5_aws_capture_pilot_result_20261020.md). | This pilot performs capture guidance only. Its one-day S3 lifecycle expires uploaded objects; it does not delete the retained pilot resources. It neither runs DINO/SigLIP2 nor changes inventory. |
| Earlier detector and recognizer work | NVIDIA Retail DINO found 27 of 33 physical instances on a separate five-image holdout at its tested settings, with six misses. Earlier SigLIP2 outputs and the recovered GPU benchmark are preserved as experimental evidence. | The detector result is a small, differently configured comparison. Archived model predictions are not live AWS inference or verified SKU accuracy on the newest scenes. |
| New-product dataset | **42 owner reference photos, 25 employee scenes, five customer scenes and ten provisional product groups** are audited. Thirty-four owner photos have paired original/masked crops. Owner-photo SigLIP2 top choice changed from **24/34 to 25/34** with masks, against provisional groupings. The hash-verified 30-scene AWS input bundle and isolated runner are ready locally. | No owner identity approvals or corrected `IMG_4857` boxes have been saved. Eight owner photos lack approved paired masks. The 30 employee/customer scenes have **no imported GPU predictions or scored recognition results**. The owner-photo result is not scene accuracy. |
| AWS 30-scene GPU run | One temporary `g6.xlarge` evaluation was approved within the existing **$10 maximum new AWS spend**. | AWS Support denied the 4-vCPU G/VT quota request on 2026-10-20 at 17:42 Europe/Zagreb and invited a detailed appeal (case `179250925200910`). A later Service Quotas API read still showed `CASE_OPENED`, so the Support correspondence is the operative result. The attempted Support API appeal failed with `SubscriptionRequiredException`; no appeal was submitted. The applied quota was last checked at **0 vCPUs**. No experiment assets were uploaded, no GPU instance was launched, and no 30-scene inference ran. The candidate DINO TensorRT model still needs a one-image GPU preflight. |
| Competition delivery | The [submission narrative and architecture](docs/webshop_opencv_2026_submission_20261020.md), [storyboard](docs/webshop_opencv_2026_demo_storyboard_20261020.md), [technical report](docs/webshop_opencv_2026_technical_report_20261020.md), and [176-image pilot audit](docs/opencv5_aws_capture_pilot_result_20261020.md) are prepared. | A judge-accessible demo video, code delivery and full WEB&SHOP role/inventory integration remain to be completed. The current demonstration must label DINO/SigLIP2 suggestions as archived replay data. |

## Problems and potential solutions

The [new-scene recognition handoff](docs/agent_handoff_new_scene_recognition_20261020.md) states the current user-visible failure, the requested behavior, and the exact acceptance check for a follow-up agent.

| Problem | Next practical solution |
| --- | --- |
| Crowded shelves can yield missed items, one box around two products, or a mask that spills onto a neighbor. | Run the fixed 30-scene DINO/SigLIP2 experiment when compute is available, then review **one box per physical item**, including misses and merges. Measure detector recall, count error and SKU results separately before changing the detector. Keep manual split/add controls. |
| Ten new SKU identities are provisional; the tealight, battery, cup, crown and glue stick need particular care. `IMG_4857` has two identical annotation boxes for two visible cans. | Use the [local registration and scene review app](experiments/new_product_scanning_20261020/README.md) to inspect photos, approve or correct exact SKU names/variants/sizes, and redraw two distinct can boxes. Leave uncertain identities unresolved. Original photos and pre-correction annotations remain preserved. |
| Reference masks increased correct provisional owner-photo top choices by one out of 34 queries and may remove useful packaging detail. | Compare original, masked and combined galleries on **identical detector proposals and query crops** in the pending scene run. Score only after identities and physical instances are reviewed; measure reference-photo count and angle effects on a fixed query set. |
| AWS GPU evaluation is blocked by a denied quota request and zero applied G/VT vCPUs; local Windows PyTorch/ONNX Runtime cannot run the required TensorRT/CUDA path. | Appeal through the AWS Management Console with the documented one-instance, four-hour, $10-limited use case, or separately choose the existing RunPod Pod after its runtime is verified. Neither route changes the already working AWS OpenCV pilot. Recheck price, quota and runtime before any AWS launch. [Execution plan](docs/webshop_aws_30_scene_gpu_evaluation_plan_20261020.md). |
| The local application's snapshot write path does not meet the intended live stock-movement behavior. | After the recognition review, implement the [isolated movement design](experiments/new_product_scanning_20261020/inventory_movement_design_20261020.md): confirmed SKU, location, quantity, `ADD`/`SUBTRACT`, actor, time, source scan, unique approval ID, atomic balance update and reversal. Do not post unconfirmed detections or change production inventory during this evaluation. |

## Next steps

1. Review the ten provisional product registrations and correct `IMG_4857` in the [local review app](experiments/new_product_scanning_20261020/README.md).
2. Reopen AWS Support case `179250925200910` with the detailed use case, or separately approve reuse of the existing RunPod Pod. For an AWS run, proceed only after the applied G/VT quota reaches at least four vCPUs; then run the one-image GPU preflight and the 30-scene comparison if it passes. Stay within the $10 AWS ceiling and preserve the existing 176-image pilot evidence.
3. Review the resulting annotated scenes before calculating detection, counting or SKU accuracy. Use those findings to decide whether detector changes, more reference photos or masks are justified.
4. Keep the stock-movement implementation separate until reviewed counts and the approval contract are ready; record the competition video from the already working OpenCV 5 AWS guidance, including the new B07 live check and clearly identified archived proposals.

The [complete GPU benchmark audit](data/recognition_checks/final_comparative_gpu_benchmark_20261018/complete_final_gpu_benchmark_audit_20261019.md) and the `Owner_test`, `employee_test`, `customer_test`, product-registration and new-product experiment files remain traceable supporting evidence. Production inventory and source datasets were not changed for this competition work. Training custom weights and video tracking are not implemented. AMD inference remains unverified.

## Public source checkout

The public Git repository contains source code, documentation and small written/tabular experiment records. Raw owner/employee/customer photos, generated crops and visual review pages, model weights, GPU input/output archives, local SQLite inventory, runtime caches and AWS upload manifests remain in this workspace and are excluded from Git. Some experiment documents refer to those local-only artifacts; their hashes and summaries are retained for provenance. A public checkout is therefore a source release, not a complete runnable copy of the private test dataset or deployed AWS pilot. Do not treat missing local artifacts as failed experiment results.

## Start On Windows

From PowerShell in this project directory:

```powershell
.\.venv\Scripts\python.exe model_assets.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

Open the local URL printed by Streamlit. Leave the server running while testing.

Local demo accounts (for this development environment):

| Role | Username | Password |
| --- | --- | --- |
| Owner | owner | owner123 |
| Employee | employee | employee123 |

Owners can create additional employee accounts in the Team tab.
These seeded credentials are for a local demo; change the account setup before public hosting.

## Try The Shared Workflow

1. Sign in as owner. Existing reference products appear under Products.
   Register one or more photographs together, or select an existing product and
   use Additional reference photographs > Add references. Each SKU supports up
   to four distinct views. For the current trial, use three views plus an optional
   fourth; registration still allows fewer while the protocol is evaluated.
   Removal keeps at least one active view; original
   image files are retained for historical diagnostics.
2. Choose or create a location under Inventory.
3. Optionally import a UTF-8 CSV with unique `sku` and nonnegative `quantity` columns.
   Preview and explicitly confirm replacing matching quantities at that location.
4. Open a second browser or private window and sign in as employee.
5. Choose the location, upload a photograph or use Camera, then run detection.
6. Review the original/annotated photo and select which SKUs this count covers.
7. Correct quantities, confirm the scope, and approve the count.
8. Return to the owner window and click Refresh inventory. Check history and download the location CSV.

Camera access depends on browser permissions and a secure origin. For a phone,
use an HTTPS deployment/tunnel with appropriate access control, or upload a photo.
Phone camera capture has not yet been tested on a physical device.

See [the provisional photo checklist](docs/photo-capture-protocol.md) for the
front/left/right/raised-front views, background setup and separate test scenes.

## Current Local MVP Count Scope

- Partial counts replace the selected SKU totals at the chosen location only.
  Excluded SKUs keep their existing quantities. Choose zero explicitly for an empty SKU.
- Complete counts include every registered SKU at that location, including zeros.
- Counts are snapshots; they are never added to previous counts.
- Replaying an approved scan session does not create a duplicate or reapply old quantities.
- Blank inventory quantities mean not counted yet, rather than zero.

CSV templates are stored per location. Exports update matching quantities while
preserving unrelated rows, columns, and existing names. Unregistered CSV rows are
retained in the export but are not added to the recognition catalogue.

## Persistence And Diagnostics

`data/inventorylens.sqlite3` is now the source of application inventory data.
The old `data/catalog.json` is imported once, preserved, and no longer edited by the app.
Reference photographs and historic test logs are preserved.
The test tracker still saves `data/last_detection.json`, `data/last_annotated.png`,
and `data/test_runs.jsonl`; new runs also retain original photographs.
Expected counts are optional and are not scored unless explicitly enabled.

## Photo Recognition

The default path creates real MobileSAM masks, embeds each masked product crop,
and compares it with every registered reference crop. Each SKU's score is its
highest similarity across its views, not an average or vote. The primary match
requires enough cosine similarity (default 0.80) and a sufficient gap from the
runner-up SKU (default 0.05). These are development defaults, not calibrated
probabilities. Similar packaging may still be confused.

If similarity passes but the embedding gap is ambiguous, an additional OpenCV
SIFT / RANSAC check compares printed label details across candidate reference
views. Recovery requires at least 12 distinct geometric inliers, 35% inlier
support, and coverage of at least 8% of both crops, with a plausible non-reflected
perspective transform. Exactly one SKU must verify, and it must agree with the
embedding winner. Close fourth/later SKUs are checked too, not just the displayed
top three. Weak embeddings, conflicting verified SKUs and insufficient detail
stay unassigned. Diagnostics preserve the failed embedding gap and report
`geometric_reference_match` separately; this is not a lowered gap threshold.
This check adds no model downloads or training. It can still miss feature-poor,
blurred, reflective or heavily curved labels and does not make all closed-world
embedding matches safe for unregistered products.

Unassigned regions are shown in yellow and excluded from proposed quantities.
They may include background regions; they are not proof that an unknown product
exists. Missed objects cannot be highlighted. Always inspect the original image.
The spotlight uses actual masks and preserves original image dimensions.
Legacy detections without masks get an explicitly labelled bounding-box spotlight.

Model setup downloads about 84 MB once into `data/models/` from official sources.
Processed references are cached by image content and reference-selection version
in `data/models/reference_cache/`. Selection changes rebuild the affected cache
without changing source photographs or reusing older, incorrectly selected crops.
Cold preparation scales with the number of photographs, not just the number of
SKUs. The 50-view grocery catalogue took roughly 25 minutes to prepare on this
CPU (about 30 seconds per view). Existing cached views are reused. The scanner
now shows reference preparation progress, the current analysis stage and measured
timings; diagnostic snapshots also store those timings. Previous warm photo
checks took approximately 40 seconds; this is not a latency guarantee.
New references require no model retraining.
The reference board includes every active view, and match diagnostics record
which view produced each SKU's score. The experimental YOLOE path still uses
only the primary reference. Adding/removing a view invalidates a pending scan
result; rerun detection before approving it.
Matcher updates also invalidate pending results, but reuse the existing reference
crop/embedding caches when reference preparation has not changed.

Scene matching rejects masks with broad contact on three image borders,
including table/background masks smaller than half the image.
The check uses a narrow border band to allow for one-pixel gaps in SAM outputs.
Large enclosed products and products cropped at only one or two borders remain
eligible. A true product heavily cropped at three borders may also be excluded;
this is a conservative geometry check, not semantic proof of background.
Reference photographs have a stricter rule: a mask touching at least 10% of
even one frame edge is ineligible. A registration photo must show one complete
product with background around every side. This prevents two-edge cardboard
walls/floors from winning over the product; it does not prove every remaining
mask is the correct whole object. Scene close-ups retain the less restrictive
rule above. If no eligible reference mask exists, recognition reports an error
rather than silently accepting the scene. Inspect processed views in the visual reference
board. Keep reference photographs separate from test scenes, and test absent
SKUs and similar packages as well as successful matches.

See [the recognition audit](docs/recognition-audit.md) for identified code problems,
the model choices, and the limits of the development results.

## Build A Segmentation Dataset

The approved 3-photo seed batch is preserved under `data/segmentation_dataset/approved/`.
To propose masks for new original grocery scenes, copy photos into
`data/segmentation_dataset/input_images/` and run:

```powershell
.\.venv\Scripts\python.exe tools\segmentation_dataset.py propose
```

Review every MobileSAM proposal in CVAT before explicitly importing an export as
approved. The proposal workflow does not use SKU matching and never starts model
training. See the [annotation workflow](docs/segmentation-annotation-workflow.md)
and [47-scene photo collection plan](docs/segmentation-photo-collection.md).

Reproduce a development count check on an original, unannotated photo:

```powershell
.\.venv\Scripts\python.exe check_recognition.py --image PATH_TO_PHOTO.png --expect TEST-002 2
```

Supply every present SKU with another `--expect SKU QUANTITY`; omitted registered
SKUs are expected to be absent. Reports and images go into `data/recognition_checks/`
without overwriting the latest app test. Use `--engine yoloe` for comparison, or
`--exclude-sku TEST-002` to test rejection with K18 absent from the reference catalogue.

Replay matching against saved real SAM masks, without rerunning segmentation:

```powershell
.\.venv\Scripts\python.exe check_matching.py
```

Use `--run RUN_ID` to select cases. Optional `--truth-overrides PATH.json` supplies
explicit, reviewed count annotations for comparison only; original app history is
never changed. Reports record source/reference hashes and the saved annotations.
Replay tests matching only. Use `check_recognition.py` for a complete fresh model
run, and new independent photographs to measure generalization.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use temporary databases and synthetic images. The Streamlit approval test
stubs inference to test permissions, manual correction, persistence, and duplicate
protection; it is not an accuracy evaluation. Test recognition separately using
unseen photographs and manually recorded expected counts.
