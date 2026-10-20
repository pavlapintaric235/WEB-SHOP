# Web&Shop live inventory tracking — InventoryLens visual scanning module

**Web&Shop is the application and OpenCV AI Competition 2026 entry. InventoryLens is the product-registration, visual scanning, SKU recognition, and inventory-intelligence module being integrated into it.** This workspace contains the existing local prototype and its supporting datasets and benchmarks. The AWS guided-scan pilot is part of the same Web&Shop demonstration, not a second project.

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
| Live AWS OpenCV 5 pilot | The existing private S3 bucket and active Lambda in the Web&Shop project, selected Region `eu-north-1`, process explicitly submitted photos. The frozen audit verified matching hashes, actions and metrics for **176 distinct originals**. A later B07 guided-review check returned live OpenCV **5.0.0**, `proceed`, and the exact selected-photo SHA-256; the interface displayed the result alongside clearly labelled archived proposals. [Pilot evidence](docs/opencv5_aws_capture_pilot_result_20261020.md). | This pilot performs capture guidance only. Its one-day S3 lifecycle expires uploaded objects; it does not delete the retained pilot resources. It neither runs DINO/SigLIP2 nor changes inventory. |
| Earlier detector and recognizer work | NVIDIA Retail DINO found 27 of 33 physical instances on a separate five-image holdout at its tested settings, with six misses. Earlier SigLIP2 outputs and the recovered GPU benchmark are preserved as experimental evidence. | The detector result is a small, differently configured comparison. Archived model predictions are not live AWS inference or verified SKU accuracy on the newest scenes. |
| New-product dataset | **42 owner reference photos, 25 employee scenes, five customer scenes and ten provisional product groups** are audited. Thirty-four owner photos have paired original/masked crops. Owner-photo SigLIP2 top choice changed from **24/34 to 25/34** with masks, against provisional groupings. The hash-verified 30-scene AWS input bundle and isolated runner are ready locally. | No owner identity approvals or corrected `IMG_4857` boxes have been saved. Eight owner photos lack approved paired masks. The 30 employee/customer scenes now have imported GPU predictions from the completed 2026-10-20 RunPod run (30 scenes, 93 boxes) but **no scored recognition results**: those predictions are provisional SigLIP2 ranks awaiting SKU approval and scene-label review. The owner-photo result is not scene accuracy. |
| 30-scene GPU run | One temporary `g6.xlarge` evaluation was approved within the existing **$10 maximum new AWS spend**, but the AWS route never ran. The user then authorized the evaluation elsewhere, and on 2026-10-20 it completed on a rented RunPod B300 GPU — outside AWS and outside that $10 ceiling — producing [30 scenes and 93 detected boxes](experiments/new_product_scanning_20261020/results/imported_runpod_scene_results_20261020_v5_flat/report.json) ranked against original, masked and combined galleries on identical detector proposals and query embeddings, with the previously validated detector remap (`c6185e3b…c36f41`) and a preflight gate on which every node ran on TensorRT with no CPU fallback. | AWS Support denied the 4-vCPU G/VT quota request on 2026-10-20 at 17:42 Europe/Zagreb and invited a detailed appeal (case `179250925200910`). A later Service Quotas API read still showed `CASE_OPENED`, so the Support correspondence is the operative result. The attempted Support API appeal failed with `SubscriptionRequiredException`; no appeal was submitted. The applied quota was last checked at **0 vCPUs**. No experiment assets were uploaded to AWS and no AWS GPU instance was launched. The RunPod result carries `accuracy_metrics` **null by design**: the ten SKU identities are unapproved and the scene labels unreviewed, so it is provisional SigLIP2 ranks and no accuracy, recall or count-error figure exists yet. The separate `7abf51f6` TensorRT-plugin DINO candidate was not used in that run and still needs its own one-image GPU preflight. |
| Local CPU recognition (new, 2026-10-21) | `shape_recognition.py` proposes one box **and** one instance mask per physical item on CPU, using OWLv2 and SAM 2.1 Hiera-tiny — both Apache-2.0, both loaded through `transformers`, no GPU and no AGPL dependency. `scan_photo.py` scans any photograph; `run_local_shape_detector.py` regenerates all 30 scenes into a directory both `review_app.py` and `evaluate_scene_predictions.py` can read. Measured: 30 scenes, 358 boxes, 7.3 s per photograph, and **80 of 93 GPU DINO boxes reproduced at IoU ≥ 0.50, mean IoU 0.885**. 25 new tests cover the box-suppression geometry. | This is detector **agreement**, not accuracy; no run here produces an accuracy figure, and Steps 1 to 3 are still required. It over-proposes (358 boxes against DINO's 93). Two identical items stacked and touching are still returned as one box. The detector finds only what `PROMPTS` names — two scenes returned nothing until ornament words were added. |
| Competition delivery | The [submission narrative and architecture](docs/webshop_opencv_2026_submission_20261020.md), [storyboard](docs/webshop_opencv_2026_demo_storyboard_20261020.md), [technical report](docs/webshop_opencv_2026_technical_report_20261020.md), and [176-image pilot audit](docs/opencv5_aws_capture_pilot_result_20261020.md) are prepared. | A judge-accessible demo video, code delivery and full Web&Shop role/inventory integration remain to be completed. The current demonstration must label DINO/SigLIP2 suggestions as archived replay data. |

## Problems and potential solutions

The [new-scene recognition handoff](docs/agent_handoff_new_scene_recognition_20261020.md) states the current user-visible failure, the requested behavior, and the exact acceptance check for a follow-up agent.

| Problem | Next practical solution |
| --- | --- |
| Crowded shelves can yield missed items, one box around two products, or a mask that spills onto a neighbor. | The fixed 30-scene DINO/SigLIP2 experiment has run (2026-10-20, rented RunPod B300 GPU: 30 scenes, 93 boxes, provisional ranks only, no accuracy). Next, review **one box per physical item** in those scenes, including misses and merges, and only then measure detector recall, count error and SKU results separately before changing the detector. Keep manual split/add controls. |
| Ten new SKU identities are provisional; the tealight, battery, cup, crown and glue stick need particular care. `IMG_4857` has two identical annotation boxes for two visible cans. | Use the [local registration and scene review app](experiments/new_product_scanning_20261020/README.md) to inspect photos, approve or correct exact SKU names/variants/sizes, and redraw two distinct can boxes. Leave uncertain identities unresolved. Original photos and pre-correction annotations remain preserved. |
| Reference masks increased correct provisional owner-photo top choices by one out of 34 queries and may remove useful packaging detail. | That comparison has now run: the 2026-10-20 scene run ranked original, masked and combined galleries on **identical detector proposals and query embeddings**. Its output is provisional ranks, so score only after identities and physical instances are reviewed; measure reference-photo count and angle effects on a fixed query set. |
| The **AWS** GPU route is blocked by a denied quota request and zero applied G/VT vCPUs, and local Windows PyTorch/ONNX Runtime cannot run the required TensorRT/CUDA path. | The rented RunPod route was separately authorized and used instead: its runtime passed the one-image preflight and the 30-scene comparison ran there on 2026-10-20, yielding provisional ranks rather than accuracy. If the evaluation is ever repeated inside AWS, appeal through the AWS Management Console with the documented one-instance, four-hour, $10-limited use case and recheck price, quota and runtime before any launch. Neither route changed the already working AWS OpenCV pilot. [AWS execution plan](docs/webshop_aws_30_scene_gpu_evaluation_plan_20261020.md). |
| The local application's snapshot write path does not meet the intended live stock-movement behavior. | After the recognition review, implement the [isolated movement design](experiments/new_product_scanning_20261020/inventory_movement_design_20261020.md): confirmed SKU, location, quantity, `ADD`/`SUBTRACT`, actor, time, source scan, unique approval ID, atomic balance update and reversal. Do not post unconfirmed detections or change production inventory during this evaluation. |

## Finishing the project — exact steps

Written for someone taking this over who has not worked on it before. Work top to bottom:
each step depends on the one above it. Where a step says a claim is not yet earned, that is
deliberate — this entry's credibility rests on never stating more than the evidence supports.

**The submission deadline is 2026-10-26, 23:59 Pacific** ([competition page](https://opencv26.devpost.com/)).
The Devpost header has been seen showing 23:45 PDT, so treat **23:45 Pacific on 2026-10-26**
as the real cutoff. Judging runs 2026-10-27 to 2026-11-09. Verify this on Devpost yourself
before planning around it, and confirm the mandatory midpoint check-in
(2026-09-21 to 2026-10-02) was filed — nothing in this repository records that it was.

### Step 0 — set up and confirm the repository is sound (about 15 minutes)

Use `.\.venv\Scripts\python.exe` for everything. A system Python without `streamlit` fails
twelve of the workflow tests for that reason alone, which looks alarming and means nothing.

```powershell
& .\.venv\Scripts\python.exe model_assets.py
& .\.venv\Scripts\python.exe -m unittest tests.test_local_features tests.test_matching_replay `
    tests.test_oct17_benchmark_eval tests.test_recognition tests.test_segmentation_dataset `
    tests.test_shape_recognition tests.test_workflow
```

Expected: 131 tests, all `OK` (15 + 3 + 11 + 33 + 8 + 25 + 36). Anything else is a real regression —
fix it before going further. `model_assets.py` downloads MobileSAM and ResNet18 into
`data/models/` and verifies their checksums; it is safe to re-run.

A public checkout is a **source release, not a runnable copy**. Photographs, crops, model
weights, the SQLite database and GPU archives are excluded from Git by design. Steps 1 to 3
need the private workspace. Steps 4 and 6 do not.

### Step 0b — recognize objects in a photograph locally, on CPU

Until now the recognition pipeline could not run on this machine at all. NVIDIA Retail DINO
needs TensorRT/CUDA, so scene proposals were frozen archived GPU output and no new photograph
could be processed. `shape_recognition.py` closes that gap with two permissively licensed
Hugging Face checkpoints that run on CPU through `transformers`:

| Stage | Model | Licence | Repository |
| --- | --- | --- | --- |
| Boxes | [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble) | Apache-2.0 | [google-research/scenic](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) |
| Masks | [`facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny) | Apache-2.0 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |

Scan any picture — no registry, no reference photographs, no GPU:

```powershell
& .\.venv\Scripts\python.exe scan_photo.py path\to\photo.jpg
```

It writes `<name>_shapes.jpg` (one colour per item) and `<name>_shapes.json` (boxes, mask
outlines, and the word the detector used for each shape). Add `--prompt "a screw"` to look
for something the built-in vocabulary does not name.

Regenerate proposals for all 30 evaluated scenes, in the format both the review app and the
evaluator read:

```powershell
& .\.venv\Scripts\python.exe experiments\new_product_scanning_20261020\run_local_shape_detector.py
```

**What this is and is not.** Boxes and masks are model proposals: every row is written
`review_status=unreviewed_proposal` and `acceptance_status=abstained` with an empty `sku_id`,
because none of the ten identities is approved. It produces no accuracy figure, and Steps 1
to 3 below remain exactly as necessary as before.

**Measured on the 30 evaluated scenes, 2026-10-21** (run `local_owlv2_sam21_20261021`).
30 photographs, 358 proposed boxes, 368 s total on a 6-core CPU — 7.3 s per photograph on
average, 12.8 s at worst. Against the committed GPU DINO run, `compare_detector_runs.py`
matched **80 of 93 DINO boxes at IoU ≥ 0.50, mean IoU 0.885**:

```powershell
& .\.venv\Scripts\python.exe experiments\new_product_scanning_20261020\compare_detector_runs.py `
    --a results\imported_runpod_scene_results_20261020_v5_flat\predictions.csv `
    --b results\local_owlv2_sam21_20261021\predictions.csv
```

That is agreement between two detectors, not accuracy — neither run is ground truth, and a
box found by only one of them may well be the correct one.

Three honest limitations, all observed rather than assumed:

* It **over-proposes**: 358 boxes where DINO drew 93, so only 22% of its boxes correspond to
  a DINO box. A tall item can attract several boxes. Over-proposal is recoverable by a
  reviewer rejecting a box; a missed item is not, which is why the thresholds lean this way.
* Two identical items **stacked vertically and touching** are returned as one box. `IMG_4857`
  is exactly this case and the detector does not separate it at any usable threshold, so the
  reviewer must still split that box by hand.
* It finds only what it has a **word** for. The first full run returned nothing at all on
  `IMG_4887` and `IMG_4889`, whose subject is the gold crown, until four ornament words were
  added to `PROMPTS`. Extending that tuple needs no retraining, no labelled data and no GPU.

### Step 1 — approve the ten SKU identities (blocks everything downstream)

Nothing can be scored until a person confirms what each product actually is. The run reports
`accuracy_metrics: null` precisely because this step has not been done.

```powershell
& .\.venv\Scripts\python.exe -m streamlit run experiments\new_product_scanning_20261020\review_app.py
```

Pick **Product registration** in the sidebar's `Workspace` selector (the four workspaces are
Product registration, Reference masks, Scene review and Evaluation) and work through all ten
groups. The overview deliberately puts
the five hard identities first: tealight, battery, cup, crown, glue stick. For each, read the
source photograph and confirm the exact name, brand, variant and package size. **Leave anything
you cannot positively identify unresolved** — an unconfirmed SKU is a usable result, a guessed
one silently corrupts every metric computed afterwards.

Done when `experiments/new_product_scanning_20261020/annotations/sku_registry_review.json`
exists. It does not exist today: **zero identities are currently approved.**

### Step 2 — review the scene labels, and fix the known IMG_4857 defect

In the **Scene review** workspace, go through all 30 scenes and mark one box per physical item,
including items the detector missed entirely and single boxes covering two products.

`IMG_4857` carries **two identical annotation boxes for two visible cans** and must be redrawn
as two distinct boxes. Label genuinely unrecognisable objects `__unknown__` — that is the only
way unknown-rejection becomes measurable at all.

Done when `annotations/scene_labels.json` exists. It does not exist today: **no scenes are
reviewed.**

### Step 3 — only now, compute accuracy

With Steps 1 and 2 complete:

```powershell
& .\.venv\Scripts\python.exe experiments\new_product_scanning_20261020\evaluate_scene_predictions.py `
    --predictions experiments\new_product_scanning_20261020\results\imported_runpod_scene_results_20261020_v5_flat\predictions.csv
```

`--predictions` is required; `--iou` defaults to 0.50. This matches boxes one-to-one at that
threshold, excludes unconfirmed identities, and reports
employee and customer figures separately. It scores boxes, not mask IoU. Until it runs,
the honest and correct description of the 30-scene run remains "provisional SigLIP2 ranks",
and **no precision, recall, count-error or accuracy number may be quoted anywhere.**

This is the single highest-value remaining task: it converts the existing GPU run from
"we ran something" into the measurable evaluation evidence the competition explicitly requires.

### Step 4 — record the video (hard requirement, under five minutes)

Follow [the storyboard](docs/webshop_opencv_2026_demo_storyboard_20261020.md) shot by shot; it is
already timed to fit. No video file exists in this repository yet.

Show the live AWS OpenCV 5 capture gate in `eu-north-1` actually returning `retake` or `proceed`
— that is the deployed component, and it genuinely works. Label archived DINO/SigLIP2 output
**on screen** as replayed archived data. The visual walkthrough at `docs/demo/index.html`
(rebuild with `& .\.venv\Scripts\python.exe docs\demo\build_demo.py`) is a good screen-recording
source for the recognition section.

Do not state an accuracy figure in the video unless Step 3 produced one.

### Step 5 — optional, only with time left over

- **Approve the 8 owner photos still lacking masks** (34 of 42 are approved) in the review app's
  **Reference masks** tab.
- **Review the 93 scene masks.** They are MobileSAM proposals prompted with the committed
  detector boxes, every row marked `review_status=unreviewed_proposal` in
  `03_newest_10_products_owner_employee_customer/scene_masks/scene_mask_index.csv`. They were
  not part of the ranking run. Regenerate with
  `& .\.venv\Scripts\python.exe experiments\new_product_scanning_20261020\build_scene_masks.py`.
- **Re-rank with masked scene crops as queries.** This needs a GPU and costs money; the AWS G/VT
  quota is still 0 vCPUs, so it would mean renting again. Get explicit approval first.
- **The ADD/SUBTRACT movement ledger** is specified in
  [the movement design](experiments/new_product_scanning_20261020/inventory_movement_design_20261020.md)
  and is **not implemented** — the app still replaces SKU totals on approval. This is a known,
  documented gap, not a defect to hide. Do not start it before Steps 1 to 4 are finished.

### Step 6 — submit

Assemble on Devpost: the [submission narrative and architecture diagram](docs/webshop_opencv_2026_submission_20261020.md),
the [technical report](docs/webshop_opencv_2026_technical_report_20261020.md), the
[176-image pilot audit](docs/opencv5_aws_capture_pilot_result_20261020.md), this repository, and
the Step 4 video. Re-read the requirements on Devpost directly rather than trusting this list.

Before submitting, check every claim once more against what actually shipped. If Step 3 was not
completed, say so plainly in the submission — a documented limitation is worth more to judges
than an unsupported number, and the competition asks explicitly for limitations and
responsible-use considerations.

### Would more photographs help?

Asked often, so here is the measured answer. **Nothing in this project is trained** — there is no
training code anywhere in it. Retail DINO proposes boxes, SigLIP2 embeds crops, MobileSAM
segments, ResNet18 matches locally, and all four are used off the shelf. Extra photographs never
adjust any weights; they enter a **gallery** compared by cosine similarity at inference time. The
real question is therefore "more reference views per SKU?", and that has been measured on this
project's own photographs.

**Where more images demonstrably help.** From the
[reference retrieval report](experiments/new_product_scanning_20261020/results/reference_retrieval_report_20261020.md),
16 fixed queries over 3 groups, each query excluded from its own gallery:

| Gallery size | Top-1 macro | Top-5 micro |
|---:|---:|---:|
| 1 reference | 88.8% | 92.9% |
| 2 references | 94.4% | 98.3% |
| 3 references | 98.1% | 100% |
| 4 references | 100% | 100% |

A real effect, steepest from 1 to 2. But that is a controlled capture over 3 groups with
unconfirmed identities and no scene queries: the 100% is not a deployment estimate. The catalogue
currently holds **42 references across 10 SKUs** — five SKUs at 4, two at 3, the rest at 5-6.

**Where more images do not help at all.**

1. *Abstention.* The [catalogue-growth test](experiments/new_product_scanning_20261020/results/catalogue_growth_robustness_20261020.md)
   ran the scenes against a gallery containing none of the photographed products. The system
   labelled all 93 boxes anyway, with **79 of 93** below a 0.05 top1-top2 margin. More references
   make this worse, not better — a larger gallery offers a closer-looking wrong answer. The fix is
   a calibrated rejection threshold, which needs the `__unknown__` labels from Step 2.
2. *Detector misses.* DINO found 27 of 33 instances on the five-image holdout. The detector is
   class-agnostic and never sees the references, so no reference photograph can recover a missed
   item.
3. *The dominant confusion.* Of nine wrong top-1 results in the all-available run, **seven were
   oatmeal (`IL-NEW-005`) versus rice mash (`IL-NEW-006`)** — same brand, both chocolate, nearly
   identical front faces. More photographs of the same front face cannot separate them;
   discriminative views of the differing panel, or a barcode, can.

**What to do before the 2026-10-26 deadline.**

1. **Finish Steps 1 to 3 first.** You cannot tell whether more images help scenes, because scene
   accuracy has never been computed. The GPU run is already paid for; the measurement is free.
   Photographing before measuring is guessing.
2. **If you do shoot more, shoot exactly nine.** The report derives this: five groups need one
   more, two groups need two more, for a uniform four-reference test with one held-out query each.
   That is an afternoon, not a dataset campaign. Prioritise **angle diversity** (front/side/back
   where missing) over raw count, and capture the discriminating faces of the oatmeal/rice-mash
   pair.
3. **Do not bulk-collect.** At this catalogue size the curve saturates by four references, and
   unreviewed images only add review burden against the deadline.

On the idea of adding the folders in sequence, `01_` then `02_` then `03_`: that experiment has
already run, as the cumulative stages A, B and C of the catalogue-growth test. Nearly doubling the
catalogue to 19 SKUs changed the top-1 proposal on **1 of 93 boxes — 98.9% stable**. Catalogue
growth is not the risk. Answering confidently about products that were never registered is.

### Rules that are not negotiable

- **Never** quote an accuracy, recall, precision or count-error figure that Step 3 did not produce.
- **Never** present the AWS pilot as running DINO/SigLIP2. It performs capture guidance only.
- **Never** present replayed archived model output as live inference.
- Do not create billable AWS resources, deploy, or touch production inventory without explicit
  approval. All Regional resources belong in `eu-north-1`.
- Original photographs and pre-correction annotations are never edited. Review decisions are
  append-only.

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
