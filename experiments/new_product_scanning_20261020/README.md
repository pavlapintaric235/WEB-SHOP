# New Product Scanning Experiment

This InventoryLens module experiment is supporting evidence for the single [WEB&SHOP OpenCV competition entry](../../docs/webshop_opencv_2026_submission_20261020.md). It is an isolated local workflow for the 72 audited owner, employee, and customer photos. The app hash-checks source bytes and applies EXIF orientation for display/annotation. It does not connect to the production registry or database and never edits original photos. The [technical report](../../docs/webshop_opencv_2026_technical_report_20261020.md) keeps these photos and product-registration reviews distinct from the live AWS OpenCV 5 capture pilot and archived GPU predictions.

## Newest saved recognition review

Open the local review page (local artifact: `results/latest_recognition_review_20261020.html`) in a browser to see all ten provisional product groups, their box and masked reference crops, every paired owner-photo top-1 result, and contact sheets of the newest employee/customer scenes. The page is generated from the frozen `siglip_reference_retrieval_20261019T233831Z` run; regenerate it without running a model using:

```powershell
& .\.venv\Scripts\python.exe -B experiments\new_product_scanning_20261020\build_latest_recognition_review.py
```

As of 2026-10-20, 34 owner masks are approved. On the same 34 owner-photo queries, top-1 retrieval was 24/34 with box crops and 25/34 with masked crops; the mask corrected `IMG_4832.jpeg` and did not change the correctness of the other 33. Both owner query and gallery images used the selected crop style in this comparison. This is controlled SigLIP2 reference retrieval against **unconfirmed provisional SKU groups**, not employee/customer recognition accuracy. The new v3 scene GPU bundle includes both paired reference galleries, but no results ZIP for its 25 employee plus 5 customer scenes has been imported. Reference masks do not change DINO's scene detection boxes in the current workflow.

An isolated [OpenCV 5 recognition feasibility check](results/opencv5_recognition_feasibility_20261020.md) tested retail barcode decoding and a simple ORB package matcher on these owner photos. Photo-based product counting remains the primary planned workflow: DINO boxes, SigLIP2 suggestions against confirmed registered reference photos, OpenCV capture/item guidance, and human approval before any scoped stock update. Barcode verification is optional identity evidence for an already counted item. The check is reproducible with `evaluate_opencv5_recognition.py`; it does not add barcode decisions to the live AWS pilot or production registry.

## Human-boxed item SKU suggestions (2026-10-23)

The complete 30-scene human item-box export was ranked locally on CPU against all 34 approved owner reference crops using the pinned SigLIP2 model. This produced provisional top-three suggestions for 291 item crops in [`reviewed_item_sku_ranking_20261023T075154Z/`](results/reviewed_item_sku_ranking_20261023T075154Z/report.json). Open `sku_decision_review.html` (local artifact: `results/reviewed_item_sku_ranking_20261023T075154Z/sku_decision_review.html`) to inspect each photo and record a human choice, unknown item, or uncertain result; export decisions JSON from the page when finished. Choices remain in that browser until exported. The older `sku_review.html` is a read-only ranking view.

All ten candidate product identities are still unconfirmed, so the run reports no SKU accuracy. Confirm each candidate against packaging or a barcode first, then correct any item box that merges multiple physical units or covers only part of an item. In the spot-check, `IMG_4867.jpeg` has a large tealight-stack box plus several tiny overlapping boxes around the stack; review whether those represent separate candles or annotation errors before scoring. Dense photos such as `IMG_4900.jpeg` and `IMG_4902.jpeg` also need careful box and count review. After decisions are exported, import/validate them against the frozen item-box export, then compute per-SKU top-1/top-3 accuracy and unknown coverage. This remains a cropped-item retrieval test; a separate end-to-end detector-plus-SKU evaluation is still needed.

The owner's 2026-10-24 decision export is preserved at `annotations/inventorylens_sku_decisions_20261024.json` (local artifact: `annotations/inventorylens_sku_decisions_20261024.json`). It has 281 unique item decisions out of 291. Of those, 276 have both a `confirmed` state and a candidate ID; 10 items are absent, and 5 exported rows were excluded because they have no review state, are uncertain, or have no selected candidate. Against the combined SigLIP2 ranking, the confirmed subset has top-1 candidate-group agreement of 260/276 (94.2%) and top-3 agreement of 270/276 (97.8%). The reproducible comparison is [`sku_decision_evaluation_20261024/report.json`](results/reviewed_item_sku_ranking_20261023T075154Z/sku_decision_evaluation_20261024/report.json) with per-item rows in `item_comparison.csv`; run `python experiments/new_product_scanning_20261020/evaluate_sku_decisions.py` from the repository root to regenerate. Because the review page displayed the model suggestions and the candidate groups are not verified retail SKUs, treat these as provisional human/model agreement on the reviewed subset, not blind-test or sellable-SKU accuracy. No confirmed unknown/not-listed examples were supplied, so unknown rejection remains unmeasured.

To continue the decision review without losing prior selections, the preloaded page `sku_decision_review_preloaded.html` (local artifact: `results/reviewed_item_sku_ranking_20261023T075154Z/sku_decision_review_preloaded.html`) contains those 281 rows and leaves all 291 item slots visible. Rebuild it with `python experiments/new_product_scanning_20261020/build_sku_decision_review.py experiments/new_product_scanning_20261020/results/reviewed_item_sku_ranking_20261023T075154Z --decisions experiments/new_product_scanning_20261020/annotations/inventorylens_sku_decisions_20261024.json`. The remaining 10 are `IMG_4874.jpeg#14–#19`, `IMG_4876.jpeg#9`, `IMG_4892.jpeg#5`, and `IMG_4902.jpeg#4, #17`. The page does not auto-confirm them; each remains for human review. Exact retail SKU confirmation also needs package identifiers such as a readable barcode/GTIN, since several candidates are generic objects or lack enough visible variant details.

An updated complete export from 2026-10-24 is preserved at `annotations/inventorylens_sku_decisions_20261024_v2.json` (local artifact: `annotations/inventorylens_sku_decisions_20261024_v2.json`). It covers all 291 items: 290 confirmed candidate labels and one uncertain item. On the 290 confirmed labels, combined-gallery top-1 agreement is 271/290 (93.4%) and top-3 agreement is 283/290 (97.6%). See [`sku_decision_evaluation_20261024_v2/report.json`](results/reviewed_item_sku_ranking_20261023T075154Z/sku_decision_evaluation_20261024_v2/report.json) and [`item_comparison.csv`](results/reviewed_item_sku_ranking_20261023T075154Z/sku_decision_evaluation_20261024_v2/item_comparison.csv). The preloaded page has been refreshed from this complete export. Interpretation limits above still apply: these are human/model candidate-group agreement scores, not blind or verified retail-SKU accuracy.

## Product identity and barcode verification (2026-10-24)

Rechecked all 42 hash-verified owner reference photos with OpenCV 5 at original and 1.5x resolution. The same three retail codes decode; higher-resolution scanning found no additional codes. Each of the three codes agrees with package text: IL-NEW-002 is Nescafe Classic 3-in-1, 16.5 g (EAN-8 `59945383`); IL-NEW-005 is Dobra Mogador oatmeal with chocolate, 65 g (EAN-13 `8595054904258`); IL-NEW-006 is Dobra Mogador rice porridge with chocolate and banana, 50 g (EAN-13 `8595054909086`). External product listings corroborate the first two [Nescafe listing](https://zakupy.bydgoszcz.pl/nescafe-3in1-classic-rozpuszczalny-napoj-kawowy-1/105), [Dobra oatmeal listing](https://cenik.rs/cena/dobra-zobena-ovsena-kasa-cokolada-8595054904258/); the rice-porridge match is supported by the source photo barcode and package text. The remaining seven have not been identified to exact retail GTIN from these photos. Their packaging descriptions are retained as candidate descriptions only.

The evidence ledger is [`sku_identity_verification_20261024.json`](results/sku_identity_verification_20261024.json) with a readable summary in [`sku_identity_verification_20261024.csv`](results/sku_identity_verification_20261024.csv). These findings do not change the owner-confirmation field, candidate registry, or production data. Candidate IDs such as `IL-NEW-005` remain experiment-local labels; the GTINs are the exact package identifiers established here.

## Start (Windows PowerShell)

From Windows PowerShell, run these two lines exactly:

```powershell
Set-Location 'C:\Users\U938\Downloads\inventorylens_starter'
& .\.venv\Scripts\python.exe -m streamlit run .\experiments\new_product_scanning_20261020\review_app.py
```

The app sets its own local cache and runtime directories. The equivalent explicit environment setup from the repository root is:

```powershell
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) 'experiments\new_product_scanning_20261020\runtime\ultralytics'
$env:MPLCONFIGDIR = Join-Path (Get-Location) 'experiments\new_product_scanning_20261020\runtime\matplotlib'
$env:WEBSHOP_HF_CACHE = Join-Path (Get-Location) '.venv\huggingface_cache'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m streamlit run experiments\new_product_scanning_20261020\review_app.py
```

Open the local URL Streamlit prints, normally `http://localhost:8501`. The prepared local environment already has required packages. PyTorch is CPU-only, so MobileSAM and SigLIP2 run locally and may take time; cached model weights only are used. Missing checkpoints/cache are reported rather than downloaded.

## Review sequence

1. **Product registration:** the ten-row overview opens with the five difficult identities first: tealight, battery, cup, crown and glue stick. Inspect each source photo and its approved masked crop before editing the candidate ID, name, brand, variant and package size in the form. Confirm a distinct sellable SKU only when its identity is known. Save writes the experimental registry to `annotations/sku_registry_review.json` and appends an audit event to `annotations/sku_registry_review_events.jsonl`. Uncertain items remain unresolved; original photos and the candidate CSV are untouched.
2. **Reference masks:** choose an owner photo, generate MobileSAM proposals, inspect the green mask boundary, then approve or reject. Approved artifacts include a source-sized mask, polygon JSON, unmasked box crop, masked crop, and optional SigLIP2 embeddings for both crops. Decisions are append-only in `annotations/mask_review_events.jsonl`; proposals are not accepted automatically.
3. **Scene review:** the employee page opens on `IMG_4857.jpeg`, whose two saved records have exactly overlapping boxes. Inspect the side-by-side visual (local artifact: `results/img_4857_duplicate_annotation_review_20261020.html`), redraw one different full-resolution box for each physical can with the form, check identities, then save. The pre-correction annotation file is preserved as `annotations/scene_labels_before_img4857_review_20261020.json`; saved changes append to `annotations/scene_review_events.jsonl`. Add one box per physical item, including detector misses; use `__unknown__` for unresolved identity; link items to model boxes and mark merged detections; choose a coarse visibility class; mark the complete image reviewed only after checking the entire frame. Touching products need separate boxes. A multi-view arrangement must be explicitly confirmed; the evaluator never sums counts across views.
4. **Evaluation:** the owner-photo retrieval script excludes each query from its own SKU gallery and reports top-1/top-5 while varying the target SKU's gallery count. Rival SKU galleries stay complete to isolate the target's reference-count effect. Full-photo, unmasked box-crop, and approved masked-crop comparisons use fixed query photos where available. This controlled retrieval test is not scene accuracy and cannot measure unknown rejection.

## 30-scene GPU predictions

The current [AWS-first plan](../../docs/webshop_aws_30_scene_gpu_evaluation_plan_20261020.md) is the proposed path for the WEB&SHOP evaluation. AWS Support denied the four-vCPU GPU quota request on 2026-10-20 and invited a detailed appeal; the applied quota remains zero. A separately hash-verified TensorRT-plugin DINO candidate and AWS v4 input ZIP (local artifact: `results/aws_scene_inference_input_20261020_v4.zip`) are prepared locally; the candidate still needs GPU validation, and no 30-scene predictions have been imported. The RunPod material below documents the preserved v3 input/runner and earlier environment audit. The user independently started the existing Pod and asked that it be left running; this experiment has not launched inference there.

The isolated v3 input bundle and instructions are in `runpod_inference/README_RUNPOD.md`. It packages 30 scenes and 34 paired owner references (68 crops) and keeps model weights external. GPU inference has not been authorized or started by this workflow. After a permitted run, download the Pod's results ZIP and use `verify_scene_results.py` with its Pod-printed SHA-256 and `--import-results`; the verifier imports only after checking outer and inner hashes, ZIP CRC, scene count, gallery conditions and embeddings. The Scene review page will then show NVIDIA DINO boxes and original, masked, and combined SigLIP2 top-3 suggestions separately from human-reviewed labels. Similarity scores are rankings, not calibrated confidence; predictions do not become labels automatically.

Scene prediction CSV format (one row per detector box; coordinates are EXIF-corrected display pixels):

```csv
photo_filename,detection_id,sku_id,acceptance_status,confidence,x1,y1,x2,y2
IMG_4876.jpeg,det-1,,abstained,0.91,120,80,280,420
```

The scene evaluator includes only images marked review-complete, rejects duplicate human boxes, matches boxes one-to-one at IoU 0.50 by default, and reports employee/customer metrics separately. SKU metrics exclude identities not confirmed in the experimental registry. An accepted prediction requires a separately justified acceptance decision; the model's raw top choice is not automatically accepted. Unknown rejection is measurable only for human-reviewed objects explicitly labeled `__unknown__`. It scores boxes, not mask IoU.

## Boundaries

- Dataset audit: `audit/dataset_audit.md`; file provenance and hashes: `audit/photo_manifest.csv`.
- Proposed experiment-only IDs and reference mapping: `audit/candidate_skus.csv`.
- Outputs and annotations remain within this folder; each result run uses a new timestamped directory.
- No model training, paid GPU, CVAT task, production change, network download, or benchmark overwrite is performed.
- Check the existing MobileSAM and SigLIP2 model/checkpoint licenses and intended commercial-use terms before deployment or redistribution. We do not copy model weights into this experiment package.
