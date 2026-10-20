# Web&Shop guided visual scanning — InventoryLens OpenCV 5 experiment

This experiment supports the single [Web&Shop competition entry](../../docs/webshop_opencv_2026_submission_20261020.md). InventoryLens is the module under integration. The [storyboard](../../docs/webshop_opencv_2026_demo_storyboard_20261020.md) and [technical report](../../docs/webshop_opencv_2026_technical_report_20261020.md) distinguish live AWS capture guidance, local guidance, archived model predictions, and planned full integration.

This experiment implements the [frozen replay contract](../../docs/opencv5_guided_scan_replay_contract_20261020.md). It uses eight existing photos and the audited, saved NVIDIA Retail DINO and SigLIP2 predictions. It runs OpenCV 5 capture checks locally, then generates a visual review page and a machine-readable trace. The offline replay does not execute the models, contact AWS, or update InventoryLens inventory, registrations, source photos, or test histories. The separate live demo has an optional, explicit-click AWS capture check described below.

The experiment's `requirements.txt` pins OpenCV, NumPy, and Streamlit for this local demo. The main app's dependency file is unchanged.

From the project root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -B experiments\opencv5_guided_scan\replay.py `
  --archive 'C:\Users\U938\Downloads\inventorylens_nvidia_dino25_e2e_results_20261019.zip' `
  --run-name local_replay_20261020
```

The name must be unused. Outputs go only to `experiments/opencv5_guided_scan/runs/<run-name>/`:

- `replay.json`: hashes, OpenCV 5 metrics and capture decisions, DINO boxes, SigLIP2 suggestions, model provenance, and explicit scoring limits;
- `previews/*.jpg`: reduced overlays generated from the frozen photos;
- `review.html`: a local, offline page for changing SKU decisions, removing false boxes, adding missed items, splitting merged boxes into manual items, and entering one B10 arrangement total. Open this file in a browser. Its **Download review-only JSON** button saves decisions to your browser downloads; it does not call a server or commit inventory.

The script validates all eight source hashes, the saved archive hash, the one-to-one 29-box prediction join, and the two in-memory retake controls before creating the output directory. An existing output directory is never overwritten. The capture thresholds are illustrative and were chosen after looking at these photos; do not report capture quality accuracy or tune detection/recognition thresholds using this held-out set.

The historical replay page shows prior human observations as context. `recognized` means a threshold accepted a suggestion, not that the SKU is correct. E06 has no proposals despite passing the capture check. B07's single box covers several SKUs, B08 has a missed Oreo, and both B10 photos represent one arrangement. The earlier 11-unit B10 note in that frozen trace is superseded by the user's 10-unit correction below.

The user's 2026-10-20 E01 box observations are saved separately in `runs/local_replay_20261020/E01_box_review_20261020.json`. The saved predictions in `replay.json` remain unchanged. The review page includes an `Overlap/partial - exclude from count` action so a real product in a merged or overlapping proposal is not mislabeled as a false box. Only `accept` box decisions and manually added items on a photo explicitly marked reviewed are marked `countable` in a downloaded review.

The later user corrections are recorded in `runs/local_replay_20261020/review_corrections_20261020.md` and applied to a separate `runs/local_replay_20261020/review_reconciled_20261020.json`. The original browser export is unchanged. E01's eight-unit total is user-confirmed, while its per-SKU split is marked as a visual interpretation; B07 includes the user-confirmed Fentimans, and B10 has the user-confirmed 10-unit breakdown across one arrangement and two views.

## Live local guided scan

From the project root, run:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run experiments\opencv5_guided_scan\live_demo.py --server.address 127.0.0.1 --server.port 8599
```

Open `http://127.0.0.1:8599` locally. If that port is occupied, choose another local port in the command. Choose an upload, camera capture, or frozen sample. OpenCV 5 analyzes the actual image bytes and asks for a retake or lets you proceed. A person can override a retake prompt. Only an exact SHA-256 match to a frozen source photo receives the saved DINO/SigLIP2 proposals; all other photos get manual review without fabricated model results. The page downloads one review-only JSON for the selected photo. B10 requires one arrangement total; a single-photo export explicitly says it includes only one of the two views. The default local flow does not contact AWS or write to inventory.

To show the deployed AWS path, sign in locally with the `Web&Shop` AWS CLI profile, select a photo, tick **Use AWS OpenCV 5 capture check for this photo**, and click **Run AWS capture check**. The exact image bytes go to the private `eu-north-1` pilot bucket, and the app waits for a matching Lambda result before showing product review. The photo and result follow a one-day S3 lifecycle. Selecting the checkbox alone does not upload. The result appears in the review-only export under `aws_capture`. The pilot has a $10 new-spend ceiling and only approved test photos are in scope; see the [pilot result](../../docs/opencv5_aws_capture_pilot_result_20261020.md). The demo still does not execute DINO/SigLIP2 or write inventory. For an exact-hash match to an audited newer employee or customer scene, such as `IMG_4859.jpeg`, the app identifies the scene **before** the AWS button and explains that its 30-scene model proposals are still pending. It never substitutes annotations or guesses for missing predictions.

For a repeatable retake demonstration, select frozen sample B01 and choose `Blur in memory` or `Dark in memory`. The transform leaves the original source photo unchanged and never matches archived model output. The transformed bytes remain local unless the operator explicitly runs the AWS capture check; then the approved pilot stores that control temporarily under its one-day S3 lifecycle.

## AWS capture pilot

The isolated [AWS pilot source package](aws_pilot/README.md) contains a Lambda handler, pinned ZIP dependencies, and offline checks for the same OpenCV 5 capture decision. It was deployed and connected to the optional local-demo button. The [pilot plan](../../docs/opencv5_aws_capture_pilot_plan_20261020.md) lists the resources, cost assumptions, safeguards, and approval boundary; the [result report](../../docs/opencv5_aws_capture_pilot_result_20261020.md) records measured behavior and remaining limits.
