# Fresh-photo SKU scan: local application and GPU worker

## Latest connection: 25 October 2026

The current GPU worker is on RUNNING Pod `EXISTING_POD_ID`, listed at **$0.89/hour**. The previous Pod `PREVIOUS_POD_ID` is no longer returned by RunPod. Key-only direct SSH is `root@GPU_HOST -p 11998`; its host key was verified through the authenticated RunPod gateway and added to `data/webshop_demo/runpod_known_hosts`. Password and keyboard-interactive authentication are disabled.

The existing epoch-11 checkpoint, spec and worker were copied without retraining or downloading models. The checkpoint hash remains `8bb11d6d63f26edfd480bdf1264467bd5c080f190fbb83a034b252d13ee20d27`. The worker binds only to remote loopback, and the restored private tunnel serves `http://127.0.0.1:8765`. The saved worker token was preserved. The existing local app remains at **http://localhost:8509**.

Actual fresh DINO computation on historical `IMG_4699.jpeg` returned **six proposals**, **14.950 s** detector time and **24.658 s** request time. This verifies the connection, not unseen-photo accuracy or a new SKU-matching evaluation. No inventory approval was submitted. Evidence: `data/webshop_demo/reconnection_20261025.json`.

To reopen the tunnel from this project directory:

```powershell
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=data/webshop_demo/runpod_known_hosts -p 11998 root@GPU_HOST
```

This replacement also has no persistent volume: `/workspace` is on its container overlay. Application/database hosting was not migrated. The older connection details below are historical.

The **25 October practical UX demo** is on **http://localhost:8509**, with an isolated fictional database. See [the workbook, grouped review, buyer totals and map guide](webshop_practical_ux_20261025.md). The original release server remains on 8508. Fresh-photo review now uses SKU groups and a separate confirmation step that defaults to review only.

## LIVE connection restored — 24 October, approximately 22:24 Zagreb

Following explicit user approval, the existing RUNNING Pod `PREVIOUS_POD_ID` was restored using its registered SSH public key, with password authentication disabled. Direct SSH is `root@GPU_HOST -p 19088`; the server host key was verified through its authenticated RunPod gateway and saved locally in `data/webshop_demo/runpod_known_hosts`. No Pod was created/restarted, no weights downloaded, and no training performed. The existing local epoch-11 checkpoint, spec and worker were copied to `/workspace/webshop-worker`; the remote SHA-256 matches `8bb11d6d63f26edfd480bdf1264467bd5c080f190fbb83a034b252d13ee20d27`. The existing worker token was preserved.

The loopback worker and private SSH tunnel are running. Both launchers reuse `data/webshop_demo/worker_connection.json`. The release app is available at **http://localhost:8508**. Employee → Fresh-photo SKU scan → Upload photo / Camera → On this computer → Check this photo → Find products. Select Upload/Camera for LIVE, not the explicitly archived replay choice. Review and final approval remain required.

Every successful recognition is archived automatically under `<WEBSHOP_DATA_DIR>/fresh_scans/<business_id>/<scan_id>/` (for example, `data/webshop_release_demo/fresh_scans/1/<scan_id>/`). Each folder contains the original photo, EXIF-corrected `upright.png`, annotated `analysis.png`, and `analysis.json` with DINO output, SigLIP2 SKU suggestions, capture metrics, timings, and review edits. The JSON and annotated image update as edits are made and after inventory approval. The scan screen shows the saved folder path. These files stay on this computer and are separate from the inventory receipt; taking a photo or running analysis does not change stock. Original uploads may include image metadata.

Actual UI smoke: `IMG_4699.jpeg`, historical input, real DINO and local current-shop SigLIP2, six proposals reached review, **87.81 s** recognition including **15.437 s** DINO, zero scan sessions before/after and unchanged stock. Evidence: `data/webshop_release_demo/live_connected_ui_20261024.json`. A separate direct request returned six boxes in 15.621 s DINO / 24.331 s round trip (`live_detector_20261024.json`). These are actual computations on an existing photo, not replay, new-photo generalization or accuracy claims. Only upload selection was automated in the UI check.

One initial UI run exited with Windows native access violation `0xC0000005`, without a Python traceback. A diagnostic rerun with `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2` and fault diagnostics passed. This does not establish the root cause of the first crash. The launchers now use those CPU limits by default; `WEBSHOP_CPU_THREADS` can explicitly override them. No models or inference algorithms were changed.

The existing Pod remains running at its listed **$0.89/hour** for use. Its current `/workspace` is on the container overlay without a mounted persistent volume; stopping/recreating the container may require restoring these local assets and SSH setup again. The verified checkpoint remains local. No resources were stopped or deleted.

To reconnect a closed local tunnel, use the verified existing host-key file (if the Pod's host key changes, verify it again through the gateway first):

```powershell
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=data/webshop_demo/runpod_known_hosts -p 19088 root@GPU_HOST
```

In another terminal: `.\tools\start_webshop_release.ps1`. To repeat the actual UI smoke without approving inventory:

```powershell
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'; $env:OPENBLAS_NUM_THREADS='2'
.\.venv\Scripts\python.exe -X faulthandler tools\check_connected_scan_ui.py --data-dir data\webshop_release_demo --output data\webshop_release_demo\live_connected_ui_20261024.json
```

An optional `--photo <path>` tests a different supplied image through the same live flow. Complete Python files updated for this connection work: `./tools/setup_dino_test_worker.py` and `./tools/check_connected_scan_ui.py`.

The inspection notes below describe the state **before** this approved restoration and are retained as historical diagnosis.

**Release update:** the shared catalogue import, buyer view, distinct stock actions and explicit replay fallback are now implemented. Use [the 24 October release report](webshop_release_20261024.md) for current launch instructions, test results and limitations. The notes below preserve the earlier worker setup and historical smoke evidence; their feature-status paragraphs are superseded by that report.

**Latest live inspection, 24 October:** RunPod `PREVIOUS_POD_ID` is RUNNING in the existing TAO 5.5 image, at listed $0.89/hour. Authenticated gateway inspection verified RTX 4090 / 24564 MiB, CUDA available, `/usr/local/bin/dino`, and importable TAO deformable attention. Its expected `/workspace/webshop-worker` and `/workspace/inventorylens-dino-eval` assets are absent; no worker, direct SSH daemon or authorized_keys file exists. Direct SSH `GPU_HOST:19088` refuses the handshake; gateway PTY works. The saved local connection still names port 37882, and the local 8765 tunnel is unreachable. The verified 390,300,806-byte epoch-11 checkpoint and pinned SigLIP2 weights already exist locally; no model download or retraining is needed. TensorRT 8.6.3 initializes NVIDIA plugins, while the installed mmdeploy plugin fails to load `libcublas.so.11`; the existing live TAO PyTorch worker does not use that TensorRT path. Detailed local evidence: `data/webshop_demo/live_connection_check_20261024.json`. Restoring direct SSH access and copying the existing assets needs approval under the user's permission/credential-change rule. Neither action has been performed. Both demo launchers now reuse the existing connection file; the worker setup helper preserves its saved token.

The existing application now includes **Fresh-photo SKU scan** in the Workspace menu for owners and employees. It uses registered references from the signed-in shop. The original Scanner remains available with its existing model choices.

## Isolated second-set demo catalog

`tools/prepare_webshop_demo.py` creates the isolated initial demo. `tools/expand_webshop_demo_catalog.py` adds the rest of the registered second-set catalog and reference photos without replacing existing demo records or counts. The current demo has 14 products and 50 references under `data/webshop_demo`. New products have blank price, shelf and opening quantity for the owner to set. Verify package variants and sizes before using a catalog identity for an accuracy claim.

The prepared Excel workbook is `data/webshop_demo/demo_catalog.xlsx`; a CSV copy and `demo_shop.json` record the same products. The Products and Shelves sheets are demo fixtures. The existing owner import accepts SKU/quantity CSV; full catalog/shelf XLSX import is still planned. Do not imply that the workbook import is implemented.

Start in a local **PowerShell** terminal in the project directory:

```powershell
$env:WEBSHOP_DATA_DIR = Join-Path $PWD 'data\webshop_demo'
& .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8507 --server.address 127.0.0.1
```

Open http://localhost:8507. The local demo credentials are `owner` / `owner123` and `employee` / `employee123`. They are intended for this local demo only. Select **Fresh-photo SKU scan** after sign-in.

## What works before a detector is connected

- Select the shop location, upload/take a photo and run local OpenCV 5 capture checks.
- Explicitly choose the existing AWS capture service and click **Check this photo** to call it. AWS credentials and the deployed pilot must be available; errors request a retry or an explicitly selected local check. No automatic cloud/local substitution occurs.
- Run actual local SigLIP2 gallery setup/self-check using `tools/check_webshop_matcher.py`. Its output is `data/webshop_demo/matcher_readiness.json`. Queries are gallery references, so this is not unseen-photo accuracy.

The new scan path does not substitute archived predictions. It clearly reports when the worker is not connected. On 24 October, actual DINO inference and local SigLIP2 matching were verified through the scan screen on a historical second-set photograph, as detailed below. Generalization to new photographs remains to be tested.

## GPU-side worker

Use the existing matching TAO 5.5 PyTorch environment, trained checkpoint and training experiment spec. The worker starts `dino inference` for each request; initial model startup contributes to latency. This is a bounded development test, not a completed production serving architecture. No employee needs a RunPod account or terminal; the application calls the worker through its backend.

Copy `tools/tao_dino_worker.py` to the GPU machine. In that machine's **Bash** terminal, set a randomly generated secret of at least 24 characters locally, then run:

```bash
export WEBSHOP_DINO_TOKEN='<set-a-random-secret-locally>'
python3 tao_dino_worker.py \
  --spec /workspace/inventorylens-dino-eval/experiment.yaml \
  --checkpoint /workspace/inventorylens-dino-eval/model_epoch_011.pth
```

The worker binds only `127.0.0.1:8765`. Forward that port through an authenticated **direct SSH** connection to the worker (`ssh -N -L 8765:127.0.0.1:8765 ...` using its current direct SSH address). Do not expose this development HTTP service publicly. The RunPod proxy SSH gateway may not support forwarding; confirm a direct SSH endpoint before relying on it.

In the local PowerShell terminal that starts Streamlit, set:

```powershell
$env:WEBSHOP_DINO_URL = 'http://127.0.0.1:8765'
$env:WEBSHOP_DINO_TOKEN = '<the-same-secret-set-locally>'
```

Restart the local application if these variables were set after it started. Secrets belong in the process environment, not committed files or chat.

The client checks normalized image SHA-256, upright dimensions, coordinate convention, fresh-inference type, box bounds, finite scores and checkpoint SHA-256. The default expected checkpoint hash is `8bb11d6d63f26edfd480bdf1264467bd5c080f190fbb83a034b252d13ee20d27`; an intentionally promoted model requires explicitly setting `WEBSHOP_DINO_SHA256`.

## Review and inventory

After fresh inference, the screen shows top-three SKU suggestions and unknowns. The reviewer can change SKUs, quantities and box coordinates, exclude merged/false boxes, and add missed items as separate rows. Every counted row must be confirmed. These controls currently use a table; drawing correction boxes with the mouse is not implemented in this screen.

Explicit approval writes a **partial shelf-count snapshot** through the existing authorized service. It does not implement the planned ADD/SUBTRACT ledger. Source photos, catalog digest, worker provenance, capture results and reviewed rows can be downloaded as scan evidence. Repeated approval uses the existing unique approval key. Changing the photo, catalog, user or location invalidates pending scan state.

## Current test status

On 24 October, the local pinned SigLIP2 model loaded and ranked 17 reference views for five products. Each product's reference self-query ranked its own SKU first. This verifies execution and mapping only.

The initial attempt to start `iv9unxo7k3m3gb` was rejected with HTTP 402, insufficient balance. The user then started a new TAO 5.5 RTX 4090 pod, `acvfpsy9gwjecc`, listed at $0.89/hour, and explicitly authorized SSH setup for that test pod. Existing keys and exposed port 22 were used; the HTTP worker binds to loopback and is reached through an authenticated SSH tunnel. Its checkpoint SHA-256 matched the expected epoch-11 checkpoint.

Two actual inference checks completed on historical `IMG_4699.jpeg`. A direct worker-to-matcher check returned six DINO boxes in 14.827 seconds and completed detection plus local SKU matching in 54.79 seconds. Five items received accepted SKU suggestions; the second Jana bottle remained uncertain because of its match margin. These were fresh computations on an existing photo, not archived predictions or blind-test accuracy.

A second check exercised the actual Streamlit scan flow with only file selection automated: real DINO and SigLIP2 ran, six proposals appeared, the item-confirmation boundary was shown, and scan history stayed at zero entries before/after. Cold-start elapsed time was 58.74 seconds. Evidence is held locally at `data/webshop_demo/live_worker_smoke.json`, `live_worker_smoke.png` and `connected_ui_smoke.json`.

The worker-connected local demo runs at http://localhost:8507. `tools/start_webshop_demo.ps1` loads the private connection file automatically when present. The temporary pod was stopped at **2026-10-24 17:00 UTC / 19:00 Europe/Zagreb** by `tools/stop_test_pod_at.py`; the completed stop is recorded in `data/webshop_demo/pod_stop_timer.jsonl`. After shutdown, OpenCV can still check a photo locally, but DINO calls fail until both the worker and its private SSH tunnel are running. The local app and catalog remain available. Restarting the pod incurs its listed GPU charge while running.

On 24 October, nine registered products were added to the isolated demo from the existing second-set catalog, including Freshona beans, Garden Good passata and dmBio curry sauce. The demo workbook and CSV were regenerated with all 14 products; newly added products have no invented price, shelf assignment or opening count. Any pre-expansion scan result must be rerun against the expanded gallery before evaluating these SKUs.

The three-role platform, buyer map/cart, complete spreadsheet catalog import, stock movements and unattended production model serving remain subsequent application work. The current change is the fresh-scan connection and its isolated second-set demo data.

Verification completed: 45 scan/legacy workflow tests and two new scan-screen tests passed (47 total). The new screen tests use explicitly simulated worker results and verify that proposals cannot write stock before item confirmation. The actual local SigLIP2 reference self-check is recorded separately. The local demo launched on port 8507 and its Streamlit health endpoint returned HTTP 200. SKU similarity/margin defaults are illustrative pending validation on second-set photographs; unknown rejection has not been benchmarked.

After the live connection and box-number display adjustment, the nine scan contract tests and two simulated scan-screen tests were rerun: 11 passed. The additional real-inference screen check above is separate from those simulated tests.
