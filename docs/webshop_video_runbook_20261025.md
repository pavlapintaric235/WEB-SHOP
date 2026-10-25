# WEB&SHOP: four-minute recording runbook

Use the current application at http://localhost:8509. No hosting, model, threshold or UI changes are needed.

**Latest recording preference:** record the screen only, with both camera and microphone OFF. Add narration/audio later in the chosen cloud editor. The narration column below is an editing script, not a requirement to speak during capture. Audio is excluded from the current test's pass criteria. Submitted test link: https://www.loom.com/share/73fccb42932e4160b77e6a917be976e1 . Automated access could not open this link; playback, privacy and judge access have not been independently verified.

## Recording destination: browser-based cloud archive

**Selected by the user: Loom Chrome extension. Start with the Loom ten-second test below.** YouTube instructions are retained only as an unused alternative. No OBS, local MP4, external drive, video download, or local editing is required. Browser buffering/cache still uses some local storage; this plan avoids saving a completed video on C:. Successful Loom upload and playback still need the ten-second test below. No stream or recording has been started by this preparation.

### Prepare the three roles without recording passwords

Open three separate Chrome tabs at http://localhost:8509. Sign in as the recording owner in tab 1 and employee in tab 2; choose Buyer and the recording shop in tab 3. Each tab has its own Streamlit session. Finish all logins before sharing. Close password-manager popups and disable personal notifications. Keep terminals, AWS/RunPod consoles, connection JSON, this runbook and personal tabs outside the capture. Do not open browser developer tools or Loom Bug Report mode.

Prefer sharing **only the WEB&SHOP Chrome tab**. For YouTube role changes, open the already signed-in next role tab and choose Chrome's **Share this tab instead** when offered, or return to Live Control Room and change the shared tab. Confirm the selected source in the preview. Do not expose a login while changing tabs. A separate Chrome window containing only the three already authenticated demo tabs is the alternative Window capture source.

Do not include native file pickers, local folder paths, scan-folder captions, evidence JSON, or personal metadata in the final video. Tab-only capture excludes other desktop applications, but inspect the test/preview rather than assuming every dialog is hidden. Keep diagnostic paths off the visible portion of the page. The demo shop contains fictional data; do not switch to another shop.

### Unused alternative: YouTube ten-second cloud test

1. Open https://studio.youtube.com in an unshared tab. **Create → Go live → Webcam**. If asked to enable/verify the channel, complete the on-screen steps; first-time activation can take up to 24 hours. If it is not ready now, use Loom below. No encoder or stream key is needed for webcam mode.
2. Choose title **WEB&SHOP — 10-second recording test**, visibility **Unlisted** (not Public), the correct audience setting, and your webcam/microphone. YouTube screen sharing in webcam mode requires both a webcam and microphone connected. If either is unavailable, use Loom screen-only recording.
3. Click **Next**, inspect the thumbnail/preview for private material, then **Go live**. The webcam becomes live first: prepare a neutral background and be ready for that short lead-in.
4. In Live Control Room click **Share screen** next to the microphone control → choose the **WEB&SHOP Chrome tab** → **Share**. Mute any stream playback tab to avoid echo; keep microphone input enabled.
5. On the harmless Products screen, say **“WEB&SHOP cloud recording test, one, two, three.”** Move the pointer and scroll slightly for approximately ten seconds. Do not register a product, run inference, or approve stock in this test.
6. Return to the unshared control room → **End stream** → confirm. Open **YouTube Studio → Content → Live** and wait for the archived video to become playable. Do not download it. The archive includes the webcam lead-in, so its total duration can exceed ten seconds.
7. Play the archive in the browser. PASS requires the app screen, pointer/scroll movement, readable labels, intelligible microphone audio, and no private material. Open its Unlisted watch link in a signed-out/incognito window to verify judge access. Keep this test separate from the final submission video.
8. Send the test link and the checklist results back here. Channel eligibility and successful playback are not verified until that check is complete. I cannot operate your signed-in browser through the tools available in this session.

YouTube documents webcam screen sharing and automatic archiving of short streams: [webcam instructions](https://support.google.com/youtube/answer/9228389?hl=en), [cloud archive](https://support.google.com/youtube/answer/6247592?hl=en).

### Selected recorder: Loom Chrome extension ten-second test

1. Use the official Loom Chrome extension, linked from [Loom's setup instructions](https://support.atlassian.com/loom/docs/get-started-with-the-loom-chrome-extension/). Sign into Loom in an unshared tab. Do not install the desktop app or start a paid trial for this task.
2. In WEB&SHOP, open the extension → capture **Current Tab**, Camera **Off**, Microphone **Off**. Use normal recording, not Bug Report mode. For a multi-role take, a dedicated browser **Window** containing only the three authenticated demo tabs avoids capturing unrelated windows; inspect the preview before recording.
3. **Start recording** → move/scroll for ten seconds without narration → **Finish recording**. Leave the Loom share/upload page open until cloud processing completes. Do not click Download.
4. Play it from the cloud share page. Check readable app text, screen motion, no private material and the absence of unwanted camera/audio capture. Set viewer access to **Anyone with the link**, if the workspace allows it, and test that link while signed out. A workspace-only/login-required link is not judge-ready.
5. Send the link and test result here before the full take. [Loom Starter](https://www.loom.com/pricing) currently permits five minutes per recording, with up to 25 videos; the raw take must fit that limit. Starter editing options are limited, so plan a continuous take and do not assume paid trimming or stitching is available.

After a successful test, create a **new Loom recording** for the four-minute storyboard below. Keep the final cloud video under five minutes including lead-in and genuine processing waits. Do not erase a processing failure to imply success. Loom provides Pause/Resume in the recording controls: for a long real computation, show that processing started, pause only the recording while the app continues, then resume to show its real result and elapsed time. Say “The processing wait was shortened” and preserve actual timings. No local video export is part of this plan. [Loom Pause/Resume](https://support.atlassian.com/loom/docs/pause-and-resume-while-recording/).

## Prepared shop and photographs

- Recording shop: **WEB&SHOP VIDEO - FICTIONAL DEMO**, business ID **9**.
- Owner login: `video-owner` / `<LOCAL_OWNER_PASSWORD>`; employee login: `video-employee` / `<LOCAL_EMPLOYEE_PASSWORD>`. These are local fictional-demo logins. Sign in off-camera and keep this document/password managers out of the capture.
- The shop has the existing 14 baseline products / 50 references and copied shelf positions, but **no lentil SKU and no inventory transactions**. The original shop, products and ledger were preserved. References reuse existing files; no photo collection was duplicated.
- Current single database: `data/webshop_ux_demo/inventorylens.sqlite3`.
- Consistent backup before preparation: `data/webshop_ux_demo/before_video_shop_20261025.sqlite3`. This is a rollback backup, not another writable application database. Do not restore it over a running application.
- Reference files: `<LOCAL_FILES>\folder+\IMG_5029.jpeg`, `IMG_5030.jpeg`, `IMG_5031.jpeg`, `IMG_5032.jpeg`, `IMG_5033.jpeg`.
- Separate employee query: `<LOCAL_FILES>\folder+\IMG_5034.jpeg`.
- All six files exist and have distinct SHA-256 hashes. The query is separate from the registration references. It was used in an earlier rehearsal; do not call it an unseen accuracy test or a newly collected photograph.

The original shop already contains Freshona Lentils with five references. **Use the recording logins above** to avoid duplicate registration. After saving the product once in this recording shop, subsequent takes must show the saved registration or resume with Employee Scan; do not invent a different identity or delete stock history to re-enact registration.

## Pre-recording checklist

1. The ten-second **cloud** test is playable with readable app text and screen motion, has no private content or unwanted camera/audio, and its judge-access link works while signed out. No local recording or downloads are enabled. Add narration later.
2. Application http://localhost:8509 is open in the three prepared role tabs. Close unrelated tabs, consoles, cloud billing pages, personal notifications and password-manager overlays. Share only the app tab or the dedicated demo-only browser window.
3. Current Pod **EXISTING_POD_ID**, RTX 4090, was read as RUNNING at **$0.89/hour**. Its SSH tunnel forwards local `127.0.0.1:8765` to remote loopback `127.0.0.1:8765`. Do not open another tunnel while this one is working. Recovery instructions are in `docs/webshop_fresh_scan_setup.md`.
4. The authenticated worker validation returned its expected `413 invalid_request_size` to an intentionally empty request. This was a connection check, not another model benchmark. The app health endpoint returned 200. No recognition/stock action was repeated for this preparation.
5. Sign into `video-owner`, confirm the sidebar shop name, open Workspace → Employee Scan → Prepare scanner before recording → **Prepare live scanner**. This loads the existing model and the baseline gallery, without running DINO. Keep the app process running. Return to Owner → Products before filming.
6. Keep the five reference filenames and the separate query ready in the file chooser. Do not upload the query as a registration reference. Inspect the actual can to confirm identity and package size.
7. The recording shop starts with no inventory receipts. Register the lentil SKU with opening stock **0**, shelf **A1**. A single approved ADD of one checked can should show **0 → 1** for that SKU at A1.
8. Choose **Upload photo**, never **Archived photo · REPLAY MODE**. The result must say **LIVE**. Leave thresholds unchanged.
9. AWS quality requires a valid existing AWS session. During the recording, show provider **AWS** and its actual decision. If unavailable, stop that segment and resolve access; do not narrate a local result as AWS. Identical reference/query hashes can reuse genuine saved AWS results; do not claim a new Lambda invocation on every click.
10. Do not run the acceptance script or another inference benchmark. Registration and the single recorded employee scan are the demonstration.

## Exact clicks and storyboard — 4:00 edited duration

| Time | UI actions and evidence | Short narration |
|---|---|---|
| 0:00–0:12 | Show team/member identity briefly, then the WEB&SHOP app and recording-shop sidebar. Logins happen off-camera. | “I’m presenting WEB&SHOP: owners register products, employees review visual counts, and buyers use the same catalogue.” |
| 0:12–1:15 | Owner → **+ Add Product**. Enter the fields below → **Continue to photographs**. Upload IMG_5029–5033 only. Set reference views front / side / back / side / angled after inspecting each image. Show local quality feedback; on the front image click **Analyze this photo on AWS**. Show provider/decision, then check **I checked the image, framing and product identity** for each usable photo. **Review registration summary** → check **I approve this identity, references, price, shelf and initial stock** → **Register Product**. Hold on generated SKU and “References embedded and available…” success. | “Five views describe the same physical product. OpenCV checks blur, exposure and resolution; the existing AWS Lambda supplies the cloud quality decision. Registration saves its SKU and SigLIP2 reference embeddings without retraining.” |
| 1:15–2:05 | Share the already signed-in employee tab. Confirm the same shop. If opened before registration, click Workspace → Buyer → Employee Scan to refresh the catalogue without signing out. Employee Scan → **Shelf / count location: A1** → **Upload photo** → upload **IMG_5034.jpeg**. **Capture check: AWS capture service** → **Check this photo** → show decision → **Find products in this photo**. Wait for real processing. Hold on **LIVE**, actual boxes, proposal count and elapsed time. | “This is a different photograph. DINO runs on the RunPod GPU. SigLIP2 then compares the detected crops with this shop’s references. These are suggestions, including unknowns, for a person to review.” |
| 2:05–2:50 | Inspect the front lentil can and actual suggestions. Expand **Inspect / correct [product name · SKU] boxes** to show candidates; retain an unknown if identity is uncertain. For a bounded receipt of one checked lentil can, narrow the draft to that SKU and one verified unit. Remove the partial-can proposal from this transaction; other real products are outside this transaction scope, not automatically “false detections.” Use **−**, **+**, or **Set quantity** to demonstrate a genuine correction, then **Mark group reviewed**. If it was missed, **Add missed product** → select the registered lentil SKU → **Missed units: 1** → **Add missed product**; disclose this manual correction. Confirm or exclude every remaining row. **What should happen next? → Receiving / ADD**. Enter the transaction note below. Show before/after **0 → 1**. Check **I checked the scope and before/after stock; apply this inventory action** → **Approve inventory action** once. Hold on saved receipt and locked draft. | “I check the proposal and correct the quantity. This receipt covers one visually checked can. Stock changes only after I inspect the before-and-after preview and explicitly approve ADD. The saved draft is locked against duplicate posting.” |
| 2:50–3:30 | Share the prepared Buyer tab → **Shop: WEB&SHOP VIDEO - FICTIONAL DEMO**. If opened before registration, switch **Shopping list & map → Products** to refresh. **Search the catalogue: Freshona Lentils** → **Shopping quantity: 2**. Show **€2.98** subtotal. Enable **Set a budget**, enter **€5.00**, show **€2.02** remaining. Open **Shopping list & map**; show A1 highlighted and the list/total together. | “The buyer sees the same product, its €1.49 demo price and shelf A1. Two cans total €2.98, leaving €2.02 from a five-euro budget. The map uses owner-supplied shelf positions. Shopping-list changes do not reduce inventory.” |
| 3:30–3:50 | Show a simple architecture caption while retaining app evidence: **AWS Lambda / OpenCV 5 quality → RunPod DINO → local SigLIP2 → human review → shared catalogue/ledger**. Do not show connection files or keys. | “AWS performs the OpenCV quality analysis. RunPod supplies GPU detection; this release runs SigLIP2 matching locally. Owner, employee and buyer share one database.” |
| 3:50–4:00 | Show receipt, registered product and map as a short closing montage. | “Crowded and partial items still need human review. This demonstrates one reproducible workflow, not a general accuracy claim.” |

**Owner fields:** Product name **Freshona Lentils 400 g**; Brand **Freshona**; Variant / flavor **Lentils**; Package size **400 g**; Quantity unit **can**; Price in EUR **1.49**; Initial stock quantity **0**; Shelf / location **A1**; SKU **leave blank to generate a stable ID**; Barcode / EAN **blank unless checked on the physical package**; Description **Fictional video demo price and opening balance.** Verify the physical can before accepting these identity details.

**Transaction note:** `Video demo: ADD one visually checked lentil can at A1. Other photographed products are outside this receipt scope.`

The existing UI has **one** “Find products in this photo” button: it runs DINO followed by SigLIP2. There is no separate matching button to click. Do not add one for the video. Candidate names and unknowns are inside the per-box review expander; show actual results rather than promised labels.

The owner wizard uses original/upright reference images and has no automatic-mask approval step. Existing optional legacy masks are retained separately. Skip a mask segment; do not imply that this wizard generated masks.

For processing waits, narrate while the real computation runs. If you later shorten a wait using available cloud editing, disclose **“processing wait shortened; real inference”** in the video or description and retain the UI’s actual elapsed time. Do not assume a cloud editor can accelerate arbitrary sections. Previous timing is context, not a promised result for this take. Do not cut around a failure to imply success. If AWS or recognition fails, show the actual state and disclose a manual correction or retry. Keep the raw take below Loom's five-minute limit when using Starter.

## Save the evidence and give judges access

- Preserve the cloud recording and its watch/share URL. Play the **archived** video after stopping the stream/recorder; a live preview is not proof of a saved archive. Test access while signed out. Keep the original raw cloud take if making a cloud-trimmed version. Do not download video/audio files to C:.
- Visual evidence to retain in that cloud video: registration summary/success; AWS provider/decision; LIVE proposals with actual timing; correction; stock preview and receipt; buyer €2.98 total/budget; A1 map. Existing screenshots are available under `docs/webshop_8509_*.png`, but they belong to the earlier rehearsal, not this new recording shop.
- Automatic scan evidence stays under `data/webshop_ux_demo/fresh_scans/9/<scan_id>/`; registration originals/upright images/embeddings stay under `data/webshop_ux_demo/registered_references/9/`. No application storage migration or cloud backup was performed. Preserve those local files and the consistent preparation backup. A later complete backup must include all pre-existing image paths referenced by the database; copying only the new shop folder omits the reused baseline references. Do not display raw evidence JSON or private filesystem paths in the recording.
- Keep one writable app database. Stop app writes before the final backup/cutover or use SQLite’s online backup API; do not copy a live database as an ordinary file and assume consistency.
- Judge instructions: start with the video and supplied architecture/evidence; use a scheduled live screen-share for the localhost app. `localhost:8509` is only reachable on this laptop; do not publish it as a judge endpoint. For a live screen-share, keep the app, tunnel and GPU running, sign in off-camera, and use Review only unless another fictional stock action is explicitly intended.
- If a second take occurs after approval, show the existing receipt or narrate a new explicit transaction with its real starting balance. Do not present another ADD as the original 0 → 1 action.

## Safe shutdown, only after recording and evidence are safe

1. Verify playback of the completed cloud archive (not just live preview), judge access, and preservation of local scan/registration evidence and a consistent database backup. Preserve the archive URL. No Pod shutdown should occur before this is done.
2. In RunPod, verify the exact current Pod is **EXISTING_POD_ID**, then choose **Stop**, not Terminate. No stop has been performed by this preparation. Stop releases compute; retained disks, if any, can still bill.
3. This Pod has no persistent mount. Its worker/checkpoint copies on `/workspace` are temporary and can be lost when the container stops or resets. The local epoch-11 checkpoint/spec/worker and verified restoration instructions are the recovery sources; retain them. Do not assume the current remote worker survives shutdown.
4. Close the private SSH tunnel after the Pod is stopped. The local catalogue/buyer app can remain available; LIVE inference will be unavailable until a worker is restored.

## Preparation result

Application health: PASS. Authenticated worker connectivity: PASS. Active Pod: RUNNING. Reference/query file existence and distinct hashes: PASS. Separate fictional recording shop: PREPARED. Original database rows/stock history preserved: PASS. New registration/inference/stock action for this recording: **not executed; reserved for the take**. Video destination: **Loom cloud, selected by user**. Ten-second recording/upload/playback test: **pending user browser steps**. No OBS, local MP4, benchmark, model download, training, deployment, UI change or Pod stop was performed.
