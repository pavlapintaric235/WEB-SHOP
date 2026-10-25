# WEB&SHOP recording and private demonstration

## Completed recording

**Recorded with Loom; edited with CapCut.** Screen capture used camera and microphone off; narration/audio can be added during editing. CapCut was the editing tool, not an unspecified cloud editor.

**Final video:** [Watch WEB&SHOP](https://www.loom.com/share/8596593e2eed45c59e453312c88de972).

The latest reviewed local cut is approximately **3 minutes 5 seconds**. The final Loom link was supplied by the entrant. Signed-out playback of that link, final audio, real team introduction and public product-photo permission have not been independently verified. The earlier ten-second test link is superseded; another test recording is not required to correct this documentation.

The narration in [the script](webshop_video_narration_20261025.md) was timed to the earlier 3:33 cut. Its timestamps are not an exact timing guide for the final edit.

## What the demonstration shows

1. **Owner:** register Erbsen Eintopf with four reference photographs, product details, price, opening quantity and shelf. Save its unique SKU and SigLIP2 reference embeddings without SKU-specific retraining.
2. **Employee:** upload a different scene photograph; show the AWS OpenCV 5 capture check, actual DINO boxes and SigLIP2 suggestions. Review assignments, unknowns and repeated detections; correct quantities before approval.
3. **Stock and Excel:** preview subtraction, explicitly approve it, then show the saved receipt and updated current-stock export. The recorded new-product example changes ten units to eight; opening stock remains ten. Downloaded workbooks are snapshots and must be downloaded again after an action.
4. **Buyer:** open the same shop, inspect product prices and shelves, choose quantities, show shopping totals/budget and the simple 2D shelf map. The camera-cart subtotal of EUR 11.56 is an unreviewed estimate, not a completed checkout. Buyer previews do not write inventory.

AWS Lambda runs the existing OpenCV 5 image-quality service. RunPod runs NVIDIA DINO detection. This release runs SigLIP2 matching locally against the selected shop's references. One shared database stores catalogue and confirmed stock history.

## Editing and privacy checks

- Keep the final video below five minutes. Disclose shortened processing waits and retain actual inference timings. Show manual corrections honestly.
- Review the final CapCut edit and published Loom playback for readable UI, synchronized narration/captions and a clear ending.
- Crop browser tabs, recorder controls and personal paths. Keep passwords, API keys, SSH details, cloud consoles and private metadata off screen.
- Preserve the original Loom take and CapCut project/final edit. Do not create additional large video copies on the nearly full C: drive.
- A supplied link is not proof of signed-out judge access. Test the final Loom link in an incognito/signed-out browser.

## Chosen judge-access route: private live screen-share

The entrant selected a **private live screen-share**, rather than completing public HTTPS hosting. It is **not yet scheduled or confirmed by the organizers**. See the [organizer request](demo.md#private-live-screen-share-request). The official competition page permits an arranged live screen-share: [submission requirements](https://opencv26.devpost.com/).

The operator launches http://localhost:8509 locally. Judges attend the agreed meeting; localhost is not a public judge URL. Confirm the meeting platform, date/time, organizer acknowledgment and judge attendance before marking access complete.

Before the session:

1. Confirm the existing RunPod worker and private SSH tunnel are connected; do not change a working configuration.
2. Sign into owner and employee off camera; prepare Buyer in the same fictional shop. Share only the app window/tab.
3. Confirm the shop's reference gallery and actual LIVE status. Rehearsal/replayed results must remain labeled accurately.
4. Use the saved registration and receipt for a repeat demonstration; do not reset inventory or repeat a stock action merely to reproduce the original quantities.
5. Use Review only unless a further fictional stock transaction is explicitly intended. A live demonstration does not itself approve public publication of product photographs.

## Evidence, backup and shutdown

Keep the single authoritative application database and all referenced photographs, embeddings and scan evidence. Use SQLite's online backup API or stop app writes before making a consistent database backup. Do not overwrite the working database or start another writable copy.

Keep the existing Pod and tunnel available until the video/evidence are safely preserved and the live-session plan is agreed. Do not stop or terminate it automatically. Temporary Pod storage is not a durable backup.

When shutdown is explicitly authorized, verify the actual active Pod, preserve any required remote-only files, and choose **Stop**, not Terminate. Temporary files may be lost; retained storage may still bill. Close the SSH tunnel afterward. The local catalogue can remain accessible, but LIVE detection needs the GPU worker restored.
