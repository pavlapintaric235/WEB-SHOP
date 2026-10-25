# WEB&SHOP video narration

Recorded with **Loom**, edited with **CapCut**. [Final published video](https://www.loom.com/share/8596593e2eed45c59e453312c88de972).

Timing draft for the earlier 03:33 cut (`B0A9FD0E-E2D0-4804-99A4-D8721F5D220F.mov`). The latest reviewed local cut is approximately 03:05; timestamps below are not exact timings for that final edit. Read at approximately 120?130 words per minute, allowing short pauses.

**00:00–00:12 — Introduction**

This is WEB&SHOP, a visual inventory application connecting shop owners, employees and buyers through one shared catalogue. This demonstration uses fictional shop data.

**00:12–00:35 — Registering a new product**

The owner starts by adding a new product: Erbsen Eintopf. I upload four reference photographs and enter its product details, price, opening quantity and shelf. The quality feedback helps me review the photographs before approving them for registration.

**00:35–00:48 — Registration and Excel**

The product receives a unique SKU. SigLIP2 reference embeddings are saved without SKU-specific retraining. The registered product appears in the shared catalogue and the current Excel workbook.

**00:48–01:16 — Capture check and detection**

Next, I switch to the employee workspace and upload a different shelf photograph. The existing AWS Lambda service runs OpenCV 5 to check image quality. NVIDIA DINO runs on the RunPod GPU to propose product boxes. SigLIP2 matching runs locally, comparing those crops with this shop’s registered references, including the new product.

**01:16–02:04 — Human review and corrections**

These results are suggestions that need review. The interface shows boxes, candidate SKUs and unknown detections. I inspect the proposals, correct product assignments and quantities, and remove unwanted detections. Repeated instances are grouped under the same SKU. The new product is included in this review alongside the existing catalogue products. This step matters because overlapping boxes and uncertain matches can produce incorrect counts. The employee checks the final quantities before any inventory action is approved.

**02:04–02:33 — Stock approval and updated sheets**

I select subtraction and use each product’s assigned shelf. The preview shows the stock before and after the proposed action. Removing two units of the new product changes its balance from ten to eight. After explicit approval, the current stock updates in the catalogue and Excel export. Opening quantities remain unchanged, and the transaction is recorded in history.

**02:33–03:07 — Buyer catalogue and shelf map**

The owner can inspect the transaction history. Buyers use the same shop catalogue, with product prices and shelf locations. I choose shopping quantities and build a list. The two-dimensional map highlights the shelves containing those products. It provides shelf guidance using the owner’s saved layout.

**03:07–03:33 — Camera cart and closing**

The camera-assisted cart also accepts a photograph and produces product suggestions with a price estimate. Here, the displayed subtotal is eleven euros and fifty-six cents. Unreviewed suggestions still need confirmation, and buyer previews do not change inventory. WEB&SHOP brings product registration, reviewed stock actions and shopping assistance into one workflow.

## Editing notes

- Suggested on-screen disclosure: “Edited for length; processing waits shortened.”
- Crop the browser tabs and recorder controls from the final presentation.
- The final cart is an unreviewed estimate. Do not describe it as a completed checkout or fully confirmed recognition result.
- The map shows shelf locations, not a verified shortest walking route.
