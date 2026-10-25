# WEB&SHOP demonstration

[Watch the demonstration](https://www.loom.com/share/8596593e2eed45c59e453312c88de972) - recorded with Loom and edited with CapCut.

## Demonstrated workflow

1. **Owner:** register a product from reference photographs, enter price, opening stock and shelf, review image quality, and save its SKU and SigLIP2 reference embeddings.
2. **Employee:** analyze a different photograph, inspect DINO boxes and SKU suggestions, correct assignments and quantities, and explicitly approve a stock action.
3. **Inventory and Excel:** inspect the transaction receipt and current stock, then download an updated workbook. Previously downloaded workbooks remain snapshots.
4. **Buyer:** browse the same catalogue, select shopping quantities, inspect totals and budget, and locate products on the 2D shelf map. Buyer previews do not change inventory.

## Architecture

The existing AWS Lambda service runs OpenCV 5 image-quality checks. NVIDIA DINO detection runs on RunPod; SigLIP2 matching runs locally against the current shop's references. The shared database stores product registrations and confirmed inventory transactions.

See [architecture](architecture.md), [evaluation and limitations](evaluation.md), and [build and deployment instructions](deployment.md).

## Live demonstration access

A private live screen-share is the intended judge-access route; scheduling requires organizer confirmation. No public application endpoint is claimed. The operator's localhost address is not a judge-accessible URL.

GPU inference is enabled for the scheduled demonstration rather than kept running continuously. Recorded results remain available in the video.

## Interpretation of results

Detection and matching produce suggestions for human review. Unknowns and manual corrections remain explicit. Stock changes require employee approval. The shelf map shows saved shelf locations, and camera-cart prices are estimates until reviewed.
