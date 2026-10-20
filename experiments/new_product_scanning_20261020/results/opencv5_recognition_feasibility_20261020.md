# OpenCV 5 recognition feasibility on the newest products

Local isolated check, 2026-10-20. Run [`evaluate_opencv5_recognition.py`](../evaluate_opencv5_recognition.py) to reproduce from the hashed owner photos and approved reference crops. No model download, GPU, AWS action, inventory update, or source-image edit was performed.

## What ran

| OpenCV 5 method | Input | Result | Interpretation |
|---|---|---:|---|
| `barcode_BarcodeDetector.detectAndDecodeWithType` | 42 owner registration photos, EXIF-corrected and resized to at most 1800 px | 3 photos with a decoded EAN-8 or EAN-13 | A useful exact-code path **when the barcode is visible and readable**. This is not 3/42 product-identification accuracy: most photos were not intentionally framed as barcode scans. Decoded values still require owner confirmation against the sellable SKU registry. |
| ORB + BF Hamming, 2-nearest ratio < 0.75; maximum good-match count per provisional SKU | Same 34 approved owner photos as the SigLIP2 paired run; leave query photo out of gallery | Box crops 8/34 top-1; masked crops 8/34 top-1 | This simple printed-feature matcher is not competitive as a general replacement on these multi-view references. It has no geometric verification or calibration. |
| Existing SigLIP2 paired owner retrieval, shown for cohort context | Same 34 owner photos and provisional groups | Box crops 24/34; masked crops 25/34 top-1 | Stronger in this controlled reference test; still not employee/customer scene accuracy. |

The [barcode CSV](opencv5_barcode_owner_20261020.csv), [per-query ORB CSV](opencv5_orb_owner_20261020.csv), and [machine-readable run record](opencv5_recognition_feasibility_20261020.json) retain the exact outputs and method. All ten identities remain provisional. ORB and SigLIP2 numbers are owner-photo retrieval results, not a detector comparison or a final model leaderboard.

This ORB check is distinct from the existing local application's [OpenCV SIFT/RANSAC printed-label verifier](../../../local_features.py), which is applied selectively after ambiguous MobileSAM/ResNet18 matches. The 8/34 ORB result does not measure that verifier or the planned DINO/SigLIP2 pipeline.

## Recommended role for OpenCV 5 in Web&Shop

**Photo-based product counting is the primary workflow.** The optional barcode path should help resolve identity; it must not create an extra counted unit.

1. **Owner SKU registration:** confirm each sellable SKU and its product reference photos. Keep the photo gallery as the primary visual reference. An optional OpenCV barcode capture can attach a decoded EAN/UPC only after the owner confirms which registered SKU it identifies.
2. **Employee/customer photo counting:** OpenCV 5 checks the full photo, prepares display/crops, and can give item-specific capture feedback. The current experimental detector candidate, DINO, proposes one box per physical item. SigLIP2 compares each valid single-product crop with the confirmed registry gallery. Counts are grouped by registered SKU for review; unknown, missed, and merged products stay editable.
3. **Optional barcode check:** if a person chooses to scan a code on an uncertain product, OpenCV can decode it and look up a previously confirmed SKU mapping. This resolves the identity of an existing proposed/manual item. It never increments the picture count a second time and never silently overwrites a conflicting visual result.
4. **Inventory:** the planned Web&Shop behavior is an approved add/subtract movement, with an explicit action and location. A customer photo count could help review a cart, but should not itself subtract stock; a finalized sale needs its own authorized transaction path. A shelf audit is an observed count and needs a separately approved adjustment, rather than adding every visible item. The current local application uses reviewed **snapshot replacement**, not a movement ledger. The new OpenCV/DINO/SigLIP2 experiment is not wired into either stock path and writes no inventory data.
5. **Detector failures:** study missed and merged instances using verified scene labels. OpenCV contours or circle/edge evidence may flag a suspicious multi-item box for review, but must be measured before becoming a splitter or counter. Reference masks do not repair DINO boxes.

OpenCV's DNN module can execute supported ONNX models, including detection models, but that changes the **runtime**, not the learned detector's accuracy by itself. Registry photos alone do not make a robust multi-product detector. OpenCV 5's new DNN engine is CPU-only in 5.0; the existing DINO path is a verified TensorRT GPU path. A move to OpenCV DNN would need compatibility, latency, and accuracy checks before it could replace that path. Item-specific photo guidance and optional barcode verification give OpenCV visible, testable decisions within the picture-first workflow.

For a competition-quality claim, first import the prepared 30-scene DINO/SigLIP2 outputs, fix the duplicate/incomplete scene labels, and score detected instances, SKU suggestions, count error, and review corrections on an arrangement-held-out set. Then test item-specific OpenCV quality guidance for whether it improves recapture and correction outcomes. Test optional barcode verification separately on deliberately captured codes with human-verified SKU mappings, reporting correct identification, wrong identification, abstention, and latency. Do not count barcode reads as extra products.

Primary references: [OpenCV 5 barcode detector](https://docs.opencv.org/5.0/tutorials/objdetect/barcode_detect_and_decode/barcode_detect_and_decode.html), [OpenCV 5 descriptor matching](https://docs.opencv.org/5.0/main_modules/features_match.html), [OpenCV 5 DNN engine selection](https://docs.opencv.org/doc/doxygen/html/de/db7/api_dnn_engine_selection.html), and the [OpenCV AI Competition 2026 judging rules](https://opencv26.devpost.com/rules).
