# InventoryLens Segmentation Annotation Workflow

This workflow creates drafts with the existing MobileSAM checkpoint, then requires you to review every image and polygon in CVAT. Drafts never enter `approved/` automatically. The detector class is always the generic class `product`; the current ResNet18/SIFT SKU matcher is a separate stage and is not used here.

## Workspace

| Folder | What belongs there |
|---|---|
| `data/segmentation_dataset/input_images/` | New, original grocery scene photographs. Copy them here; do not use screenshots or annotated previews. |
| `data/segmentation_dataset/proposals/<batch>/` | Normalized photo copies, clean SAM draft labels, metadata and a CVAT import ZIP. |
| `data/segmentation_dataset/previews/<batch>/` | Numbered preview: green is an unflagged suggestion; amber needs extra scrutiny. Rejected red masks are saved separately so broad background regions do not obscure the preview. |
| `data/segmentation_dataset/uncertain/<batch>/` | Binary masks for proposals that need extra review; these are not included in the draft YOLO labels. |
| `data/segmentation_dataset/rejected/<batch>/` | Binary masks for obvious background/scene or too-small proposals. |
| `data/segmentation_dataset/approved/images/` and `labels/` | Only manually reviewed CVAT results. The three approved seed photos are already here. |
| `data/segmentation_dataset/exports/` | YOLO Segmentation exports made only from approved labels. |
| `data/segmentation_dataset/reports/` | Validation summaries and initialization report. |

The first setup already copied the approved revision 2 CVAT batch here without changing its source ZIP. All three seed images share one `group_id`; treat them as development/training examples, never independent test evidence.

## Propose Masks

1. Copy new original scene photos into `data/segmentation_dataset/input_images/`. Keep filenames unique, and leave the original files there.
2. Open PowerShell in `C:\Users\U938\Downloads\inventorylens_starter` and run:

   ```powershell
   .\.venv\Scripts\python.exe tools\segmentation_dataset.py propose
   ```

3. Open the printed preview folder and `proposal_report.json`. Green means MobileSAM produced a candidate with no simple heuristic warning; it is **not** an approval. Amber indicates a possible edge crop, overlap, duplicate, small fragment, or other concern. Masks covering less than 1.5% of the working image are kept for review, not put into draft labels. Rejected masks are listed in the report and saved separately under `rejected/`.
4. The `cvat_import.zip` includes all input photos and only green, unflagged candidates. Amber candidates are intentionally omitted so uncertain masks are not silently promoted; inspect their PNGs and add a real product polygon manually if needed. A green SAM mask can still be cardboard, a shadow or background, and a product can be missed entirely.

Each new run writes to a timestamped batch. It refuses to overwrite an existing batch. The image bytes in `input_images/` are never edited; proposal images are orientation-normalized PNG copies used to keep labels and CVAT frames aligned.

## Review In CVAT

The import ZIP follows the Ultralytics YOLO Segmentation layout used by CVAT: `data.yaml`, `train.txt`, `images/train/`, and `labels/train/`. CVAT accepts this layout and also accepts `train/images/` and `train/labels/` layouts. See the [CVAT Ultralytics YOLO format guide](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/) and [CVAT dataset import guide](https://docs.cvat.ai/docs/dataset_management/import-datasets/).

1. In CVAT, create a task for the photo batch. Use exactly one label named `product`. Use the photos with their original batch filenames; do not rename or edit them.
2. Upload the batch's `cvat_import.zip` as Ultralytics YOLO Segmentation annotations for that task. If you create the task directly from a dataset ZIP, use the same format and check that the task contains the expected images.
3. Review every preview against the original photo. For each visible physical product, keep one separate polygon, even if its SKU is not registered. Remove background, shelf, shadow, text-only, cap-only, duplicate and multi-product masks. Add missed products. For partial occlusion, trace only the visible product boundary; do not invent a hidden outline. Do not annotate products hidden completely behind other objects.
4. Check repeated units carefully: two physical packages must be two polygons even when they have the same SKU. Use the CVAT polygon tool to adjust corners and edges at full zoom.
5. Export the **task dataset**, not just a screenshot, as `Ultralytics YOLO Segmentation 1.0` with images included. Keep the image names unchanged. Save the downloaded ZIP outside the approved folder, for example in `data/segmentation_dataset/exports/cvat_returns/`.

## Import Reviewed Labels

Only run this after you personally checked all images, visible products, and outlines in CVAT. Replace `<batch-id>` with the exact timestamp folder printed by `propose`:

```powershell
.\.venv\Scripts\python.exe tools\segmentation_dataset.py import-cvat "PATH_TO_CVAT_EXPORT.zip" --batch-id <batch-id> --reviewed
```

The importer verifies that CVAT exported every batch image with unchanged decoded pixels and matching filenames, that labels map unambiguously to those photos, and that every polygon is class 0, normalized and geometrically simple. If an image has no label file, the importer stops instead of assuming it is empty. Only after you visually confirm that it truly contains no products, add `--negative "filename.png"` for that image. Empty `.txt` files are valid explicit negatives.

The import refuses to overwrite an approved image or annotation, import a byte-identical duplicate photo, or accept changed/ambiguous image mapping. It copies the batch's normalized source photos and the reviewed labels into `approved/`, and records the review and `group_id` in `approved/review_manifest.json`.

## Validate And Export

```powershell
.\.venv\Scripts\python.exe tools\segmentation_dataset.py validate
.\.venv\Scripts\python.exe tools\segmentation_dataset.py export
```

Validation checks image/label pairs, class IDs, finite normalized coordinates, polygon size and self-intersections, duplicate files/rows, empty negative-image confirmation, review-manifest counts, and suspiciously overlapping boxes. Box-overlap flags are only a prompt to inspect masks; touching or occluded products can legitimately overlap. A script cannot prove that every visible package was annotated, so manual completeness review remains mandatory.

The export is a training-format Ultralytics YOLO Segmentation dataset with one class. It contains only approved images and labels. At this stage all images are in a `train` subset; this command does **not** train anything and does not create a validation/test split. Near-identical photos from one physical layout share a `group_id`; keep every group together when we later create splits. Reserve separate, untouched scene groups for the eventual test set.

## One-Photo Smoke Test

To exercise MobileSAM without feeding an approved photo into the normal proposal queue, use a separate output directory:

```powershell
.\.venv\Scripts\python.exe tools\segmentation_dataset.py propose --image data\cvat_fixed\revision_02\images\train\image1.png --batch-id smoke_test --output-dir data\segmentation_dataset\smoke_test
```

This tests code and model execution only. The resulting masks are not evidence of generalization and are not approved annotations. The original photo and the approved CVAT files remain unchanged.

## Limitations

- MobileSAM is class-agnostic and can propose packaging fragments, shadows, shelves or background. It can also miss a product or merge touching units.
- Heuristic flags are conservative triage, not semantic product detection and not calibrated confidence.
- YOLO polygons trace the model mask at MobileSAM's working resolution (longest side 1280), so small contour details may be approximate.
- The first three approved scenes are only a seed. Do not train or claim broad grocery generalization from them.
