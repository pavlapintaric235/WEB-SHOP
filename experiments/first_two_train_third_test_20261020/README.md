# First two sets for training; newest set for evaluation

This is an isolated InventoryLens detector experiment supporting Web&Shop. It does not alter the application, approved labels, source photos, inventory, AWS pilot, or cloud resources.

## Audit completed

`audit_split.py` checked the two historical handoff manifests, all 26 approved grocery image/label SHA-256 pairs, the group split, four archived early originals, and the newest photo manifest. Its output is [`readiness.json`](readiness.json) and [`split_manifest.csv`](split_manifest.csv).

| Cohort | Status |
|---|---|
| First two sets | 100 scene paths resolve to 83 distinct image hashes. Twenty-one additional distinct hashes are registration references, not detector training scenes. |
| Reviewed grocery data | 26 scene images, 163 separately annotated products: 21 images/130 instances for training, 5 images/33 instances for validation. These 26 all occur in the first two sets, and no approved arrangement group crosses training and validation. |
| Other first-two scenes | 57 distinct scene images have no approved per-product instance labels. Four are the early K18/WOW/coconut photographs, despite seven saved source paths because some bytes repeat. The two K18-only views appear to show the same two bottles and must stay in one arrangement group if both are retained. |
| Newest set | 42 owner registration photos are reserved for a confirmed SKU reference gallery. The 25 employee and 5 customer scenes have zero byte-identical overlap with the first-two scene pool. Their physical product boxes were reviewed on 2026-10-23; the ten SKU identities remain provisional. |

The first-two/newest boundary is verified by SHA-256. It does not prove that different files cannot show a similar physical arrangement. The newest scenes have already been inspected during the frozen-model inference experiment, so this is a useful held-out comparison rather than a pristine blind final test. Collect fresh arrangement groups for final confirmation.

## Early product-box review

Four early source photographs were verified against the historical manifest. Local MobileSAM and a retail YOLO checkpoint were tried only as **unapproved annotation drafts**. MobileSAM produced many fragments on the six-product scene (5 suggested, 31 uncertain, 28 rejected components). The retail checkpoint proposed 0, 0, 3, and 0 boxes across the four photographs at confidence 0.20; the three in the mixed scene cover package parts or partial packages. These drafts are unsuitable as training labels.

The user reviewed all four scenes and returned an export. `annotations/early_product_boxes_user_reviewed.json` (local artifact: `annotations/early_product_boxes_user_reviewed.json`) records 14 reviewed products (2, 2, 6, 4). All original SHA-256 hashes, image dimensions, box coordinates, review counts, and duplicate-box checks passed. The two K18 views share one arrangement group. The export is retained separately from source images and historic annotations.

## Training and evaluation gate

1. Import only human-reviewed early boxes after verifying the source SHA-256, image dimensions, nonduplicate boxes, and the reviewed product count. Keep the K18-only near-duplicate pair in one training arrangement group. Leave the 53 other unreviewed first-two scenes out of the initial fit; include them later only after equivalent review.
2. [`prepare_detection_dataset.py`](prepare_detection_dataset.py) creates an ignored, isolated YOLO detection dataset: 25 training images (21 approved grocery images plus 4 reviewed early images) and the unchanged 5 grocery validation images. Grocery polygons become enclosing boxes, and the user's early boxes remain as boxes. Every object has the generic class `product`; this detector does not identify SKU names. The generator verifies source hashes, annotation counts, and split groups. Fine-tune a **generic `product` detector** from the selected pretrained checkpoint only after confirming the training runtime and cost. Keep pretrained checkpoint and frozen DINO predictions as baselines.
3. On the same newest 30 scenes, compare pretrained and trained detector boxes against one human-reviewed box per visible physical item. Report one-to-one detection precision/recall, missed/merged products, and per-photo count error for employee and customer scenes. Do not score predictions as truth.
4. Confirm the ten newest SKU identities and register their owner reference photos. Compare SigLIP2 rankings on the same validated product crops. A generic detector can locate new products; it does not learn ten unseen SKU names from the older sets.

## CPU fine-tune status (2026-10-20)

The generic product detector was fine-tuned locally from the existing Retail YOLO11n checkpoint. No GPU, RunPod compute, AWS resources, production inventory, or customer purchase records were used or changed. The newest 30 scenes were excluded from training and validation.

- Training split: 25 images / 144 boxes; validation split: 5 images / 33 boxes.
- Configuration: CPU, 640 px, batch 2, up to 20 epochs, patience 5, seed `20261020`.
- Early stopping ended at epoch 7; best checkpoint is from epoch 2.
- On the same five-image validation split, pretrained checkpoint mAP50 was `0.161`; fine-tuned best checkpoint mAP50 was `0.531` (precision `0.729`, recall `0.489`). This is preliminary and highly uncertain because the validation set has only five photos.
- Full run settings, validation comparison, and limitations are in [`cpu_training_report_20261020.md`](cpu_training_report_20261020.md).
- Best weights: `runs_cpu/retail_yolo11n_product_finetune_20261020-2/weights/best.pt`.

## Newest-set provisional predictions (2026-10-20)

The user supplied the full newest-set photo folders. All 42 owner references, 25 employee scenes, and 5 customer scenes matched the manifest by filename, file size, and SHA-256. Only the 30 employee/customer scenes were copied into the ignored experiment artifacts and passed to the generic detector; owner references remain reserved for SKU registration.

Both checkpoints ran locally on CPU against the same 30 scenes at 640 px, confidence 0.25, and IoU 0.7. The pretrained baseline proposed 50 boxes (22 scenes with none); the fine-tuned model proposed 167 boxes (10 scenes with none). These are candidate boxes only. The held-out photos were subsequently reviewed and used for the DINO evaluation below; they are no longer a pristine blind test for future iterations.

- Side-by-side review gallery (local artifact: `artifacts/third_set_review_20261020/review_gallery.html`) shows source images and both checkpoint predictions.
- `predictions.json` (local artifact: `artifacts/third_set_review_20261020/predictions.json`) records per-image boxes, confidence, checkpoint-specific class names, settings, and source hashes.
- Local runner: [`run_third_set_review.py`](run_third_set_review.py); gallery builder: [`build_third_set_gallery.py`](build_third_set_gallery.py).

This detector has one generic `product` class; it does not identify any of the ten new SKUs. Human review of scene boxes and confirmation of SKU identities are separate steps.

## Human-reviewed DINO evaluation (2026-10-23)

The user completed one-box-per-visible-item annotations for all 30 employee/customer photos. The export contains 291 item boxes. It was checked against the packaged source photos by SHA-256 and upright dimensions; all boxes are within image bounds. The earlier DINO proposal review marked 206 boxes approve, 69 merged, and 2 reject. “Merged” means a proposal spans multiple products; it is retained as review context and does not replace the separately drawn item boxes.

At the original confidence threshold 0.25, one-to-one matching at IoU 0.50 gives precision 0.745, recall 0.711, and F1 0.728 (207 TP, 71 FP, 84 FN). Employee F1 is 0.743; customer F1 is 0.698. These are provisional human-reviewed localization metrics for these 30 previously inspected scenes. Crowded and partially occluded annotations remain the main review risk. Per-photo counts and matching details are in `dino_ground_truth_eval_20261023.json` (local artifact: `artifacts/third_set_review_20261020/dino_ground_truth_eval_20261023.json`) and `dino_ground_truth_eval_by_photo_20261023.csv` (local artifact: `artifacts/third_set_review_20261020/dino_ground_truth_eval_by_photo_20261023.csv`). Reproduce with [`dino/evaluate_dino_review.py`](dino/evaluate_dino_review.py), which reads the preserved user export `annotations/dino_item_boxes_review_20261023.json` (local artifact: `annotations/dino_item_boxes_review_20261023.json`).

This evaluation does not measure SKU recognition. Before reusing these 30 scenes for another training run, collect a fresh held-out set of different product arrangements for a fair final comparison. The current next experiment can use reviewed item boxes to test reference-photo SKU ranking separately, after the ten candidate SKU identities are confirmed.

## DINO fine-tuning feasibility

The archived NVIDIA Retail DINO artifact is an ONNX inference graph. The local project archive contains no trainable DINO detector checkpoint, and this machine's Python environment is CPU-only (`torch 2.14.1+cpu`, CUDA unavailable) with no TAO Toolkit CLI. NVIDIA TAO DINO fine-tuning expects a PyTorch checkpoint, COCO-format boxes, and a GPU training runtime. The ONNX graph cannot serve as that training checkpoint. NVIDIA's public trainable Retail DINO release `trainable_retail_object_detection_binary_v2.2.2.3` includes `dino_model_epoch=011.pth` (836.84 MB); NVIDIA confirms this checkpoint can be fine-tuned with the FAN base backbone.

The existing reviewed split was converted without changing membership into COCO JSON under ignored `training_data/tao_coco/`: 25 train images / 144 boxes, 5 validation images / 33 boxes, one generic class (`product`, category ID 1). The newest 30 employee/customer scenes remain held out. Converter: [`prepare_tao_dino_coco.py`](prepare_tao_dino_coco.py). The TAO 5.5.0 spec is in [`dino/train.yaml`](dino/train.yaml); [`dino/package_tao_dino.py`](dino/package_tao_dino.py) validates and packages only the reviewed training and validation images, annotations, and spec. The resulting ignored 249 MB archive is `dino/inventorylens_tao_dino_20261021.zip`.

RunPod setup was attempted on 2026-10-21. The saved `nvcr.io` registry credential worked: a TAO 5.5.0 container image downloaded to a secure RTX 4090 Pod in EU-RO-1, and `nvidia-smi` confirmed the GPU. Community 4090 placements were unavailable, so the selected secure rate was $0.89/hour. The Pod is stopped after setup; no training ran, and the local package was not uploaded. A 10 GB network volume could not be created because RunPod requires a $5 balance and the last reported balance was $2.45. The current package also lacks the 836.84 MB pretrained checkpoint `dino_model_epoch=011.pth`, which is a separate NGC download from the container image. The local RunPod CLI is authenticated; the container's receive command still needs a secure transfer path. Resume training only after the pretrained checkpoint and dataset are both present on the Pod. The newest 30 evaluation scenes remain held out.
