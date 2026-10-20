# Web&Shop GPU Inference Package (RTX 3090)

This is a reproducible, read-only inference experiment. It does not train a model, write to SQLite, alter registered products, or update the production recognizer.

## What Is In The Upload ZIP

- All 41 unique photos from `InventoryLens_Codex_experiment_READY.zip`: 25 scene photos and 16 separately supplied reference photos. The byte-identical duplicate `IMG_4711 (1).jpeg` is excluded and recorded in the source metadata.
- Scene groupings, draft counts, review notes, and the matching reviewed workbook.
- A read-only copy of available registered catalogue references and the matching precomputed MobileSAM reference crops. The SQLite database is not included.
- Local MobileSAM, ResNet18 support assets, MobileNetV2, and the SKU-110K-trained YOLO11n detector checkpoint.
- A pinned SigLIP2 download script and a separate optional COCO-pretrained YOLO11n-seg experiment.
- CUDA preflight, GPU reference-crop preparation, inference, CSV/JSON/overlay output, and a final download-archive script.

The SKU-to-catalogue crosswalk is deliberately explicit. Some ILX IDs map to catalogue SKUs only as candidates because variant/size identity is unresolved. Croco salt is not matched to the catalogue's Croco Pizza SKU. Candidate mappings are surfaced in outputs and are not treated as verified catalogue registrations.

## Ground Truth And Evaluation Limits

The input package contains no per-instance boxes or segmentation polygons. Therefore this package cannot calculate detection precision/recall, object-level duplicate rates, or mask IoU. It scores only user-confirmed scene totals and explicit `human_verified_YES_NO=YES` SKU rows from `scene_counts_DRAFT.csv`. All other proposed counts are carried as provisional context, never scored.

For a multi-view group, the first photo listed in the verified manifest is the scoring view. The script also reports each view and a max-per-SKU evidence summary, but never adds view counts together. That max summary is not a fused inventory count. Unregistered-product rejection accuracy cannot be measured without object-level labels for unregistered products.

## Start A RunPod Session

1. In RunPod, create a GPU Pod and select an **RTX 3090 (24 GB)** when it is currently available. Choose an official CUDA-enabled PyTorch template with JupyterLab enabled. Capacity, template labels, and prices change, so confirm the live GPU and price shown in RunPod before deploying. Do not use a CPU-only template. See RunPod's [GPU reference](https://docs.runpod.io/references/gpu-types), [Pod overview](https://docs.runpod.io/pods/overview), and [storage guide](https://docs.runpod.io/storage/network-volumes) for current availability and persistence behavior.
2. Allocate at least **20 GB of container disk** for the download, SigLIP2 weights, intermediate caches, and the final results/model ZIP. A persistent network volume is optional. If you use a Pod without persistent storage, download the final ZIP before stopping or terminating it.
3. Open JupyterLab and upload `inventorylens_runpod_inference.zip` from your computer using the Jupyter file browser.
4. In a JupyterLab Terminal, extract the upload and enter the package directory. Adjust the uploaded filename/path if Jupyter placed it elsewhere:

```bash
mkdir -p /workspace/webshop_runpod
python -m zipfile -e /workspace/inventorylens_runpod_inference.zip /workspace/webshop_runpod
cd /workspace/webshop_runpod
```

5. Verify the GPU before installing anything. This check performs a real matrix multiplication on CUDA and reports the GPU name:

```bash
python verify_cuda.py
```

It must report `cuda_available: true`, `matrix_test: passed`, and an RTX 3090 device. If it fails, stop here; do not continue on CPU.

6. Install the experiment dependencies while preserving the template's existing CUDA PyTorch build. `torch` and `torchvision` are intentionally absent from the requirements file:

```bash
python -m pip install -r requirements-runpod.txt
python verify_cuda.py
```

If the second check fails or PyTorch reports a CPU build, stop and select a compatible official CUDA PyTorch template. Do not install a generic CPU `torch` wheel over the RunPod image.

7. Download the pinned SigLIP2 checkpoint and the optional, separate pretrained YOLO11n-seg comparison checkpoint:

```bash
python prepare_models.py --with-yolo11-seg
```

SigLIP2 Base is pinned to revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. Expect roughly 1.5 GB for its weights plus cache overhead. YOLO11n-seg is a distinct COCO-pretrained comparison; it does not replace SKU-110K YOLO + MobileSAM.

8. Prepare the reference crops on the GPU. Existing registered crops are reused read-only; supplied experiment reference photos without a cache are processed with MobileSAM on CUDA. A fallback crop method, if needed, is recorded in `models/reference_cache_manifest.json`:

```bash
python prepare_reference_cache.py
```

9. Run the baseline and optional comparison. The three SKU matcher arms (MobileNetV2, SigLIP2, and their equal-weight fusion) use the existing calibration-only cutoffs. No thresholds are changed:

```bash
python run_inference.py --include-yolo11-seg
```

The baseline is SKU-110K YOLO11n boxes followed by MobileSAM masks. The second detector is separately reported as COCO-pretrained YOLO11n-seg. Both use the same reference set, matcher, and scoring rules. Outputs appear under `results/`.

To run again without overwriting prior output, choose a new directory:

```bash
python run_inference.py --include-yolo11-seg --output results_run2
```

## Review The Results

- `results/report.md`: concise metrics and limitations.
- `results/results.json`: full per-image detections, polygons, top-five scores, stage timings, multi-view notes, and summary metrics.
- `results/predictions.csv`: one row per proposal and matcher arm; mask polygons are pixel coordinates in JSON-encoded columns.
- `results/overlays/`: colored masks with numbered SKU/unknown labels.
- `results/masks/`: one grayscale mask image per proposal.
- `results/manual_review.csv`: unknowns, candidate-only catalogue mappings, and near-identical proposal pairs. Near-identical boxes are flagged, never automatically merged.
- `results/scene_metrics.csv`: confirmed count evaluation; unverified rows remain unscored.

Detection-stage timings are synchronized CUDA timings and are separated from MobileSAM, MobileNetV2, SigLIP2, CPU crop/matching work, one-time reference setup, and end-to-end photo time. Timings include this Pod's current hardware/software; do not compare them to CPU timings or another GPU as if hardware were equal.

## Download Before Stopping A Non-Persistent Pod

1. Package reports, images, inputs, and all model artifacts (including the SigLIP2 cache):

```bash
python package_downloads.py
```

This creates `inventorylens_runpod_results_and_models.zip` in the package folder and verifies its CRC. It can be large (around 1.5 GB or more). Check that the Jupyter filesystem has enough free space before packaging.

2. In JupyterLab's file browser, navigate to `/workspace/webshop_runpod/`, right-click `inventorylens_runpod_results_and_models.zip`, and select **Download**. Wait until the browser download has fully completed and verify the file is present on your computer.
3. Only after the download finishes, return to RunPod and **Stop** or **Terminate** the Pod. A stopped Pod without persistent storage does not preserve its container disk. Terminating also removes Pod storage; a separate network volume persists and may continue to incur storage charges until deleted.

If the browser cannot handle the large archive, keep the Pod running and use JupyterLab's file browser to download the smaller `results/` directory artifacts first, or create split archive parts with a standard `split -b 512M inventorylens_runpod_results_and_models.zip ...` command and download each part. Do not terminate until every required part has finished downloading.

## Model And Data Caveats

- The included SKU-110K checkpoint is a third-party retail-trained checkpoint with restrictive/unclear upstream data and model terms. Treat it as research-only; it is not cleared for proprietary commercial deployment.
- Ultralytics code and YOLO11 weights are AGPL-3.0 by default unless a commercial license applies. The optional YOLO11n-seg arm is for this experiment only; Ultralytics currently describes a commercial license for closed-source and SaaS use in its [license guide](https://www.ultralytics.com/license).
- SigLIP2's Hugging Face card lists Apache-2.0. MobileNetV2's TorchVision implementation and its pretrained weights have their own notices. Check every dependency and weight license before commercial use.
- Inference outputs are proposals for review, not inventory transactions. They do not write to the app or database.
