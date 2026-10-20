# RunPod Scene Inference

This isolated pass processes 25 employee and 5 customer scene photographs. It uses the previously tested NVIDIA Retail DINO model with the TensorRT 10.14 / ONNX Runtime 1.24 environment, then ranks each DINO box against 34 paired approved owner reference crops using the pinned SigLIP2 model. The original gallery contains unmasked box crops; the masked gallery contains the same crop windows with background replaced by gray. A combined score averages each SKU's two gallery scores. Detection boxes and query crops are encoded once and reused for every gallery. It does not train, edit production data, use scene annotations as references, or generate scene masks. None of the proposed SKU identities is owner-confirmed.

## Pod requirements and run

The v3 ZIP contains the photos, paired reference crops, manifest and scripts. It intentionally contains **no model weights or TensorRT plugins**. The existing Pod must expose a CUDA GPU, the remapped NVIDIA Retail DINO ONNX file, both Python environments, the SigLIP2 snapshot, ONNX Runtime's TensorRT provider, and the NVIDIA custom-op plugins. The shell/Python syntax and archive checks performed locally do not prove those remote requirements. The runner verifies the DINO file SHA-256, runtime providers, full TensorRT graph profiling and CUDA-backed SigLIP2 execution; stop and report any failed preflight instead of changing models or downloading weights.

Only after the user explicitly authorizes GPU inference, confirm that the existing Pod is healthy and that these prerequisites are present. Upload `runpod_scene_inference_input_20261020_v3.zip` to `/workspace` using the Pod's JupyterLab file browser or an authenticated SSH transfer. Open a JupyterLab Terminal and run:

```bash
printf '%s  %s\n' '<SHA-256 printed in the local readiness report>' '/workspace/runpod_scene_inference_input_20261020_v3.zip' | sha256sum -c -
unzip -q -o /workspace/runpod_scene_inference_input_20261020_v3.zip -d /workspace/new_product_scanning
bash /workspace/new_product_scanning/run_all.sh /workspace/runpod_scene_inference_input_20261020_v3.zip
```

The runner requires the already-used Pod paths:

- `/workspace/trt1014-probe-20261019/bin/python`
- `/workspace/webshop-venv/bin/python`
- `/workspace/ngc_detector_probe_20261019/retail_detector_domain_remap.onnx`
- `/workspace/webshop_benchmark/models/siglip2`

It checks the NVIDIA detector SHA-256 against the verified model, forces TensorRT-only profiled graph execution, verifies actual PyTorch CUDA inference for SigLIP2, and creates a timestamped results ZIP in `/workspace`. If a required path is absent, stop and inspect the workspace rather than downloading weights or changing model settings.

Copy the generated `new_product_scanning_results_<run-id>.zip` to the local experiment using JupyterLab download or authenticated SSH transfer. Record the `sha256sum` printed on the Pod, then on Windows verify and import it with:

```powershell
& .\.venv\Scripts\python.exe .\experiments\new_product_scanning_20261020\verify_scene_results.py '<downloaded-results.zip>' --expected-sha256 '<Pod-printed-64-character-SHA-256>' --import-results
```

The verifier checks the outer SHA-256, ZIP CRC, every inner `SHA256SUMS` entry, 30 scene records, all three gallery outputs and paired embeddings before adding a new isolated results directory. Reopen the local Scene review page to inspect predictions and annotate one physical item per box. Predictions do not approve labels or update inventory.

## Interpretation

- NVIDIA DINO score is a detector score at the existing 0.50 threshold.
- SigLIP2 similarity is the mean of the two highest cosine scores among that SKU's approved owner crops for a given gallery. It is a ranking score, not calibrated confidence.
- `predictions.csv` has one row per DINO detection and side-by-side original, masked, and combined top-three rankings. All three use exactly the same detection boxes and query embeddings. The unprefixed `top1` fields mirror original for older review tools.
- `embeddings.npz` preserves paired reference and one-time query embeddings so the 1, 2, 3, and all-reference registration comparison can reuse the same held-out scene queries after their identities are reviewed, without another GPU pass.
- No unknown threshold is applied; every displayed SKU is a provisional suggestion for human review.
- The current saved review for `IMG_4857.jpeg` has duplicate boxes and must be corrected before scoring. This inference pass reports predictions, not accuracy.
- Detection boxes are shown; MobileSAM masks affect only the owner registration references. Counts are per photo and must not be added across multiple views of the same arrangement.
