# AWS one-time scene inference runner

This archive is for the same isolated 25 employee plus five customer scene experiment as the preserved v3 input. It contains no model weights and makes no AWS API calls. The source folder name `runpod_inference` is historical; do not use the RunPod paths as evidence that AWS inference has run.

The ONNX Runtime TensorRT plugin path requires the NVIDIA custom nodes in the `trt.plugins` domain. [ONNX Runtime's TensorRT provider documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#tensorrt-plugins-support) specifies that domain. The local experiment's `prepare_trt_dino_model.py` changes only those 12 node domains and their opset import in the original, hash-verified NVIDIA ONNX. Its candidate SHA-256 is `7abf51f62888433d27811caa6af4b486cccd6382fcfe83a2b9c3a4bcd244aafd`. This differs from the earlier successful Pod remap hash `c6185e3b1cd01abfe9d83e7f6f13083a8485b6cd0ca409642f237a5c65c36f41`. Local ONNX checker success is **not** GPU execution or numerical-equivalence proof.

After AWS quota approval, explicit authorization for the billable run, and hash-verified transfer, create `/workspace`, provide the candidate ONNX, the pinned SigLIP2 snapshot, and two CUDA-capable Python environments. Set `TRT_PY`, `TORCH_PY`, `DINO_MODEL`, `DINO_MODEL_SHA256`, and `SIGLIP_MODEL` to their actual paths. `run_all.sh` defaults to the earlier Pod model hash unless `DINO_MODEL_SHA256` is explicitly set. Use a fresh TensorRT engine cache on the L4 GPU.

Preflight with a single packaged image before the 30-scene run. Require: `nvidia-smi`, PyTorch CUDA availability, ONNX Runtime `TensorrtExecutionProvider`, registered `MultiscaleDeformableAttnPlugin_TRT` version 1, successful graph profiling with no CPU graph nodes, and plausible detector boxes. A one-image pass checks execution, **not** numerical equivalence to the earlier remapped model; compare on identical holdout photos before claiming equivalence. If the candidate fails, stop without scoring the 30 scenes and recover the exact earlier remapped file or diagnose the runtime. Do not silently change detector weights or thresholds.

With preflight passing, run from `/workspace`:

```bash
printf '%s  %s\n' '<locally-recorded-v4-ZIP-SHA-256>' '/workspace/aws_scene_inference_input_20261020_v4.zip' | sha256sum -c -
printf '%s  %s\n' '7abf51f62888433d27811caa6af4b486cccd6382fcfe83a2b9c3a4bcd244aafd' '/workspace/retail_detector_trt_plugins_20261020.onnx' | sha256sum -c -
unzip -q /workspace/aws_scene_inference_input_20261020_v4.zip -d /workspace/new_product_scanning
export DINO_MODEL=/workspace/retail_detector_trt_plugins_20261020.onnx
export DINO_MODEL_SHA256=7abf51f62888433d27811caa6af4b486cccd6382fcfe83a2b9c3a4bcd244aafd
export SIGLIP_MODEL=/workspace/siglip2
PREFLIGHT_ONLY=1 LIMIT_SCENES=1 RUN_ID=preflight_$(date -u +%Y%m%dT%H%M%SZ) bash /workspace/new_product_scanning/run_all.sh /workspace/aws_scene_inference_input_20261020_v4.zip
bash /workspace/new_product_scanning/run_all.sh /workspace/aws_scene_inference_input_20261020_v4.zip
```

The environment-specific `TRT_PY` and `TORCH_PY` values must be set before the preflight command. Inspect the preflight detector report and overlay before running the final command; the wrapper does not run recognition or create a results ZIP in preflight mode. These commands are an execution template, not a claim that the environment exists. Download the result ZIP before the pilot bucket's one-day S3 expiration, record its remote SHA-256, and use the existing local `verify_scene_results.py --expected-sha256 ... --import-results` check. Predictions remain provisional until owner SKU and physical-instance review.
