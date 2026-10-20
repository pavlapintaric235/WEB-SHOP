# RunPod Preflight and SSH/SFTP Guide

Snapshot date: 2026-10-18, refreshed after the CUDA minimum-version correction. No Pod was created.

## Live Secure GPU Catalog

The latest live query used `cloud=SECURE`, `product=POD`, `count=1`, and `minCudaVersion=12.8`. This treats CUDA 12.8 as a minimum, not an exact host-driver version. Memory is reported by RunPod in GB.

| GPU | VRAM | Secure GPU rate | Latest Secure availability / region | CUDA result |
|---|---:|---:|---|---|
| RTX A5000 (preferred) | 24 GB | $0.27/hr | LOW; CA-MTL-1 LOW | CUDA 13.0 available; template permits CUDA 13.0 |
| RTX 3090 (preferred fallback) | 24 GB | $0.50/hr | NONE | No Secure stock in this filtered read |
| B300 MIG 1g.34gb | 34 GB | $0.50/hr | LOW; US-WA-2 LOW | CUDA 13.2 available; template allows 13.2 |
| RTX PRO 4000 Blackwell | 24 GB | $0.57/hr | LOW; EU-RO-1 LOW | CUDA 13.0 and 13.2 available |
| A40 | 48 GB | $0.59/hr | LOW; CA-MTL-1 and EU-SE-1 LOW | CUDA 12.8, 13.0 and 13.2 available |
| L4 | 24 GB | $0.59/hr | LOW; EU-RO-1, EUR-IS-1 and US-MO-2 LOW | CUDA 12.8, 13.0 and 13.2 available |

RunPod's live `runpod-torch-v280` template is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 30 GB container disk, 50 GB persistent `/workspace`, Jupyter on `8888/http`, and SSH on `22/tcp`. Its allowed CUDA list includes 12.8 through 13.4. The `minCudaVersion=12.8` filter is a floor, not an exact driver match. The Secure A5000 is currently the cheapest available matching GPU at $0.27/hr in CA-MTL-1, and its host CUDA 13.0 can run the template's CUDA 12.8 runtime under NVIDIA's documented backward-compatibility model. The RTX 3090 currently has no stock. The B300 MIG is the next available option at $0.50/hr, in US-WA-2, with host CUDA 13.2. Actual device initialization must still pass the pre-benchmark CUDA smoke test.

### Storage-Inclusive Estimate

The RunPod pricing page currently lists container disk at $0.10/GB/month while running, volume disk at $0.10/GB/month while running and $0.20/GB/month while stopped. The proposed 30 GB + 50 GB configuration therefore adds $8/month while running. Using 730 hours/month, the estimate is:

| GPU candidate | GPU | Running storage estimate | Estimated all-in running rate |
|---|---:|---:|---:|
| RTX A5000 | $0.27/hr | $0.01096/hr | $0.28096/hr |
| B300 MIG 1g.34gb | $0.50/hr | $0.01096/hr | $0.51096/hr |
| RTX PRO 4000 Blackwell | $0.57/hr | $0.01096/hr | $0.58096/hr |
| A40 or L4 | $0.59/hr | $0.01096/hr | $0.60096/hr |

Rates are estimates; the deployment screen is authoritative. With the Pod stopped, the container disk is not charged, and the retained 50 GB Pod volume is $10/month (about $0.01370/hour). Do not terminate the Pod or delete its volume.

## Current Local and Account Prerequisites

- Windows OpenSSH `ssh`, `scp` and `sftp` are installed.
- `ngc` and `runpodctl` are not installed locally.
- The RunPod account read returned no Pods and no registered SSH public keys.
- Both `preflight_final_v2` ZIPs remain intact. Their current SHA-256 values are recorded below.

The RunPod MCP can inspect and manage infrastructure but cannot open a Pod shell or transfer files. Use JupyterLab/Console Terminal for Pod commands and OpenSSH SFTP/SCP for transfer.

## NVIDIA NGC Direct-HTTPS Access

NGC's official guide documents direct HTTPS downloads for guest-access models and bearer-key HTTPS downloads for authenticated models; an NGC CLI is not required. Some entities require one-time governing-terms acceptance per NGC organization. The detector catalog lists the signed, pinned ONNX model (299.56 MB) and describes it as ready for commercial use; review the exact version's governing terms before deployment. The recognizer File Browser lists `FANDualHead_Base_NVIN_op16.onnx` (195.86 MB), and its model page explicitly says use is covered by a Model EULA; downloading accepts those terms. Catalog visibility and public curl/wget buttons do not prove this account's current entitlement or that required terms have been accepted.

The NGC File Browsers expose curl/wget actions for both pinned files. I do not have an authenticated NGC session or your API key, so I have verified that the official HTTPS route is documented and the files are listed, but cannot verify your account's model entitlement or terms acceptance from here. `ngc` is absent locally, but it is not required for the HTTPS path.

Before another check, sign in to NGC, review each model's Governing Terms and the recognizer Model EULA, and personally accept any terms required for your intended use. The probe constructs the pinned model-file endpoint itself, so no URLs need to be copied or pasted. For authenticated access, it follows NVIDIA's documented GET shape: `Authorization: Bearer` and `Content-Type: application/json`, with no `HEAD`, `Range`, or explicit `Accept` header. The streamed probe reads at most 1,024 response bytes and closes the response without saving a model file. Since the reported 406 was with a Personal API key, first repeat that same mode using the corrected helper:

```powershell
python .\data\recognition_checks\final_comparative_gpu_benchmark_20261018\download_ngc_models_https.py --probe-only --auth-mode personal
```

The earlier probe returned HTTP 406 for both models. Its request included `Accept: application/octet-stream` and `Range: bytes=0-0`, neither of which appears in NVIDIA's documented authenticated File Browser GET command. HTTP 406 is consistent with content-negotiation rejection, making the added `Accept` header the strongest code-level suspect; the status alone cannot prove which header the endpoint rejected. The response body from that attempt was not retained, so there is no safe server-side subcode available for that result. The updated helper reports only a bounded, regex-checked JSON error-code token if a later response provides one; it does not echo response messages, URLs, or credentials.

For a separate public-access control, the guest mode below sends no API key or extra headers:

```powershell
python .\data\recognition_checks\final_comparative_gpu_benchmark_20261018\download_ngc_models_https.py --probe-only
```

Use `--auth-mode service` for a Service API key. Do not pass a legacy NGC key directly as Bearer; NVIDIA's guide requires exchanging it for a short-lived token. If the corrected GET still fails, do not rent the GPU. Record only the printed HTTP status and optional safe NGC code, then compare through the signed-in NGC browser or official CLI; never send raw logs or credentials. A successful probe does not download weights. The downloader uses the pinned filenames, validates minimum payload size and HTML/error responses, computes SHA-256, updates the package manifest, and never writes keys or request URLs. No NGC CLI is needed for the HTTPS path.

If NGC remains unavailable, the prepared experiment already supports an explicitly partial run without changing the benchmark inputs. After the normal GPU/CUDA and package checks, run:

```bash
python prepare_models.py --with-yolo11-seg
python prepare_comparative_models.py --skip-ngc
python run_comparative_benchmark.py --skip-ngc --output results_without_ngc
```

This evaluates the existing retail YOLO + MobileSAM path, the optional COCO YOLO segmentation arm, and the available DINOv2-small, SigLIP2, and MobileNetV2 recognition arms. It omits the NVIDIA retail detector and NVIDIA retail recognizer; it does not substitute other weights. Once NGC access is confirmed, download those two artifacts with the updated helper and run a second benchmark into a fresh output directory.

Official references:

- [NGC Catalog access and terms guide](https://docs.nvidia.com/ngc/latest/ngc-catalog-user-guide.html)
- [CUDA Compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/latest/)
- [Retail Object Detection model](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/retail_object_detection)
- [Retail Object Recognition model and Model EULA notice](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/retail_object_recognition)

## SSH Key and Upload Workflow

This account currently has zero registered SSH keys. Generate a dedicated key locally; only the `.pub` public key is registered with RunPod. Keep the private key on the laptop.

```powershell
$sshDir = Join-Path $env:USERPROFILE '.ssh'
New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
$key = Join-Path $sshDir 'webshop_runpod_ed25519'
ssh-keygen -t ed25519 -a 64 -C 'webshop-runpod' -f $key
Get-Content "$key.pub"
```

Add the displayed public key in RunPod Settings > SSH Keys before creating a Pod. Do not share the private key. When creating the approved Pod, enable SSH, keep port `22/tcp`, then copy the direct SSH host and port exactly from the Pod's Connect panel; they cannot be known before Pod creation.

The two preserved upload archives are:

- Main experiment: `inventorylens_final_comparative_gpu_benchmark_preflight_final_v2_20261018.zip` (684,355,770 bytes).
- Separate CVAT detector evaluation: `inventorylens_cvat_holdout_detector_eval_preflight_final_v2_20261018.zip` (48,334,096 bytes).

From PowerShell, substitute the SSH host and numeric port shown by RunPod. First use SFTP interactively:

```powershell
$hostName = '<SSH host from Pod Connect>'
$port = <SSH port from Pod Connect>
$key = Join-Path $env:USERPROFILE '.ssh\webshop_runpod_ed25519'
sftp -P $port -i $key "root@$hostName"
```

At the `sftp>` prompt:

```text
mkdir /workspace/uploads
cd /workspace/uploads
put "C:/Users/U938/Downloads/inventorylens_starter/data/recognition_checks/final_comparative_gpu_benchmark_20261018/inventorylens_final_comparative_gpu_benchmark_preflight_final_v2_20261018.zip"
put "C:/Users/U938/Downloads/inventorylens_starter/data/recognition_checks/final_comparative_gpu_benchmark_20261018/inventorylens_cvat_holdout_detector_eval_preflight_final_v2_20261018.zip"
put "C:/Users/U938/Downloads/inventorylens_starter/data/recognition_checks/final_comparative_gpu_benchmark_20261018/download_ngc_models_https.py"
bye
```

Alternatively use `scp -P $port -i $key '<absolute-local-zip>' "root@${hostName}:/workspace/uploads/"` for each archive. Do not disable SSH host-key checking. If RunPod provides a host fingerprint, verify it on first connection.

Verify transfer integrity before extraction. On Windows:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '<local zip path>'
```

In the Pod terminal:

```bash
sha256sum /workspace/uploads/inventorylens_final_comparative_gpu_benchmark_preflight_final_v2_20261018.zip
sha256sum /workspace/uploads/inventorylens_cvat_holdout_detector_eval_preflight_final_v2_20261018.zip
```

Each remote hash must exactly match the local hash. Do not extract or run a mismatching upload. Extract into fresh, versioned paths under `/workspace`; keep originals there until results have been downloaded and verified.

## Execution and Return Transfer

In JupyterLab Terminal, follow the included `README_RUNPOD.md`. The main experiment is inference-only. First verify the selected GPU with `nvidia-smi` and run `python verify_cuda.py`; it performs a real CUDA operation. Stop before benchmark execution if that smoke test fails. Install dependencies without replacing CUDA PyTorch, then run `python verify_package.py`. If NGC is still blocked, use the partial-run commands above. Once access succeeds, prepare DINOv2 with `python prepare_comparative_models.py --skip-ngc`, then fetch the two NGC weights straight to the Pod using the updated helper:

```bash
python /workspace/download_ngc_models_https.py --package /workspace/webshop_benchmark --auth-mode personal
```

It prompts for the Personal API key with hidden input. It constructs pinned URLs internally and does not display or save them. The files are written to the paths expected by the benchmark and their hashes are added to `models/comparative_download_manifest.json`. Then run `python run_comparative_benchmark.py --output results_with_ngc` in a fresh output directory. Run the separate CVAT detector evaluation in a different directory only if desired; it does not modify the 41-photo benchmark or train a model.

Before stopping the Pod, create and CRC-verify the results package with:

```bash
python package_downloads.py --output results_final
```

Download `inventorylens_final_comparative_gpu_results_and_models.zip` using JupyterLab or SFTP `get`. Verify its SHA-256 locally against `sha256sum` from the Pod. Only after the local file exists and the hashes match, stop the Pod in the RunPod console. Do not terminate it: the 50 GB `/workspace` volume must remain.

## Preserved Package Hashes

- Main v2 archive SHA-256: `B7216213512D9D6B8C582DBB8FA1D0506EBBB3C1B8AF798AA1D8704FBB630BC1`
- CVAT evaluation v2 archive SHA-256: `2FE3E584D7CDAC00FDB41779C80C69E169908E5128BB0DE4EFC8AEDCEEBC4C43`
