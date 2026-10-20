# Third-party models, datasets, and dependencies

Prepared 2026-10-20 for the [Web&Shop OpenCV AI Competition 2026 entry](webshop_opencv_2026_submission_20261020.md). This inventory records what the repository itself states, plus licence metadata read from the packages installed in this workspace. It separates verified statements from unresolved ones: an entry marked **confirm** has no licence record in this repository and must be checked against the upstream source before any release, hosted service, or commercial use. Nothing here is legal advice.

The project's own code and documentation are covered by [`LICENSE`](../LICENSE). See [Combined-work obligations](#combined-work-obligations-agpl-30) below for the constraint that matters most.

## Runtime dependencies in `requirements.txt`

Licence strings in this table were read from the `dist-info` metadata of the versions installed in this workspace's `.venv`, not from the declared pin range. A later or earlier version may declare different terms.

| Package | Pin | Installed | Declared licence | Notes |
|---|---|---|---|---|
| `streamlit` | `>=1.45,<2` | 1.66.0 | Apache-2.0 | |
| `ultralytics` | `>=8.4.0` | 8.4.183 | **AGPL-3.0** | Required, not optional. See below. |
| `Pillow` | `>=10.1` | 12.3.0 | MIT-CMU | |
| `numpy` | `>=1.26` | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | |
| `opencv-python` | `>=5.0,<6` | 5.0.0.93 | Apache-2.0 | The wheel bundles native libraries with their own notices; those are not enumerated here — **confirm** before redistributing the wheel. |
| `torch` | `>=2.4` | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | |
| `torchvision` | `>=0.19` | 0.29.1 | BSD | Metadata says only "BSD"; the project states BSD-3-Clause for the torchvision code. |
| `requests` | `>=2.27,<3` | 2.34.2 | Apache-2.0 | |

## Experiment and deployment dependencies

These appear in per-experiment requirements files and are **not installed in this workspace**, so their licences could not be verified here. Each is marked **confirm**.

| Package | Declared in | Status |
|---|---|---|
| `onnxruntime`, `onnx` | [`experiments/hybrid_pipeline/requirements.txt`](../experiments/hybrid_pipeline/requirements.txt) | **confirm** |
| `transformers`, `huggingface_hub`, `safetensors` | [`requirements-encoders.txt`](../experiments/hybrid_pipeline/requirements-encoders.txt), [`requirements-runpod.txt`](../experiments/runpod_gpu_inference/requirements-runpod.txt) | **confirm** |
| `boto3`, `botocore`, `jmespath`, `s3transfer`, `python-dateutil`, `urllib3`, `six`, `opencv-python-headless` | [`aws_pilot/requirements-zip.txt`](../experiments/opencv5_guided_scan/aws_pilot/requirements-zip.txt) — the Lambda ZIP only | **confirm** |
| `PyYAML`, `tqdm`, `scipy` | the RunPod requirements files | **confirm** |

## Models and checkpoints

No model weights are committed to this repository; [`.gitignore`](../.gitignore) excludes `*.pt`, `*.pth`, `*.onnx`, `*.safetensors`, and `*.engine`.

| Model | Where it is used | Terms as recorded in this repository | Restriction |
|---|---|---|---|
| NVIDIA Retail Object Detection (`retail_object_detection_binary_v2.2.2.3.onnx`) | Archived DINO boxes; [`prepare_trt_dino_model.py`](../experiments/new_product_scanning_20261020/prepare_trt_dino_model.py) | NGC governing terms, with one-time acceptance per NGC organization. The catalog describes the pinned ONNX as ready for commercial use; the exact version's governing terms must be reviewed before deployment. | Entitlement and terms acceptance are per-account and were not verified from this workspace. |
| NVIDIA Retail Object Recognition (`FANDualHead_Base_NVIN_op16.onnx`) | Benchmark comparison arm (uncalibrated; it abstained) | Covered by a Model EULA; downloading accepts those terms. | Not used in any claimed result. |
| SigLIP2 Base `google/siglip2-base-patch16-224`, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | SKU retrieval and reference comparison | Apache-2.0 per its model card, recorded in [`prepare_models.py`](../experiments/runpod_gpu_inference/prepare_models.py) and the [hybrid pipeline README](../experiments/hybrid_pipeline/README.md). | The model card does not clear its upstream training data or every dependency. |
| DINOv2 Small | Alternative encoder arm | Apache-2.0 per its model card. | Raw feature encoder, no task head. |
| DINOv3 | Not available; gated | Requires accepting Meta's separate DINOv3 License and gated access. The upstream reference repository's "CC-BY-NC" shorthand is not authoritative for those weights. | Not used. Do not equate it to Apache or MIT. |
| MobileSAM (`mobile_sam.pt`) | Core application segmentation; downloaded by [`model_assets.py`](../model_assets.py) from the `ultralytics/assets` v8.4.0 release and loaded via `ultralytics.SAM` in [`sam_vision.py`](../sam_vision.py) | **confirm.** This repository records no licence for the checkpoint. The [RunPod YOLO11-seg README](../experiments/runpod_yolo11_seg/README_RUNPOD.md) states it has "separate provenance/licensing questions", and the [registration experiment README](../experiments/new_product_scanning_20261020/README.md) says to check it before deployment. | It is also loaded through AGPL-3.0 Ultralytics code, so the AGPL position below applies to this path. |
| SKU-110K YOLO11n (`sku110k-yolo11-n640.pt`, SHA-256 `4888d44cee1a3c55fa961599b762bad0e5aa31194237d8d10d717dff15e0cfdb`; ONNX export `sku110k-yolo11-n640.onnx`) | Retail box proposals in the hybrid pipeline and the RunPod detection bundle; source `https://huggingface.co/chistopat/sku110k-yolo11-object-detector` per [`build_runpod_yolo_detection_bundle.py`](../tools/build_runpod_yolo_detection_bundle.py) | Licence `other`. The model card does not claim anything more permissive than SKU-110K's upstream source-data terms. | **Research only.** Per the [bundle README](../experiments/runpod_yolo_detection/README.md), do not ship it in Web&Shop or use it for a commercial service without written rights clearance. |
| YOLO11n-seg (`yolo11n-seg.pt`) | Optional COCO-pretrained segmentation comparison | Recorded in [`prepare_models.py`](../experiments/runpod_gpu_inference/prepare_models.py) as "Ultralytics AGPL-3.0 model; research comparison only." | Research comparison only. |
| YOLOE | Visual-prompt fallback imported by [`vision.py`](../vision.py) | Ultralytics model and API; AGPL-3.0 by default. | See below. |
| ResNet18 (`resnet18-f37072fd.pth`) | Core application embedding matcher; downloaded by [`model_assets.py`](../model_assets.py) from `download.pytorch.org` | torchvision pretrained weights. The installed torchvision declares "BSD". | **confirm** the terms of the pretrained weights and their source dataset separately from the torchvision code licence. |
| MobileNetV2 | Experimental feature extractor | The torchvision implementation is BSD-3-Clause per the [hybrid pipeline README](../experiments/hybrid_pipeline/README.md). | **confirm** the exact pretrained-weight and source-dataset terms before commercial distribution. |
| YOLOX | Candidate alternative only; not used | Apache-2.0 codebase. | No retail checkpoint has been licence-verified; a permissive code licence does not clear weights. |

## Datasets and photographs

| Item | Terms as recorded | Restriction |
|---|---|---|
| SKU-110K | Reaches this project only through the YOLO11n checkpoint above, whose downstream use is tied to SKU-110K's source-data terms. | Research only; no commercial clearance. |
| The source datasets behind the ResNet18, MobileNetV2, and SigLIP2 pretrained weights | Not identified anywhere in this repository. | **confirm** before commercial use. |
| Project photographs — 193 audited source paths covering 176 distinct SHA-256 images, including the 72 audited owner, employee, and customer photos | Captured by the operator. Excluded from Git by [`.gitignore`](../.gitignore), so a public checkout contains none of them. The handoff READMEs note the original bytes are unchanged and may retain camera EXIF metadata. | Product packaging and brand marks are visible in the photographs, so third-party packaging and trademark rights are unresolved. The [technical report](webshop_opencv_2026_technical_report_20261020.md) requires reviewing rights for photos, packaging, and submitted media before publication, and showing judges only approved images. |
| CVAT polygons, approved masks, and `candidate_skus.csv` identities | Produced by this project and explicitly provisional. | Not ground truth; do not publish as verified labels. |
| `sku-vision-pipeline` (`https://github.com/Sanket758/sku-vision-pipeline`) | Its README states MIT for its code. The [hybrid pipeline README](../experiments/hybrid_pipeline/README.md) notes this "does not establish the license of every checkpoint or dataset it references". | The retrieval shape was borrowed; no code or fusion weights were reused line-for-line. |
| CVAT, NGC, RunPod, and AWS as services | Service and tool terms are not recorded in this repository. | **confirm** against each provider's current terms. |

## Combined-work obligations (AGPL-3.0)

`ultralytics` is a required entry in [`requirements.txt`](../requirements.txt), not an optional extra, and the version installed here (8.4.183) declares `AGPL-3.0`. It is imported by core application modules, not only by experiments:

- [`sam_vision.py`](../sam_vision.py) imports `ultralytics.SAM` and `ultralytics.models.sam.predict.Predictor` to load and run MobileSAM.
- [`vision.py`](../vision.py) imports `ultralytics.YOLOE` and `ultralytics.models.yolo.yoloe.YOLOEVPSegPredictor`.

Web&Shop is a Streamlit application intended to be offered to shop owners, employees, and customers. AGPL-3.0's network provision ordinarily requires whoever offers such a combined work to users over a network to offer those users the corresponding source of the whole combined work. Ultralytics publishes a commercial licence as the stated alternative for closed-source and SaaS use, and this project's own experiment READMEs already record that a proprietary product must either meet the AGPL obligations or obtain that licence.

This is unresolved, and it is a decision for the team rather than a conclusion drawn here. The realistic options are:

1. Comply with AGPL-3.0 for any distributed or hosted build, and licence the combined work accordingly.
2. Obtain a commercial Ultralytics licence before any proprietary or hosted deployment.
3. Replace the Ultralytics dependency in the core path — for example a differently licensed SAM implementation plus a licence-cleared detector — and keep Ultralytics confined to research experiments that are not distributed.

Until one is chosen, treat the deployable Web&Shop build as not cleared for commercial distribution, and do not describe the project as MIT-licensed software without the scope note in [`LICENSE`](../LICENSE).
