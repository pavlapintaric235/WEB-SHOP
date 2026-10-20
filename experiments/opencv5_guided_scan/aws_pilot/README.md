# Web&Shop / InventoryLens S3-to-Lambda capture pilot

Status: isolated InventoryLens capture component deployed and verified within the Web&Shop competition demonstration in selected Region `eu-north-1` on 2026-10-20. The approved batch and optional local-demo connection passed. The user chose to keep the resources through submission. No production code, inventory, or source dataset was changed. This Lambda runs OpenCV 5 capture guidance only; earlier DINO/SigLIP2 outputs remain local archived evidence. See the [Web&Shop submission](../../../docs/webshop_opencv_2026_submission_20261020.md) and [measured result](../../../docs/opencv5_aws_capture_pilot_result_20261020.md).

The [pilot plan](../../../docs/opencv5_aws_capture_pilot_plan_20261020.md) defines the intended `eu-north-1` resources and approval boundary. `handler.py` receives an S3 `ObjectCreated` event for `incoming/<run-id>/<image-id>.jpg|jpeg|png`, reads at most 20 MiB, runs the same `capture_gate.py` module as the local guided-scan replay, and writes a private `results/<run-id>/<image-id>.json`. It never invokes DINO or SigLIP2 and never writes inventory. Results include the source SHA-256 so the review UI can reject a stale or mismatched response. The S3 trigger must be configured for the `incoming/` prefix only.

## Offline check

From the existing project root:

```powershell
& .\.venv\Scripts\python.exe -B experiments\opencv5_guided_scan\aws_pilot\test_offline.py
```

The test uses an in-memory fake S3 client and reads the existing B01 photo without changing it. It covers the original, in-memory blur and dark retakes, invalid and over-limit images, and the output-prefix guard. It does not need boto3 credentials or AWS access.

## Packaging

The selected package is a Python 3.13 x86-64 ZIP. From the existing project root, install the pinned Linux wheels into an isolated build directory and package them with the shared capture module:

```powershell
& .\.venv\Scripts\python.exe -m pip install --target experiments\opencv5_guided_scan\aws_pilot\.build_zip --platform manylinux_2_28_x86_64 --python-version 3.13 --implementation cp --only-binary=:all: -r experiments\opencv5_guided_scan\aws_pilot\requirements-zip.txt
& .\.venv\Scripts\python.exe -B experiments\opencv5_guided_scan\aws_pilot\build_zip.py
```

The Dockerfile remains an unused fallback. The current ZIP is 96,365,662 bytes compressed and 238,861,606 bytes uncompressed, below Lambda's 250 MiB uncompressed limit. It was staged in the private pilot S3 bucket because it exceeds the 50 MiB direct-upload limit. The live Lambda import and first photo result passed.

The user approved the pilot resource set, the described test-photo groups, one-day object retention, and a $10 ceiling for new pilot charges. The upload manifest lists 193 source paths and 176 unique originals totaling 648,285,670 bytes. All 176 originals matched local capture actions and metrics; two in-memory retake controls also passed. One additional E01 demo check returned a matching hash and `proceed`. AWS CLI sign-in for `Web&Shop` succeeded, and the project reported an active Free plan with $100 credits when checked. The [pilot plan](../../../docs/opencv5_aws_capture_pilot_plan_20261020.md) records Stockholm rate inputs and deployed controls.

`upload_manifest.py` performs a local dry run by default. Its upload mode requires `--execute --bucket <private-pilot-bucket>`, uses only the `Web&Shop` profile in `eu-north-1`, revalidates the complete source manifest, and refuses more than 200 distinct photos or 1 GiB. It sends one photo, waits for its Lambda result, and checks the SHA-256, action, and metrics against the local baseline before continuing. `--limit N` is available for a smoke run.

`prepare_expected_decisions.py` writes one local OpenCV result per unique original for comparison with Lambda. The upload mode also needs `boto3` and `botocore[crt]` in the local virtual environment to use the short-lived `aws login` profile; neither is used by the dry run.
