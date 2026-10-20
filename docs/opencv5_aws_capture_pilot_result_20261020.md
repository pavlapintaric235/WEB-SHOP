# Web&Shop / InventoryLens OpenCV 5 + AWS capture pilot result

Date: 2026-10-20  
Project: `Web&Shop`  
Selected Region: `eu-north-1`  
Status: deployed, checked, and retained through the OpenCV competition submission at the user's request.

Web&Shop is the competition entry; InventoryLens is the visual scanning module. This pilot is the **live AWS OpenCV 5 capture-guidance component** of the Web&Shop demonstration. Its earlier DINO/SigLIP2 predictions are archived local replay data, and a live cloud recognition service is planned work. See the [submission narrative and architecture diagram](webshop_opencv_2026_submission_20261020.md) and [technical report](webshop_opencv_2026_technical_report_20261020.md).

## What ran

The isolated [guided-scan demo](../experiments/opencv5_guided_scan/live_demo.py) can send a selected photo to the private pilot only after the operator clicks **Run AWS capture check**. An S3 `incoming/` event invokes the Python 3.13 Lambda with OpenCV 5.0.0 and NumPy 2.5.4. The Lambda writes an encrypted, private `results/` JSON containing image SHA-256, capture metrics, a `retake` or `proceed` decision, and runtime provenance. The demo accepts the result only when it matches the selected image hash and the local capture decision. It then continues to human product review. The Lambda does not run NVIDIA Retail DINO or SigLIP2 and cannot commit inventory.

The [resource manifest](../experiments/opencv5_guided_scan/aws_pilot/deployment_resources_20261020.json) records the exact bucket, Lambda, role, log group, and release key. The bucket blocks public access, uses SSE-S3 encryption and TLS-only access, and has a one-day object lifecycle rule. Logs have three-day retention. The Lambda role is scoped to read `incoming/`, write `results/`, and write its own logs. The event notification is filtered to `incoming/`, so result writes do not trigger another invocation. No GPU, API Gateway, public endpoint, ECR repository, or production integration was created.

## Measured verification

| Check | Result |
|---|---:|
| Approved source paths / distinct original image hashes | 193 / 176 |
| Original images with matching AWS source SHA, action, and metrics | 176 / 176 |
| Original capture decisions | 152 `proceed`; 24 `retake` |
| In-memory blur and dark controls | 2 / 2 requested `retake` for the expected reasons |
| Result mismatches in the frozen batch audit | 0 |
| Separate live-demo image check | 1 / 1 matching SHA and `proceed` action |
| Final S3 inventory | 179 inputs, 179 results, 1 release ZIP |
| Final S3 bytes at inventory check | 746,010,511 bytes across 359 objects |

The 24 original `retake` outputs comprise 19 focus prompts, four dark-image prompts, and one resolution prompt. Seventeen were product reference photos and seven were scenes. These are outputs of illustrative thresholds, **not** measured photo-quality accuracy. The frozen batch audit summary and the 178 downloaded per-image result JSONs are held locally and available on request; they are not in this repository. What this repository ships for that batch is the [local expected-decision baseline](../experiments/opencv5_guided_scan/aws_pilot/local_expected_decisions_20261020.csv), which records 176 distinct source SHA-256 values with 152 `proceed` and 24 `retake` decisions and exactly those 19 focus, four dark, and one resolution reasons; one preserved live Lambda [result JSON](../experiments/opencv5_guided_scan/aws_pilot/smoke_result_20261020.json) for the `early3_ref` original; and the batch's [179 CloudWatch REPORT lines](../experiments/opencv5_guided_scan/aws_pilot/lambda_reports_20261020.json). The separate live-demo result JSON, checked against the original E01 image hash and reporting OpenCV 5.0.0 with a 268.35 ms handler processing time, is likewise local only. It is an additional run, outside that frozen batch audit.

On 2026-10-20 at approximately 19:00 Europe/Zagreb, a separate B07 guided-review check used the approved frozen photo `IMG_4764.jpeg`. The existing client verified the Web&Shop project, selected Region, source SHA-256 `16f3fbf55a44395c79dc530b6f2ee149f3ef24e574385f9e2fdf06e2b1cbda57`, and matching AWS result. Lambda reported OpenCV 5.0.0, `proceed`, and 719.74 ms handler processing time. The local result was written as `B07_capture_result.json` in a local `runs/live_aws_review_20261020/` directory outside this repository; it is not included here and is not currently present in this working tree. Streamlit AppTest then exercised the B07 frozen-photo UI in AWS mode: it displayed the matching live AWS result, explicitly labelled the one DINO/SigLIP2 box as archived replay, and withheld export until a person checked every visible product. This additional run is outside the 176-original frozen audit. It did not execute recognition models or approve an inventory change.

In a run recorded on 2026-10-20, five offline handler checks passed and Streamlit's AppTest loaded the ordinary frozen-photo review without errors, showed no AWS run button until AWS mode was selected, held product review until an explicit run click, and showed the returned AWS decision and human-review controls after the click. No output from either run is committed here, and neither reproduces in a clean checkout: `test_offline.py` reads a B01 original from a `data/recognition_checks/...` path that is not in this repository, and the AppTest AWS path needs an active `Web&Shop` session. Both are a recorded past result, not a check a judge can re-run as shipped. The local decision and AWS decision agreed for that live demo photo. Product proposals shown for the exact frozen image are saved DINO/SigLIP2 output; neither model was invoked by this pilot.

For the frozen batch, CloudWatch recorded 179 Lambda REPORT entries, including the import test. Total billed duration was 136,546 ms, maximum billed duration 4,988 ms, and maximum reported memory 240 MB. The original-image handler processing p95 was 1,885.36 ms. At the public Stockholm x86 Lambda rates, the batch's Lambda portion is approximately **$0.00117 before credits**. This is a calculation, not a posted bill; the extra live-demo invocation, S3 operations/storage, logs, taxes, and other project usage are additional. The user-approved new pilot ceiling remains **$10**, while the project reported an active Free plan with $100 credits at the last Free Tier API check. Check actual spend in AWS Settings > Billing, allowing for billing delay. [Lambda price file](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSLambda/current/eu-north-1/index.json); [S3 price file](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/eu-north-1/index.json); [CloudWatch price file](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonCloudWatch/current/eu-north-1/index.json)

## Scope and next decision

This demonstrates OpenCV 5 capture guidance and an AWS S3-to-Lambda path on existing test photos. It does not establish SKU recognition accuracy, complete physical product counts, or readiness to update inventory. The [benchmark audit](../data/recognition_checks/final_comparative_gpu_benchmark_20261018/benchmark_audit_architecture_decision_20261019.md) leaves DINO and SigLIP2 provisional: a five-image detector holdout is small, recognition labels and gallery coverage are incomplete, and crowded scenes have missed or merged items. The guided review must continue to allow manual additions, splits, exclusions, and user-confirmed counts, including E01's eight units and B10's ten units across two views.

The next competition step is to record a short screen capture of the existing demo showing an original photo reaching human review, an in-memory blur/dark photo requesting a retake, and one explicit AWS run with the matching hash and OpenCV version visible. State that DINO/SigLIP2 proposals are archived and that the reviewer decides SKU and quantity. No new AWS resources or GPU run are needed for that demonstration.

The user chose to retain the private pilot through submission. The one-day lifecycle removes stored photos, results, and the release ZIP as they age; it does not delete the bucket, Lambda, role, or log group. The demo's 200-current-input guard resets as objects expire, so it is not an enforceable cumulative spend limit. Further uploads should remain within the approved pilot scope and $10 ceiling.
