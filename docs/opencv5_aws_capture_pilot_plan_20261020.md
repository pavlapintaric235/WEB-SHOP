# WEB&SHOP / InventoryLens OpenCV 5 + AWS capture pilot

Date: 2026-10-20  
Status: **deployed and verified in the WEB&SHOP project, selected Region `eu-north-1`**. The user approved the pilot, the described test-photo groups, and a $10 maximum for new pilot charges. The user chose to keep the pilot resources through the competition submission. See the [measured pilot result](opencv5_aws_capture_pilot_result_20261020.md).

## Goal and evidence boundary

WEB&SHOP is the OpenCV AI Competition 2026 entry; InventoryLens is its guided visual scanning and inventory intelligence module. Run the existing OpenCV 5 guided capture decision as the deployed AWS component of that integrated demonstration. The decision changes the next action to `retake` or `proceed to human review`. The local [guided scan demo](../experiments/opencv5_guided_scan/README.md) exercises this branch and shows archived NVIDIA Retail DINO boxes and SigLIP2 suggestions. Its frozen eight photos are seven arrangements. They are diagnostic examples, not an accuracy test or threshold tuning set. B10 is one user-confirmed 10-unit arrangement across two views. See the [WEB&SHOP submission narrative](webshop_opencv_2026_submission_20261020.md) for the full relationship and architecture.

The competition asks for substantive OpenCV 5 image analysis and a meaningful AWS component. An S3-triggered Lambda function running the capture decision meets the latter in a small, auditable way; a judge can see the image, computed metrics, and the changed review/retake action. A live screen-share demonstration is an accepted delivery path. This is a proposed interpretation of the rules, not a guarantee of judging outcome. [Competition requirements](https://opencv.org/opencv-ai-competition-2026/)

## Pilot design, all Regional resources in selected Region `eu-north-1`

```text
Operator's WEB&SHOP / InventoryLens guided-scan demo
       | consented test photo, unique key, max 20 MiB
       v
Private S3 Standard bucket: incoming/<run-id>/<image-id>.jpg
       | ObjectCreated event filtered to incoming/
       v
On-demand Lambda Event Function: Python 3.13 + OpenCV 5 headless
       | decode, 1024px resize, focus/exposure metrics, retake/proceed rule
       v
Same private bucket: results/<run-id>/<image-id>.json
       | local demo reads result with operator credentials
       v
Retake prompt or pending human review; no inventory commit
```

The deployed pilot uses one private S3 bucket, one on-demand Lambda function, one Lambda execution role and invoke permission, and a short-retention CloudWatch Logs group. It uses no API Gateway, public endpoint, GPU, VPC/NAT, provisioned concurrency, ECR repository, or production integration. The pinned Python 3.13 Linux ZIP was built locally at **96,365,662 bytes compressed / 238,861,606 bytes uncompressed**, including OpenCV 5, NumPy, boto3, and the isolated handler. It fits Lambda's 250 MiB uncompressed limit; because it exceeds the 50 MiB direct-upload limit, it was staged under `releases/` in the same private S3 bucket. This ZIP path avoided a Docker build and ECR storage. The live Lambda import and photo behavior checks passed. [Lambda ZIP quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html); [Python ZIP packaging](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html)

The trigger listens only to `incoming/`; writes to `results/` cannot invoke the function recursively. Results are keyed by the unique image ID so duplicate S3 events overwrite the same result. The local demo requires an explicit click to upload and displays the AWS result only for the matching image ID and SHA-256. The Lambda result contains `cv2.__version__`, package version, image hash, dimensions, focus/exposure metrics, decision, reason, and runtime duration. It contains no DINO/SigLIP2 inference claim. [S3 trigger and loop warning](https://docs.aws.amazon.com/lambda/latest/dg/with-s3.html); [S3 notification prefix filtering](https://docs.aws.amazon.com/AmazonS3/latest/userguide/notification-how-to-filtering.html)

Security and cost controls: S3 Block Public Access and default encryption; TLS for client access; least-privilege role restricted to reading `incoming/*`, writing `results/*`, and writing its logs; no image bytes in logs; reject non-image or over-20-MiB objects before processing; 15-second Lambda timeout, 512 MB memory, and a pilot limit of 200 unique photo uploads in the batch manifest. The local demo also stops at 200 currently stored inputs; this is not a cumulative spend limit because the one-day lifecycle removes objects. Expire photo/result/release objects after one day and logs after three days. The user chose to keep the bucket, function, role, and log group through submission. The project rejected reserved concurrency 2 because it would leave fewer than the required 10 unreserved executions; the uploader instead sends one photo and verifies its result before the next. The $10 ceiling is monitored operationally; AWS billing data may lag.

## DINO and SigLIP2 boundary

The archived DINO/SigLIP2 proposals remain local and are shown only when the image bytes exactly match an audited frozen photo. A new or recompressed image receives manual review, even if Lambda says `proceed`. This preserves the existing human correction flow and avoids presenting saved predictions as fresh inference. DINO's audited TensorRT GPU engine cannot be run unchanged in a CPU-only Lambda function. SigLIP2 inference in Lambda is unproven here; model size, packaging, cold-start time, CPU latency, and gallery coverage would need separate measurement. Do not rent GPU capacity or move either model for this pilot. The current registration and Owner_test, employee_test, and customer_test workflows remain untouched.

## Capacity and regional cost basis

The user's approved scope spans the early K18/WOW/coconut-water tests, the grocery tests, and the later 10-product owner/employee/customer experiment. The local inventory found **193 candidate original files, 176 distinct by SHA-256**, before optional in-memory retake controls. Source files total **808,416,117 bytes** before duplicate removal and **648,285,670 bytes** for upload. The exact source paths, hashes, and upload keys are recorded in the [local upload manifest](../experiments/opencv5_guided_scan/aws_pilot/photo_upload_manifest_20261020.csv). The release ZIP adds 96,365,662 bytes, so planned incoming storage is 744,651,332 bytes before result JSON. AWS's public `eu-north-1` price files dated 2026-10-07 through 2026-10-16 quote x86 Lambda at **$0.0000166667/GB-second plus $0.0000002/request**, S3 Standard at **$0.023/GiB-month**, S3 PUT at **$0.005/1,000**, S3 GET at **$0.0004/1,000**, and CloudWatch Logs ingestion at **$0.54/GB**. At 512 MB and an assumed two seconds for each of 176 calls, Lambda is about **$0.00297**; one day of the planned photo/release storage is about **$0.00053**; 353 photo/release/result PUT requests are about **$0.00177**. HEAD/GET requests, log bytes, taxes, and any other project use add to this. The first live photo was billed for 4.988 seconds and used 218 MB peak, so the two-second estimate is optimistic. Even 200 calls each running to the 15-second timeout would be about **$0.025** of Lambda compute before request charges. The Free plan API reports **$100 remaining credits** on 2026-10-20; credits do not replace the separate **$10 maximum new pilot spend**. Check actual usage in AWS Settings > Billing after the run. [AWS Price List Bulk API](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-the-aws-price-list-bulk-api-fetching-price-list-files-manually.html); [Stockholm Lambda rates](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSLambda/current/eu-north-1/index.json); [Stockholm S3 rates](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/eu-north-1/index.json); [Stockholm CloudWatch rates](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonCloudWatch/current/eu-north-1/index.json)

The [local expected-decision baseline](../experiments/opencv5_guided_scan/aws_pilot/local_expected_decisions_20261020.csv) runs the same OpenCV gate on those 176 originals: 152 `proceed` and 24 `retake` (19 focus, four dark, one resolution). Of the 24 retakes, 17 are registration/reference photos and seven are scenes. These are gate outputs, not verified photo-quality labels or SKU accuracy. Its per-key results were compared with the deployed Lambda; all 176 matched.

## Acceptance and rollback

1. Local and Lambda OpenCV 5 metrics and decisions agree for an approved original, its in-memory blurred version, and its in-memory darkened version. Compare numeric metrics within a declared tolerance if JPEG encode/decode changes bytes; exact byte equality is required before attaching archived model proposals.
2. An `incoming/` upload creates one matching `results/` JSON and does not recursively invoke Lambda. A bad or large upload produces an explicit error record without an inventory update.
3. The local review keeps every SKU/count pending until human confirmation and correctly treats B10's two views as one 10-unit arrangement.
4. Record request count, actual Lambda duration/memory, S3/ECR/log storage, and a `eu-north-1` cost estimate. Document the result and failure cases for the competition report.
5. The user was asked whether to delete or retain the successfully created resources and chose to keep them through submission; continuing use remains subject to the $10 pilot ceiling.

## Authorization boundary and current access

No AWS MCP tools are available in this session. AWS CLI 2.38.2 is available at its current-user install path and `aws login --profile WEB&SHOP --region eu-north-1` succeeded. STS confirmed project `478400396665`; the Free Tier API reported `FREE`, `ACTIVE`, and $100 remaining credits at the time checked. The private S3 bucket, Lambda function, scoped execution role, and short-retention log group were deployed and verified in `eu-north-1`. Exact resource identifiers are recorded in the [deployment manifest](../experiments/opencv5_guided_scan/aws_pilot/deployment_resources_20261020.json). [Supported services](https://docs.aws.amazon.com/accounts/latest/reference/supported-services-sign-up-new.html)

The user approved the resource set, the described test-photo groups, one-day photo/result retention, and a $10 maximum new pilot spend. The selected Region `eu-north-1` is user-confirmed; AWS Settings > View all projects > Overview > Additional Info > Region is the place to recheck it. The upload manifest was revalidated immediately before the batch; the local demo uploads only after an explicit click. The workspace AWS guidance required a keep-or-clean-up choice after creation, and the user chose to keep resources through submission. No credentials need to be sent in chat.
