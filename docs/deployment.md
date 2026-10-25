# Deployment and reproducibility

## Current delivery status

The integrated development app works on loopback port 8509. The older release remains on 8508. A separate gated judge sandbox runs on loopback port 8511. A temporary HTTPS tunnel has been opened for the operator's explicitly allowed email. An actual external browser reached the Cloudflare email sign-in page with TLS enabled. Unauthenticated root, media and stream HTTP requests were denied; automated requests to the sign-in broker received 403. **Authorized email delivery, external WebSocket sessions and full interactive workflows remain unverified.** Judge addresses have not been added. The screenshots show the actual isolated sandbox, not evidence of completed external acceptance testing. The changing URL and email are kept out of public source.

Read-only AWS checks on October 25 found an active Free plan, the existing capture Lambda, no EC2 instances in eu-north-1 and no CloudFront distributions. Existing private S3/Lambda capture is reusable; it is not a Streamlit host. No billable resource was created and the existing $10 ceiling remains unchanged.

The operator's first email-PIN attempt did not deliver a code. This is an actual delivery blocker, not a successful sign-in. Keep access restricted while investigating; an arranged live screen-share is the fallback. The photo-free narrated local preview is about 2 minutes 44 seconds and is explicitly labeled as a review preview. It is not uploaded or represented as the final submission video.

## Simplest delivery options

| Option | Cost and fit | Blockers / trade-offs |
|---|---|---|
| Email-restricted temporary HTTPS tunnel to the isolated judge copy | Recommended immediate path; no new AWS host. Cloudflare Tunnel is available on all plans; its documented Quick Tunnel email control was verified in the installed official client | Explicit judge emails; laptop/server/tunnel must remain online; hostname changes on restart; no uptime guarantee; whole-host authentication and WebSockets must be externally checked |
| Named Cloudflare Tunnel + Access | More stable hostname without changing the app; free access tier is available for small teams | Existing domain/account/access setup and a configured allowlist; not prepared as an already working service |
| New AWS CPU EC2 host + HTTPS reverse proxy | Official Price List API: eu-north-1 Linux t3.medium $0.0432/hour, about $1.04/day compute only | Requires explicit billable-resource approval, storage/IPv4/traffic allowance, domain/TLS, model/private service configuration and memory/session tests. Four GiB is a candidate, not a load-tested sizing guarantee |
| Streamlit Community Cloud | Hosted HTTPS and private viewer access are supported | Current docs warn of variable resource limits (up to about 2.7 GiB); private models, AWS authentication and the existing local RunPod connection need new configuration. More migration risk before the deadline |

Sources: [Cloudflare Tunnel](https://developers.cloudflare.com/tunnel/), [Quick Tunnel controls/limits](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [Cloudflare free access pricing](https://www.cloudflare.com/plans/), [Streamlit architecture](https://docs.streamlit.io/develop/concepts/architecture/architecture), [Community Cloud limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [AWS EC2 pricing](https://aws.amazon.com/ec2/pricing/on-demand/). The EC2 number was read directly from the AWS Price List API, not inferred from another Region. Promotional credits do not authorize spending. Existing RunPod charges remain separate.

## Local judge setup

Use the verified Python 3.14 environment or create a separate environment; preserve the existing 8508/8509 environment. These are the installed/tested package pins, not a claim that every platform combination has been tested.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-judge.txt
.venv\Scripts\python.exe tools\prepare_judge_demo.py
.\tools\start_webshop_judge.ps1
```

Open http://localhost:8511. Setup generates credentials into ignored `data/judge_release/access.json`; it never overwrites an existing sandbox. There are no default public passwords. The fictional workbook catalogue and shelf coordinates are seeded without private photos, fake embeddings or opening-stock writes. The application access code precedes role login and expires after 30 minutes. The sandbox does not import the working shop.

For the existing full development copy, use `./tools/start_webshop_ux_demo.ps1` and http://localhost:8509. Its setup script requires the operator's existing release database; do not advertise that private-data copy as a clean-clone quickstart.

## Model and AWS prerequisites

Product browsing, price calculation and map use the fictional database. New reference registration needs the existing pinned SigLIP2 checkpoint available locally; set `WEBSHOP_SIGLIP_PATH` to that cache. No model download is automatic in this release. LIVE detection needs the existing operator-managed private RunPod connection. An ignored `data/webshop_demo/worker_connection.json` may contain only the loopback URL and server-side token; never add it to Git or serve the worker publicly.

The AWS capture client uses an operator-created private configuration at `experiments/opencv5_guided_scan/aws_pilot/deployment_resources_20261020.json`. Copy and fill the adjacent `capture_config.example.json` with **existing authorized** resources and your own project/profile. Authenticate the AWS profile outside the app. No infrastructure is created by setup. Keep the selected Region eu-north-1. If AWS is unavailable, explicitly choose LOCAL; failed GPU inference never becomes a fabricated LIVE result or silent replay.

## HTTPS sharing procedure

The verified official cloudflared 2026.10.3 binary is kept under ignored local tooling, with its SHA-256 recorded separately. A fresh operator can obtain the official binary and digest from [Cloudflare's releases](https://github.com/cloudflare/cloudflared/releases). Do not use an unverified executable or expose the working 8509 app.

```powershell
.venv\Scripts\python.exe tools\fetch_cloudflared.py
.venv\Scripts\python.exe tools\share_judge_demo.py --allowed-email judge@example.org
```

Use actual approved addresses, not the example. The script refuses empty/wildcard lists, validates sandbox credentials/reference isolation, checks the official executable digest, and targets **8511 only**. It opens no tunnel until explicitly run with the allowlist. Keep logs private. For a new hostname, restart only the isolated judge process with `./tools/start_webshop_judge.ps1 -PublicHostname HOSTNAME`; this configures Streamlit's browser address and HTTPS port while retaining CORS/XSRF protection. Do not disable protection to work around an untested proxy.

Before distributing a judge URL, externally verify: unauthorized WebSocket upgrade denial; email sign-in; app access code; separate owner/employee login; upload/quality/inference; session reconnect/expiry; buyer no-write behavior; mobile layout; AWS failure/LOCAL choice; GPU-unavailable error. Current evidence covers local browser behavior, external TLS/email sign-in page, HTTP access denial and earlier genuine AWS/model calls. **Authenticated external workflows, email delivery, proxy/WebSocket handling and authenticated responsiveness remain unverified.**

One Streamlit process and one isolated SQLite database are intentional. Browser tabs have separate WebSocket sessions; a server restart loses sessions and active in-memory drafts, while saved catalogue/ledger/evidence remain on disk. Back up SQLite with its backup API, not a live file copy. Do not add autoscaling or replicas before persistence and session affinity are designed.

## Rollback and recording

The 8509 source/SQLite snapshot is in ignored `data/judge_release/rollback_8509`; 8508 remains a separate running release. The judge launcher has its own port and data directory. Stop only the judge/tunnel processes to withdraw access. Keep approved resources through judging if needed; no cleanup of existing AWS/RunPod resources was performed. For a reliable video, prepare the local scanner once before recording; every subsequent LIVE scan still calls real DINO. Public or unlisted upload is a final operator action.
