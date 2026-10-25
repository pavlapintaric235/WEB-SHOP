# Verified results and limits

These are measured development observations, not broad deployment accuracy. The [current technical report](webshop_opencv_2026_technical_report_20261020.md) consolidates the three datasets, completed DINO training, separate localization/retrieval evaluations, and final live demonstration.

| Check | Actual evidence | Boundary |
|---|---|---|
| New SKU registration | Five real reference photos persisted with genuine embeddings; distinct shelf photo assigned the new SKU to the front can | One product and one scene; this is development evidence, not an independent repeated test |
| AWS OpenCV 5 | Valid reference PROCEED, intentionally blurred control RETAKE, distinct scene PROCEED; returned image hashes verified | Measures image-quality decisions, not identity or stock correctness |
| Blurred control | Laplacian variance 1.18 versus 334.23 for the reference | Threshold example; not a general quality-accuracy estimate |
| Matching optimization | Same ten actual crops: serial 14.3656 s; batch-four 7.5673 s; same SKU decisions | One bounded local CPU timing check |
| Prepared LIVE recognition | 27.5903 s: DINO request 20.023 s + matching 7.5673 s; detector core 14.612 s | Excludes upload, AWS capture check, cold preparation and UI rendering |
| Cached interactions | Reference fingerprint 0.0127 s; photo preparation 0.0786 s; gallery snapshot 0.0182 s | Reuses preprocessing/reference vectors, never substitutes predictions for a new LIVE request |
| Ledger safety | Explicit one-unit isolated demo receipt; buyer two-unit estimate; buyer stock unchanged | Fictional demo prices and stock; no real inventory assertion |

The live scene returned ten proposals, including partial proposals and unknowns. Proposals are not verified unit counts. A narrow partial can received the new SKU too; its identity cannot be verified from the fragment alone. The employee review intentionally approved only the front can within a limited fictional action scope.

An earlier 35.340-second run had different loading conditions; it is not a controlled overall-speed baseline. Cold model preparation measured 24.0647 seconds and cold reference hashing 7.6612 seconds. The safe image-only reader avoids an observed Windows memory-mapped weight-loading crash and passed an exact-output tiny-model test; actual full-size scene assignments were preserved.

Earlier reports under `experiments/` document different datasets, settings and review stages. Detector localization metrics and recognition on manually supplied item crops must not be multiplied or described as end-to-end accuracy. The fetched October 20 reference-count analysis remains in Git history at commit `87685fb`; its small provisional-group retrieval result is not current deployment accuracy. No thresholds were tuned in this release.

Reproduce application safety checks with:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_judge_release tests.test_registration_8509 tests.test_practical_ux
```

These tests use isolated databases, real local OpenCV and explicit model/cloud mocks. Tiny random-model tests verify loading/math only. Genuine AWS/model evidence is recorded separately; unit-test success is not inference accuracy. No new live benchmark is required to reproduce the interface.
