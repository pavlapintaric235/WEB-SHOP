# New Product Scanning Dataset Audit

Audit date: 2026-10-20. Source folders are outside the project and were read-only during this audit. No source image, production registry/database, previous experiment, model, or annotation was changed. The manifests retain original filenames and SHA-256 hashes.

## Dataset totals and image properties

| Folder | Photos | File type | Displayed resolution range (width x height) | EXIF orientation notes |
|---|---:|---|---|---|
| `C:\Users\U938\Downloads\Owner_test` | 42 | JPEG | 661-2422 x 847-3806 | all orientation 1 |
| `C:\Users\U938\Downloads\Employee_test` | 25 | JPEG | 1907-4032 x 1756-4032 | five use EXIF orientation 6; images must be EXIF-transposed before annotation/inference |
| `C:\Users\U938\Downloads\Customer_test` | 5 | JPEG | 3024-3689 x 2685-3458 | all orientation 1 |
| **Total** | **72** | JPEG | mixed | mixed |

No byte-identical duplicates were found by SHA-256. This is not a near-duplicate test. Full per-file dimensions, byte lengths, EXIF orientation, and SHA-256 are in [photo_manifest.csv](photo_manifest.csv). Contact sheets are [owner](contact_owner.jpg), [employee](contact_employee.jpg), and [customer](contact_customer.jpg).

## Provisional owner gallery and candidate SKU identities

These are experiment-only IDs, not production registrations. There are **10 visually distinct candidate identities** in the reference folder. Seven are reasonably identifiable from visible packaging; three generic/unpackaged objects need confirmation as actual sellable inventory SKUs and exact variants. The names deliberately stop where the photos stop.

| Proposed ID | Provisional identity from visible image evidence | Owner reference photos | Count | Status |
|---|---|---|---:|---|
| IL-NEW-001 | dmBio coconut water, natural, 1 L | IMG_4810, IMG_4813-IMG_4815 | 4 | packaging/name/size visible; angles provisional |
| IL-NEW-002 | Nescafe 3-in-1 Classic single sachet | IMG_4816-IMG_4821 | 6 | packaging and variant visible |
| IL-NEW-003 | K-Classic tuna salad, Mexican Style, 220 g | IMG_4822-IMG_4825 | 4 | front and product/size visible |
| IL-NEW-004 | Pink wax tealight in metal cup | IMG_4826, IMG_4828-IMG_4829 | 3 | generic candidate; brand, scent, pack SKU need confirmation |
| IL-NEW-005 | Dobro Mogador oatmeal, chocolate, 65 g | IMG_4830-IMG_4833 | 4 | packaging/variant/size visible |
| IL-NEW-006 | Dobro Mogador rice mash, chocolate & banana, 50 g | IMG_4834-IMG_4837 | 4 | distinct from IL-NEW-005; packaging/variant/size visible |
| IL-NEW-007 | Einhorn-branded glue stick, 9 g | IMG_4838-IMG_4841 | 4 | product type and net weight visible; confirm exact sellable item/SKU |
| IL-NEW-008 | EDEKA cylindrical batteries, likely AA / 1.5 V | IMG_4842-IMG_4846 | 5 | brand/type mostly visible; confirm cell size/chemistry/pack SKU from label |
| IL-NEW-009 | Plain red disposable plastic cup | IMG_4847, IMG_4849-IMG_4852 | 5 | generic candidate; exact size/material/SKU not visible |
| IL-NEW-010 | Small gold-colored crown-shaped object | IMG_4853-IMG_4855 | 3 | object is visually distinct; confirm if it is an inventory SKU and its identity |

The counts sum to 42. Reference counts are photo counts, not necessarily distinct/useful viewing angles. Front/back/side views are evident for some packaged products, but the photos do not contain trusted angle labels. Review and label angle proposals before angle-based claims. Three-photo groups cannot support a four-reference gallery; for leave-one-out testing their maximum usable gallery is two.

## Products in employee and customer scenes

Visual inspection of the 25 employee and 5 customer photographs shows the same 10 candidate product classes represented in `Owner_test`; **no additional, clearly distinct product variant lacking an owner reference was apparent**. This is a visual audit, not exhaustive SKU/UPC verification. Exact commercial variants remain unresolved for the three generic candidates and the battery label details above. Do not silently mark an unrepresented/ambiguous item as registered: keep it `unknown` pending review.

## Same-arrangement and multi-view candidates

| Candidate group | Images | Audit judgment |
|---|---|---|
| Employee mixed test layout A | IMG_4876, IMG_4878 | Strong same-arrangement candidate: same 4 tealights, 4 glue sticks, and approximately 7 batteries in corresponding layout; likely camera/view change. Confirm before multi-view instance linking. |
| Employee mixed test layout B | IMG_4892, IMG_4893, IMG_4895 | Strong same-arrangement candidate: same apparent four cups, four crowns, four glue sticks, and four tealights; framing/view changes while the layout is substantially retained. Confirm before linking individual products. |
| Customer basket series | IMG_4897, IMG_4898, IMG_4900, IMG_4902, IMG_4903 | Same basket/background and overlapping product assortment, but object positions/orientations visibly change. Treat as separate, reconfigured scenes, not unchanged angles. Do not fuse counts. |
| Employee cup-only layouts | IMG_4880, IMG_4882, IMG_4884 | Same cup SKU, but count/layout changes (four, four, nine visible); not a confirmed unchanged multi-angle group. |
| Employee crown-only layouts | IMG_4887, IMG_4889 | Same crown candidate, but count/layout/framing differ; not a confirmed unchanged multi-angle group. |
| Other employee shelf/table scenes | IMG_4857-IMG_4874 and IMG_4880-IMG_4889 | Product sets and counts/layouts vary. Some share the same physical props, but no further unchanged-angle pair is sufficiently certain from visual evidence alone. |

The two strongest employee groups are *provisional*. No other photo group should be treated as multi-view ground truth until you confirm unchanged physical objects/arrangement. Full proposed groups are also listed in [arrangement_groups.csv](arrangement_groups.csv).

## What can be measured with these photos

| Experiment | Feasibility now | Boundary / required review |
|---|---|---|
| Registration without retraining | Feasible for 7 package-identifiable candidates; 3 need identity/SKU confirmation | Approve candidate names, variants and reference-to-SKU mapping first |
| Reference count 1/2/3/4/all | Feasible as a within-reference, leave-one-photo-out retrieval experiment | Query image must be excluded from its gallery. Each fixed query uses nested/randomized gallery subsets. A four-reference gallery is only possible for SKUs with at least five usable references when holding one query out. Three-reference SKUs max out at two. This tests controlled owner photos, not future-scene generalization. |
| Reference angle combinations | Partly feasible | First review angle labels; missing/unclear angles must not be inferred into results. Several refs are redundant or show only a narrow viewpoint range. |
| Original vs masked reference ablation | Feasible after mask review | Generate mask proposals lazily, approve/reject each, and compare paired queries with the same held-out photos/gallery counts. Mask failures must be reported separately. |
| Employee detection/counting and SKU accuracy | Feasible after review labels | All 25 photos need per-instance labels/counts, and likely SKU labels. The two multi-view groups need confirmation. Current directories contain no verified object labels. |
| Customer basket recognition/counting | Feasible as per-image accuracy after review | Review the 5 images for per-object identity, occlusion category and visibility suitability. The basket shots are rearranged; no count fusion across them. Exact visible-area percentages require masks/geometry and are not supported yet. |
| Maximum products / SKU diversity / crowding | Descriptive exploratory analysis only after labels | Current scene count is small and factors are confounded; do not claim a general limit from five customer photos. |
| Multi-angle recovery | Limited, potentially feasible on the two employee candidate groups | Confirm unchanged arrangement and instance correspondence; compare per-view results first and count a physical instance once. Customer series is not a clean angle experiment. |
| Customer/employee camera guidance | Feasible as a small pilot after verified labels | Can derive conservative prompts from actual misses/uncertainty; 5 basket images are not enough for a general visibility threshold. |

No scene photo currently has verified detection boxes, per-object SKU labels, visibility classes, or masks in this new experiment. Therefore no detector/recognition accuracy, count error, mask IoU, reference-count effect, or angle benefit is reported by this audit.

## Minimum human confirmations / additional captures

1. Confirm whether the tealights, plain red cups, and crown objects are inventory SKUs; confirm the battery type from a legible label or package. Confirm that the 10 proposed identities/variants match intended sellable units.
2. Confirm the provisional same-arrangement groups above, especially whether the exact products and layout remain unchanged in IMG_4876/IMG_4878 and IMG_4892/IMG_4893/IMG_4895.
3. In the review workflow, confirm scene product counts and each detection as correct single item, wrong SKU, unknown, missed, merged, or not assessable. Review approximate visibility as coarse human categories only.
4. For reference-angle coverage gaps, capture a clear front, side, and back photograph on a consistent neutral setup for any candidate whose current gallery lacks those views. Additional photographs are not needed to run the limited 1-4 photo leave-one-out test where enough references exist, but would be needed for missing-angle comparisons.
5. For generalization and crowding limits, collect additional **independent** shelf arrangements and baskets with reviewed instance labels; split by arrangement, not by adjacent image. Keep a final arrangement holdout out of calibration.

## Local execution constraints and next workflow

The local project has `data/models/mobile_sam.pt` and a tested MobileSAM/Ultralytics code path, plus cached SigLIP2 weights in `.venv/huggingface_cache`. Local PyTorch is CPU-only (`torch 2.14.1+cpu`; CUDA unavailable). Thus an isolated, local review app can generate masks and SigLIP features on CPU, but it will be slower than the previous RunPod experiments. No paid GPU is needed for the audit or a small pilot. The prepared app must write only under this experiment directory, preserve raw source paths/hashes, and never call production registration/database APIs.

