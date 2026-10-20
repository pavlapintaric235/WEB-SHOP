# Recognition Audit And Photo Pipeline

## Findings

The installed Ultralytics 8.4.177 implements the documented YOLOE visual-prompt API:
`bboxes` and sequential numeric `cls` indices describe objects in `refer_image`.
Reference embeddings are applied before prediction on the separate target image.
Our use of that API was compatible; the saved broad false predictions do not
establish that the API was missing or that K18 cannot be recognized.

Code inspection identified additional application-level problems:

- Center-distance deduplication could merge adjacent, non-overlapping tall bottles.
- Rejecting boxes above 35% of image area removed valid close-up products.
- Cross-SKU deduplication selected the higher score even for conflicting identities.
- The OpenCV fallback used a fixed heuristic threshold unrelated to the UI's YOLOE slider.
- The fallback forced the best match without a runner-up ambiguity check.
- YOLOE masks were discarded, so boxes could not provide true segmentation highlights.
- Repeated checks on annotated outputs could contaminate input images with drawn outlines.

The comparison path now retains masks, validates numeric class mapping, preserves
adjacent instances, allows valid close-ups, and leaves conflicting prompt labels
unassigned. The original OpenCV matcher is retained only in the experimental
YOLOE comparison path. It is not SAM and is not the new default.
A fresh real YOLOE comparison on the fabric photo completed without an API error
but proposed no assigned K18 units and two conflicting/unassigned regions.

## New Default

1. MobileSAM proposes instance masks without assigning product classes.
2. Background/scene masks and very small regions are logged as rejected.
3. Each registered reference is also segmented; one product per reference is assumed.
4. Masked crops are padded to preserve aspect ratio and embedded with pretrained
   torchvision ResNet18 features before its classification layer.
5. Normalized embeddings are compared by cosine similarity. The initial acceptance
   threshold is 0.80, with a 0.05 gap from the second-best SKU.
6. Weak or ambiguous regions remain unassigned. They do not contribute to SKU totals.
7. Mask overlap/containment removes duplicates and nested cap/label regions. A parent
   mask enclosing multiple independently matched instances is not another counted unit.
8. Real masks drive green recognized outlines, yellow unassigned outlines, and the spotlight.

The original image dimensions are preserved. SAM operates on a working image with
longest side at most 1280 pixels; masks are mapped back using nearest-neighbor
resizing. This is model segmentation at working resolution, not full-resolution
pixel-perfect ground truth. Original photographs remain available for review.

The model is pretrained. No custom training has occurred. This architecture is a
testable improvement, not a guarantee of arbitrary-SKU recognition from one example.
ResNet18 is a small general-purpose feature baseline, not a specialist retail-SKU model.
Single-SKU catalogues cannot use the runner-up check and rely on similarity alone,
even when that SKU has multiple reference views.

## Multiple Reference Views

Owners can now register multiple independent photos under one SKU and add/remove
views from existing products. Existing photos are backfilled into a separate
SQLite reference table without changing their paths or inventory history. At
least one and at most four active photos are retained per SKU in the current
retake trial. Three views plus an optional fourth are recommended while the
minimum remains flexible. Removed photos
leave the active catalogue but remain on disk for historical diagnostics.

SAM and the same frozen ResNet18 encoder process every active view. A candidate's
SKU score is the maximum similarity over that SKU's views. The runner-up is a
different SKU, not another photo of the same product. Weak and ambiguous matches
remain unassigned at the unchanged 0.80 / 0.05 defaults. Best-reference provenance
is saved in diagnostics, and reference additions/removals invalidate pending
scan review. The legacy YOLOE comparison still uses only the primary photo.

The first real regression check used the latest mixed scene and the same original
three references: coconut water 2, K18 2, WOW 0, with 20 unassigned regions. This
preserves the prior result but does not resolve WOW identity. Report:
`data/recognition_checks/20261014-220007-e17b64b5/report.json`.
No test-scene crops have been added as references. New independent WOW views
still need to be supplied before measuring any benefit. Best-view matching can
also increase false matches as coverage grows, so absent-SKU and similar-package
tests remain necessary. Automated multi-view tests use synthetic features and
temporary databases; they establish implementation contracts, not retail accuracy.

## Grocery Reference Selection Correction

The grocery run `20261014-232623-b9edd35b` exposed a reference-selection bug:
the Fentimans reference cached a 959 by 421 tabletop crop with the bottle masked
out. The "largest valid mask" rule allowed that table because its area was below
the old scene-mask threshold. The scene's table then received a Fentimans SKU.

That correction made reference selection and scene matching reject masks covering at least
10% of three frame edges. Coverage is measured within a 1% border band (at least
two pixels), since SAM can leave a one-pixel gap at the frame. Single/two-edge
cropped products and large enclosed close-ups remain eligible. True products
cropped heavily at three edges may be rejected as well; this is deliberately
conservative, and not a general semantic-background classifier. No eligible
reference mask produces an explicit error rather than a background fallback.

That correction used `mobile-sam12-resnet18-v2-border-foreground`. Older
disk and in-memory reference entries are not reused. The app's cached model and
pending-analysis signature include this version, so previous results must be
rerun rather than approved using stale reference processing. Original reference
photos, catalogue records and historic test logs are unchanged.

Real inference rebuilt all five grocery reference crops. The Fentimans crop is
now a 310 by 1105 full bottle, verified visually in the reference board. The same
original mixed-scene photo was rerun at the unchanged 0.80 similarity / 0.05 margin:

| Product | Expected | Before | After |
| --- | --- | --- | --- |
| Energia | 4 | 1 | 1 |
| Fentimans | 3 | 1 (table region) | 2 (separate bottle masks) |
| Jana | 2 | 1 | 1 |
| Maks Fripsy | 0 | 0 | 0 |
| Stobi Flips | 0 | 0 | 0 |

The original table proposal is rejected as `background_or_scene_mask`, with no
SKU. The back-row undercount remains unresolved. This is a development regression
check, not an independent accuracy evaluation or a complete counting fix.
The cold check, including five rebuilt references, took about 200 seconds while
the automated suite was also running. It did not overwrite the user's app test.
Report: `data/recognition_checks/20261015-110327-b9836a89/report.json`.

All 56 automated tests passed, including synthetic border-contact, close-up,
reference-selection, legacy-cache invalidation, ambiguity and workflow checks.

## Multi-View Cardboard Regression

The later 14-SKU / 50-view run `20261015-172409-154be5df` contained one Jana
bottle. It correctly counted Jana once but also assigned Fentimans to the
cardboard back wall and Dukat to the cardboard base. The v2 three-edge rule
missed these two-edge background masks, both in reference preparation and in
the test scene. Their similarities were high because the cached references
contained cardboard with the product cut out.

All 50 v2 crops were visually reviewed. Nine background crops were found:
reference IDs 278, 286, 293, 326, 355, 370, 371, 390 and 393, across Energia,
Fentimans, Dukat, Fripsy, Croco, Oreo and 7Days. Audit, source hashes and complete
before/after contact sheets are in
`data/recognition_checks/reference-foreground-repair/`. Forty-one reviewed
product crops and their existing features were explicitly preserved in the
new cache namespace, rather than needlessly segmenting them again. Reference
436 remains a partial side-label view of the beans can, documented in the audit;
it is not represented as a complete-object mask. Nine bad views were freshly
segmented from their original registration photos. No source photos, products,
reference registrations or approved stock quantities were changed.

Reference preparation now has its own stricter policy: broad contact with even
one frame edge (at least 10% coverage within the existing narrow border band)
makes a candidate ineligible. Registration photos must contain a whole product
with background on every side. The largest remaining eligible mask wins, and
no eligible mask gives an explicit error. This is a conservative geometry
constraint, not semantic verification that every selected mask is a whole
product. Scene matching keeps its previous close-up policy, so one/two-edge
cropped scene objects are not indiscriminately discarded.

The current cache identity is `mobile-sam12-resnet18-v3-bounded-reference`.
Production code does not automatically accept v2 files or in-memory entries.
Pending app analyses and model resources are also invalidated by this version.
The one-time, selective transfer above is recorded as a manual visual audit,
not a generic legacy-cache migration.

Real MobileSAM / ResNet18 inference reran the exact Jana photo twice, with
all 14 SKUs / 50 views and unchanged 0.80 similarity / 0.05 runner-up gap:

| Product | Expected | Before | After (Both Runs) |
| --- | --- | --- | --- |
| Jana | 1 | 1 | 1 |
| Fentimans | 0 | 1 | 0 |
| Dukat | 0 | 1 | 0 |
| Every other registered SKU | 0 | 0 | 0 |

Nine-view repair plus the first check took 357.41 seconds including artifact
saving; reference work accounted for 316.945 seconds while the automated suite
also ran. The warm repeat took 41.10 seconds including diagnostic images:
0.095 seconds for references, 27.581 seconds for photo segmentation and 30.741
seconds total inside the recognizer. Model loading and app snapshot writing are
outside that recognizer total. Six uncertain regions remain; they include
background and bottle parts, not six extra products. These checks fix a known
regression, not establish general retail accuracy or video readiness.

Report: `data/recognition_checks/20261015-175250-082304b1/report.json`.
Original app run/history were preserved. The scanner now displays reference
progress, named analysis stages and measured timings, which are also saved in
new diagnostic snapshots. Tests cover the stricter reference/looser scene
policies, v2 disk/memory invalidation, progress, timing persistence and ensuring
that running detection alone does not approve inventory.
All 64 automated tests passed after the correction; Python compilation also passed.

## Geometric Label Verification

The next registered-grocery batch exposed an embedding ambiguity problem rather
than another cardboard-reference problem. In `20261015-184339-38f870ec`, three
beans masks scored 0.856-0.888 against Freshona but their gaps from DM stew were
only 0.021-0.040. All were correctly withheld by the unchanged 0.05 embedding
gap rule. In `20261015-184518-e8a66353`, a third Passata candidate had a 0.045
gap from Croco and was likewise withheld, leaving only two counted cartons.

`local_features.py` adds an independent label-detail check only when the cosine
minimum passes but the embedding gap is ambiguous. OpenCV SIFT descriptor
matches use a 0.7 nearest-neighbor ratio test, unique point locations, and a
RANSAC homography with a 3-pixel reprojection tolerance. Verification requires
12 inliers, 35% inlier support and 8% convex-hull coverage in both crops, plus a
finite, convex, non-reflected projection with plausible area. The support and
coverage checks avoid treating a few repeated letters or duplicate SIFT
orientations as independent product evidence. These are trial heuristics, not
calibrated accuracy guarantees. Curved labels may have multiple locally planar
regions, so a homography need not explain all descriptor matches.

For ambiguous embeddings, all views of the displayed top three and any further
SKU within the configured gap are compared. Exactly one SKU must verify, and it
must agree with the embedding winner. A separate recovery handles masks whose
embedding winner scores at least 0.70 but remains below the configured 0.80
minimum, provided the mask covers at least 0.005 of the image. It compares all
registered SKUs and accepts only a unique verified label match; a geometric
match may resolve a different SKU than the embedding winner. Conflicting SKUs,
absent details and other below-minimum embeddings remain unknown. Multiple
verified views of one SKU do not count as votes or extra objects.
This catalogue-wide search runs only when the top-score gap is also within the
configured ambiguity margin, to avoid expensive label checks on clear but weak
fragments.
Existing instance-mask pruning still determines the unit count.

Resolved records retain their original cosine similarity and failed embedding
gap, with `reason=geometric_reference_match`, the original embedding reason and
separate per-view verification/provenance diagnostics. The UI exposes label
verification status and inlier count. The original 0.80 / 0.05 controls were not
retuned. This is a second decision path, not a claim that the primary embedding
gap passed. Ordinary, clearly separated embedding matches keep their previous
behavior, including their existing open-set misclassification risks.

The combined recognition version is now
`mobile-sam12-resnet18-v3-bounded-reference:resnet18-sift-ransac-v3`. It versions
app model resources, pending analyses and new snapshots, independently of the
unchanged v3 reference crop/embedding cache. No model weights, source photos or
registered references were replaced, and no test-scene crop became a reference.

Two complete fresh MobileSAM / ResNet18 / SIFT checks, with all 14 SKUs / 50
views and unchanged embedding thresholds, produced:

| Scene | Expected | Before | After | Extra SKUs |
| --- | --- | --- | --- | --- |
| Three Freshona cans | 3 | 0 | 3 | 0 |
| Three Passata cartons | 3 | 2 | 3 | 0 |

Both spotlights were visually checked for three distinct physical instances.
Reports: `data/recognition_checks/20261015-191113-4d63d626/report.json` and
`data/recognition_checks/20261015-191438-94b02b56/report.json`.
The complete checks took 68.92 and 42.77 seconds including saved diagnostic
images, excluding model load; the beans run overlapped automated tests. This is
not a controlled latency benchmark, and label verification adds CPU work.

`check_matching.py` separately replays original real SAM masks, guarded by source
image content and resolution, to test matching without recomputing segmentation.
All 17 saved grocery runs had correct counts in the replay after two explicitly
reviewed annotation corrections: the Energia-only photo had accidentally kept
Jana=1, and the Mivolis candy photo expected DM stew. These corrections are
separate comparison-only overrides in
`data/recognition_checks/label-verification-truth.json`; original app annotations,
latest snapshot and history remain untouched. All previously correct counts
were preserved. Replay report:
`data/recognition_checks/20261015-192217-80507379-matching-replay/report.json`.
Replay did not independently test segmentation, and these known regression
photos are not an unseen accuracy benchmark.

All 84 automated tests passed, including real SIFT on synthetic perspective
transforms, unrelated/featureless images, tiny-patch/reflection/collinearity
rejection, cross-SKU conflicts, below-threshold abstention, fourth-candidate
coverage, safe mask replay and matcher-version invalidation of pending approval.
Fresh mixtures and negative examples remain necessary. Whole-object highlights
can still be partial, label-poor products can fail, and unregistered products
are not guaranteed to remain unknown.

### Follow-up Scenes - 2026-10-15

Seven newer owner scans were replayed against their saved SAM masks after the
geometry-only recovery was added. The clean Passata-three and Dukat-four scenes
remained exact. One Dukat scan expected four while its saved photo visibly shows
two bottles, so that expected count was incorrect. The other scenes still expose
instance segmentation limits: Mivolis candy remained 3/6, a three-carton
Passata stack remained 1/3, and the mixed milk/Passata photo remained 1/2 milk
and 3/3 Passata. The crowded nine-item scene improved from four to five: one
stew can was recovered from a mask with a unique 27-inlier label match across
all 14 registered SKUs; Stobi, curry and Passata remain undercounted there.

A denser SAM proposal grid raised the Mivolis result only from 3/6 to 4/6 and
raised analysis time to about 95 seconds, so it was not enabled. The existing
YOLOE visual-prompt path returned two broad regions on that candy photo, so it
was not substituted either. The remaining failures need better object-instance
separation and clearer visible product evidence. They were not addressed by
lowering the cosine threshold. Replay report:
`data/recognition_checks/20261015-221008-e6dc7478-matching-replay/report.json`.

## Development Evidence

Two previously seen photos, manually verified to contain two K18 bottles each,
were processed with real MobileSAM and pretrained ResNet18 inference at the same
default matching thresholds:

| Scene | Expected K18 | Proposed K18 | Unassigned regions |
| --- | --- | --- | --- |
| Clean black background | 2 | 2 | 0 |
| Dark fabric background | 2 | 2 | 6 |

The fabric regions were not assigned to coconut water or WOW. Their presence is
evidence that class-agnostic segmentation also proposes background regions.
In a separate negative-catalogue check, K18 was excluded from the comparison
references without changing the actual database. The two-K18 photo then produced
zero assigned SKUs and five unassigned regions, rather than false coconut/WOW counts.
These are development checks on previously used images, not independent accuracy
measurements. The original source hashes and source-code hashes are recorded in
the generated reports. Historic app tests are not rewritten.

Initial reference processing plus the first photograph took about 107 seconds;
the next photograph took about 39 seconds on the current CPU. Disk reference
caching now avoids repeating reference work on later process starts.
Throughput is unsuitable for a live video promise at present.
The real employee Streamlit view was also exercised on the fabric photo: two K18
units, six unassigned regions, actual mask rendering at 4032 by 3024 pixels, and
successful diagnostic serialization. No inventory approval was submitted during
that check. Automated tests cover the separate approval and permission workflow.

Next evaluation should use separate, previously unseen photos with manually
recorded per-SKU quantities. Include mixed registered products, absent registered
SKUs, unregistered objects, similar packages, zero counts, lighting changes, and
different backgrounds. Report false matches and unassigned regions alongside
per-SKU errors. Change thresholds using development data only; then freeze them
for an independent test set. No detection-accuracy percentage is claimed here.

## Sources

- [Ultralytics YOLOE visual prompting](https://docs.ultralytics.com/models/yoloe/)
- [Ultralytics MobileSAM](https://docs.ultralytics.com/models/mobile-sam/)
- [Torchvision pretrained ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
