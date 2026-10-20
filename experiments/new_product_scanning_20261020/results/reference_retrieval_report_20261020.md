# New Product Reference Retrieval Pilot

> **Historical run.** This report records the earlier `20261019T230100Z` full-photo evaluation, before owner masks were approved. For the newest paired box-versus-mask result on 34 owner photos, open the generated review page `latest_recognition_review_20261020.html`, held locally and available on request, or the frozen [summary CSV](siglip_reference_retrieval_20261019T233831Z/summary.csv). On that same 34-photo cohort, box top-1 was 24/34 and masked top-1 was 25/34; one oatmeal query changed from a wrong rice-mash suggestion to the correct provisional group. The 42-photo full-image result below uses a different cohort and must not be compared directly with those 34-photo figures. No new employee/customer scene recognition results have been imported.

Run: `siglip_reference_retrieval_20261019T230100Z`  
Model: cached `google/siglip2-base-patch16-224`, CPU  
Status: exploratory only; no scene-level accuracy claim

## Dataset and confirmation state

The audited folders contain 72 photos: 42 owner references, 25 employee scenes, and 5 customer baskets. The owner folder has 10 provisional visual groups. At evaluation time, **zero of the ten identity mappings had been owner-confirmed**. Therefore these numbers are retrieval behavior against provisional filename-to-SKU groupings, not verified SKU accuracy.

No reference masks had been approved. The evaluation used 42 original owner photos only (`box` and `masked` representation coverage: zero). One MobileSAM proposal-generation smoke test produced 17 candidate masks for `IMG_4810.jpeg`; they remain unreviewed and are not part of these metrics.

## Method

- Leave one owner photo out as query; that photo is excluded from its own SKU gallery.
- For the 1-4-reference curve, use the **same 16 query photos** at every gallery size, from the three provisional groups with at least five owner references: `IL-NEW-002`, `IL-NEW-008`, and `IL-NEW-009`.
- Enumerate all target-SKU reference subsets of each requested size. Rival SKUs retain all their owner references. Micro accuracy weights every query/gallery-subset trial; macro accuracy first averages the subsets for each query, then averages queries.
- Report “all available” separately over 42 queries and 10 candidate groups. It has a different query cohort and must not be compared as a paired point on the 1-4 curve.
- No acceptance threshold was fit or lowered. Abstention, unknown-SKU rejection, and scene queries were not evaluated.

## Results

*Plot of SigLIP2 owner-photo retrieval accuracy by gallery size, `reference_gallery_curve_20261020.png`, held locally and available on request. The same figures are tabulated below.*

| Target gallery | Fixed query photos | Candidate groups | Top-1 macro by query | Top-1 micro by trial | Top-5 micro |
|---:|---:|---:|---:|---:|---:|
| 1 reference | 16 | 3 | 88.8% | 87.1% | 92.9% |
| 2 references | 16 | 3 | 94.4% | 92.5% | 98.3% |
| 3 references | 16 | 3 | 98.1% | 97.0% | 100.0% |
| 4 references | 16 | 3 | 100.0% | 100.0% | 100.0% |

Separate all-available leave-one-out result: **78.6% top-1 and 100% top-5** across 42 queries and 10 provisional groups. Nine queries had a wrong top-1. The main confusion was `IL-NEW-005` oatmeal versus `IL-NEW-006` rice mash (seven queries in the all-available run); other observed confusions were coconut water to rice mash and tuna salad to the red-cup group.

The 1-4 result is a useful signal that additional owner views can help distinguish these controlled reference photographs. It does **not** establish that four photos are sufficient in shelf/basket scenes: the common cohort is small, was captured in a controlled setup, includes unconfirmed identities (including the generic cup group), and has no independent-scene queries. The 100% four-reference score is not a deployment accuracy estimate.

Machine-readable details: `siglip_reference_retrieval_20261019T230100Z/report.json`, `summary.csv`, and `per_query.csv`. The source manifest hash is recorded in `report.json`.

## Not measured yet

- Employee/customer detection, SKU accuracy, physical counts, merged/missed objects, and per-scene failures: no reviewed per-instance labels or new-scene model predictions yet.
- Original-box versus masked-reference benefit: pending owner approval of masks.
- Unknown-SKU rejection: no reviewed genuinely unregistered objects in the reference-only test.
- Mask IoU, multi-angle recovery, and useful camera guidance: pending verified masks, scene labels, and unchanged-arrangement/instance correspondence review.

## Next minimum steps

1. Confirm the ten candidate identity groupings, with particular attention to tealight, red cup, crown, battery subtype, and exact glue-stick SKU; confirm front/back/side angle notes.
2. Approve/reject reference masks in the local app. Then rerun the paired photo/box/masked evaluation on the exact shared query set available across representations.
3. Review all 25 employee and 5 customer images using separate boxes per physical item, including missed items and `__unknown__` labels. Confirm whether `IMG_4876/4878` and `IMG_4892/4893/4895` preserve the same physical arrangement.
4. For a uniform four-reference test with one held-out owner query per SKU, the five groups with four references need one additional reference each, and the two groups with three need two each: **nine more reference photographs minimum**. This count alone does not ensure useful angle diversity; capture front/side/back views where missing.
5. Keep calibration arrangements separate from final held-out arrangements. Only then run the NVIDIA Retail DINO + SigLIP2 scene evaluation and derive camera guidance from actual misses/uncertainty.

