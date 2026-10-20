# Catalogue-growth robustness test, 2026-10-20

Does the SKU proposed for an already-detected box stay stable as the registered
catalogue grows? A shop's catalogue only ever gets bigger, so this is an
operational question, not an academic one.

**This is a rank-stability measurement, not an accuracy measurement.** No ground
truth is used anywhere in it. The ten `IL-NEW-*` identities are unapproved and
the scene labels unreviewed, so no accuracy, recall, precision or count-error
figure is stated or implied here.

## Method

The published 30-scene run's own query embeddings were reused, so the detector
proposals and the query crops are **identical in every stage** - only the
gallery changes. That isolates the effect of catalogue size.

- Queries: the 93 detector box crops from the 30 employee/customer scenes,
  taken from `imported_runpod_scene_results_20261020_v5_flat/embeddings.npz`.
- Per-SKU score: the mean of the two highest cosine similarities for that SKU,
  matching the runner's rule exactly.
- Encoder: `google/siglip2-base-patch16-224`, the same model as the run, on CPU.
  No GPU was used and nothing was spent.

Stages, cumulative in the order the folders were captured:

| stage | gallery | SKUs |
| --- | --- | --- |
| A | `01_` only | 3 (`TEST-*`) |
| B | `01_` + `02_` | 9 (`+ ILX-*`) |
| C | `01_` + `02_` + `03_` | 19 (`+ IL-NEW-*`) |
| D | `03_` only - the published baseline | 10 |

## Result 1 - the ranking is stable under catalogue growth

Going from the published 10-SKU gallery (D) to 19 SKUs (C) - nearly doubling the
catalogue with nine unrelated products - changed the top-1 proposal on
**1 of 93 boxes. 92 of 93 (98.9%) were unchanged.**

The single change is itself informative:

| | top-1 | similarity | margin to runner-up |
| --- | --- | --- | --- |
| baseline, 10 SKUs | `IL-NEW-001` | 0.776 | 0.131 |
| grown, 19 SKUs | `ILX-005` | 0.791 | **0.015** |

`IMG_4902.jpeg:det-1` flipped to a distractor that beat it by 0.015 - the system
is effectively undecided there. That is precisely the case the human review step
exists to catch, and it is visible in the data without any ground truth.

Confidence compresses only slightly as the catalogue grows: mean top1-top2
margin 0.176 (10 SKUs) to 0.156 (19 SKUs).

## Result 2 - the real weakness is abstention, not catalogue size

Stages A and B contain **no correct product at all**: the products in the scenes
were not registered in those galleries. The system named one anyway, on every
single box.

| stage | gallery | mean top-1 similarity | mean margin | boxes with margin < 0.05 |
| --- | --- | --- | --- | --- |
| A | 3 SKUs | 0.559 | 0.124 | 6 / 93 |
| B | 9 SKUs | 0.634 | **0.031** | **79 / 93** |
| C | 19 SKUs | 0.831 | 0.156 | 8 / 93 |
| D | 10 SKUs | 0.831 | 0.176 | 7 / 93 |

In stage B, 79 of 93 boxes have a top-1/top-2 gap below 0.05 - the ranking is
close to arbitrary - yet every box still receives a confident-looking label.
This is the documented consequence of there being **no calibrated unknown
rejection threshold**: `recognition_status` on every row of the published run
reads `unaccepted_provisional_rank_no_unknown_threshold`.

The practical reading: catalogue growth is not the risk. Presenting a rank as an
answer when the product was never registered is the risk, and the only thing
standing between that and a wrong stock figure is the human approval boundary.

## Limits of this test

- **No ground truth.** Stability is not correctness. A proposal that stayed
  stable across stages may still be wrong.
- **Reference asymmetry.** The `03_` references are tight owner-approved crops;
  the `01_` and `02_` references are whole reference photographs. That
  difference is a confound and could favour either set. A matched comparison
  would require approving crops for `01_` and `02_` first.
- **Nine distractors is a small increase.** It does not establish behaviour at
  hundreds or thousands of SKUs.
- One `02_` SKU (`ILX-005`) has a single reference photograph, so its score is
  that one cosine rather than a mean of two.

## Files

- `catalogue_growth_summary.json` - per-stage aggregates and distributions.
- `catalogue_growth_per_query.csv` - 372 rows, 4 stages x 93 queries, with
  top-1, top-2, similarities and margins.
