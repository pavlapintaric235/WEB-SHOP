# Employee photo test review — 2026-10-24

Ground-truth counts below came from the employee after reviewing the existing five photos. No photos were rescanned and no inventory was changed.

| Photo | User-confirmed products and units | Detector / SKU review |
|---|---|---|
| 1.pdf | Freshona beans 2; GardenGood Passata 3; dmBio curry 2 (7 units) | 5 proposals: curry 2, Passata 1, beans 1, and 1 unknown. It undercounts the known SKUs by 3 units. |
| 2.pdf | Freshona beans 1; Energia 2; Fentimans 3 (6 units) | 6 proposals with the same SKU totals. This is the strongest of the five reviews. |
| 3.pdf | Freshona beans 3; dmBio curry 2; Energia 2; Fentimans 3; GardenGood Passata 3 (13 units) | 15 proposals. The overlay has overlapping boxes and unknown labels; do not treat proposal count as item count. SKU matching and box placement both need improvement here. |
| 4.pdf | Dukat milk 2; Fentimans 2; Energia 1; dmBio curry 2 (7 units) | 7 proposals, with Energia 1 and Fentimans 2 assigned. Four proposals are unknown, including boxes over milk and curry products. |
| 5.pdf | Energia 2; Fentimans 1; Oreo 1 (4 units) | The sheet's assigned SKU totals match your labels. |

## Combined user-confirmed totals

| SKU / product | Units across the five photos |
|---|---:|
| Freshona beans | 6 |
| GardenGood Passata | 6 |
| dmBio curry | 6 |
| Energia | 7 |
| Fentimans | 9 |
| Dukat milk | 2 |
| Oreo | 1 |
| **Total** | **37** |

## What this says so far

- The same-SKU rule should produce one shelf-count row per SKU and sum the reviewed unit counts. The per-box view can remain separate so staff can inspect, correct, or exclude each detection.
- Photo 2 supports the live SKU matching approach for a clear, well-separated shelf image.
- Photos 1 and 4 show useful box proposals but missed unit counts and/or SKU assignments.
- Photo 3 is crowded enough that the overlapping boxes make its raw proposal count unreliable.
- Photo 5's sheet totals match your labels.

## Source and scope

- Human labels: [ground_truth_counts.json](ground_truth_counts.json)
- Source PDFs: `<LOCAL_FILES>/1.pdf` through `5.pdf`
- Labels are evaluation annotations only. They have not been written to the demo inventory database.
