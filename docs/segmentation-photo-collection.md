# Grocery Segmentation Photo Collection

## Goal

Collect about **47 distinct grocery scenes** for a small pilot with the current 14 registered SKUs. This is enough to exercise the annotation workflow and expose obvious failure modes, not enough to promise production-ready accuracy. Aim for roughly 4-8 visible packages in most mixed scenes and make sure every SKU appears repeatedly across different arrangements. Do not take 47 near-identical photographs of one shelf.

The detector has one class, `product`. Outline every visible physical package as its own instance, including an unregistered packaged product if one appears. The detector does not learn SKU names; the existing reference-photo matcher remains responsible for SKU matching. Completely hidden products are not annotated.

## Suggested 47-Scene Plan

| Scene type | Photos | What to vary |
|---|---:|---|
| Front-facing shelf layouts | 8 | Arrange 4-8 different SKUs; use different shelf positions, spacing and order. |
| Products directly beside/touching | 7 | Mix cartons, bottles, cans and snack bags. Include pairs with similar colors as well as clear gaps. |
| Repeated units of one SKU | 6 | Put 2-4 copies of the same package in one frame, then rotate which SKU is repeated. Keep each unit separately visible. |
| Mixed package types and depths | 6 | Combine cartons, bottles, cans, jars, boxes and flexible snack bags; include front and rear rows. |
| Camera height and angle | 6 | Two low-but-readable views, two raised views, and two modest left/right oblique views. Keep the products recognizable and fully in frame in most of these. |
| Warm indoor lighting | 4 | Use ordinary warm room light; vary shadow direction and intensity without using flash. |
| Background and daylight | 4 | Two indirect-daylight scenes on a simple background, one ordinary shelf, and one counter/table. Avoid changing every condition at once. |
| Partial occlusion or frame edge | 4 | Include visible products partly behind another item or partly cut by the photo edge. Keep enough visible pixels to trace the visible outline. |
| Empty/unrelated negative scenes | 2 | Photograph a shelf/table with no packaged goods, optionally with non-product objects. If a packaged item is present, annotate it as `product` even if its SKU is unknown. |
| **Total** | **47** | |

For the current catalogue, rotate coverage across: `7DAYS-CAKE-30`, `CROCO-CRACKERS-100`, `DM-CURRY-325`, `DM-EINTOPF-400`, `DUKAT-MLIJEKO-1000`, `ENERGIA-POWER-250`, `FENTIMANS-GINGER-200`, `FRESHONA-BEANS-800`, `GARDENGOOD-PASSATA-200`, `JANA-ANTIOX-500`, `MAKS-CRISPY-50`, `MIVOLIS-BOMBON-3`, `OREO-ENROBED-41`, and `VITAMINKA-STOBI-30`. A useful tracking target is at least 8 clear appearances for each SKU, across at least two backgrounds and two different neighboring-product combinations. Check the table as you go; it is a collection target, not an accuracy guarantee.

## Capture Rules

- Use the original camera file; do not use screenshots, crops, AI edits, drawn boxes or exported previews as training photos.
- Keep most products large enough to inspect, in focus, and with their physical boundaries visible. Keep a few deliberate edge/occlusion examples, but do not make every image difficult.
- Photograph real mixtures and realistic spacing. Do not place a large shadow or cardboard surface where it could be mistaken for a package boundary.
- In repeated-SKU scenes, count the physical units before the photo and ensure each visible unit has its own outline. Same SKU does not mean same instance.
- A partially hidden product is labeled only over its visible pixels. Do not guess the hidden section. Do not label products that are completely hidden.
- Take some photos on a shelf and some on a counter or plain matte background. Avoid using one clean studio background for the entire dataset.
- Keep most products upright. Include some reasonable rotations and side views, not only front labels.
- Avoid severe blur, direct flash, hard glare over the package edge, portrait-mode blur, and extreme ultrawide distortion. Keep a few realistic hard cases, but make the outlines genuinely reviewable.
- For actual empty negatives, make sure no packaged products are visible. Any visible packaged item belongs to the generic `product` class regardless of registration.

## Log Each Scene

Create or maintain a CSV with one row per original image. Use these columns:

```text
filename,layout_group,visible_sku_counts,background,lighting,camera_height,view_angle,product_orientation,overlap,edge_truncation,negative,notes
```

Example:

```text
scene_001.jpg,layout_001,"GARDENGOOD-PASSATA-200:1;DUKAT-MLIJEKO-1000:2",shelf,warm indoor,eye-level,front,mostly upright,one bottle partly behind carton,no,false,three packages touching
```

Use the same `layout_group` for multiple pictures of the same physical arrangement, even if the camera moves slightly. The future train/validation/test split must keep all pictures with the same group together. If you rearrange the products, start a new group. Record actual expected visible quantities before model inference.

## Review And Later Evaluation

The approved three-photo CVAT batch is a development seed, not an independent test set. Add the new scenes through the documented proposal/CVAT workflow. Review every visible object and every mask; fix boundaries and add missed instances before importing them into `approved/`.

When the pilot collection is complete, we will decide group-based train/validation/test splits before training. Keep some complete layout groups untouched for final testing; do not tune thresholds, prompts or annotation rules against those scenes. Forty-seven scenes are a practical first round, not a production benchmark. Expect more collection and corrections after seeing per-scene and per-package failures.
