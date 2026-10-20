# Grocery Reference Photos: Trial Protocol

This is a practical starting protocol to test, not a validated recipe or a final
minimum-photo requirement. The app now permits at most four reference photos per
SKU. Aim for the first three views and add the fourth when available. Fewer photos
are still allowed during setup; the minimum can be decided from the results.

## One SKU, One Product

Use one physical product in each reference photo, with no other packages or hands
in view. One SKU means one exact brand, variant and package size. Different units
of the same SKU do not need separate registrations. Keep the entire product,
including its cap and base, visible and away from the frame edges.

## Suggested Views

Keep the product upright. For the first three views, keep the lens approximately
at the middle of the front label. Move the camera around the product, or rotate
the product consistently. Do not turn it so far that the main label disappears.

| File Suffix | Camera Position | What To Show |
| --- | --- | --- |
| `front` | Straight-on, approximately 0 degrees | Full product, front label readable |
| `left` | Approximately 30-45 degrees left of front | Front label and left-side shape |
| `right` | Approximately 30-45 degrees right of front | Front label and right-side shape |
| `raised` | Optional: front, about 20-30 degrees above label level | Front label, shoulder/cap and some top surface |

These angles are proposed trial settings, not source-established optimums. Use
them consistently for the first batch so later comparisons are meaningful.
For cans, jars and bottles, left/right views should change the visible label and
reflections. For cartons and bags, they should show the side panels and depth.
Do not photograph only a cap, barcode, ingredient panel or isolated label.

If recognizing rear-facing packages is a real requirement, test a rear view as
an alternative fourth photo in a later batch. Three front-facing photos do not
establish that the product can be recognized from every direction. Pure top-down
shots are also not the first priority when most distinguishing branding is on
the front. Change one part of the protocol at a time when comparing results.

## Background And Lighting

Start with a large, plain, matte neutral-gray sheet. Curve it from behind the
product underneath it so both the wall and tabletop are covered. This avoids a
busy wood surface, a strong wall/table line, and a large separate table region.
Use lighter gray for dark packages or darker gray for pale packages when the
outline otherwise blends into the background. For transparent glass, check that
the bottle's sides, neck and base remain distinguishable against the backdrop;
medium/darker gray is worth trying, but contrast matters more than a fixed color.

Keep the same setup for a product's first reference set. We can compare other
backgrounds later. A neutral sweep is our proposed segmentation-friendly setup;
ordinary e-commerce photography often uses white. Do not assume one color is best
for every package or that a background choice alone fixes segmentation.

Use soft window light from the side, ideally through a sheer curtain, or a broad
diffused lamp. Avoid direct sunlight, phone flash, hard shadows, colored light
and a bright window directly behind the product. Move the light or camera when
a reflection covers the logo, but do not digitally redraw the label. A white card
outside the frame can lift shadows on the opposite side. Labels and the physical
outline should both remain visible, particularly on metal cans and glass bottles.

## Framing And Phone Settings

- Use the normal 1x camera. Avoid ultrawide distortion and digital zoom.
- Aim for roughly 70-80% of the image height, leaving about 10% margin above and
  below the full product. Leave clear space on both sides too.
- Tap the main label to focus. Keep the phone steady and check the result at full size.
- Keep original, high-resolution JPEG/PNG files. Do not use screenshots or annotated outputs.
- Disable portrait-mode blur, beauty filters and other effects that alter the outline.
- Keep packaging in its ordinary state: upright bottles, uncrushed cans and bags
  that are not flattened by a hand.

## Register First, Test Separately

Name files consistently, for example `JANA-ANTIOX-500_front.jpg`, `_left.jpg`,
`_right.jpg`, and optionally `_raised.jpg`. Upload the three/four independent
photos together under that one SKU in Owner dashboard > Products. Do not upload
three crops or edits of the same image as three distinct viewpoints.

Register every grocery SKU before mixed-scene testing. Check the processed
reference board on the first detection: it must contain actual products rather
than a tabletop, wall, cap or isolated label. The automatic selector can still
fail, so this check matters even with a clean setup.

Take new photographs for evaluation; do not reuse reference photos or crop new
references out of the test image. First test one fully visible unit per SKU, then
multiple separated units, then mixtures, and finally partial occlusion and real
shelf backgrounds. Record expected quantities before inference and keep the
0.80 similarity / 0.05 margin fixed for the initial comparison. Keep failures
as regression evidence rather than dropping the difficult SKU.

Recognition experiments should not be approved as actual inventory counts.
CSV import can wait unless there is a real grocery baseline/template to preserve.

## Later Camera Work

Multi-view references may make SKU matching more useful across camera viewpoints,
but the current app is not being trained by these uploads. They are registered
examples used by a frozen matcher. The protocol does not by itself solve motion
blur, partial visibility, duplicate counting across frames or hidden inventory.
Video needs separate tracking and counting validation after the photo baseline.

## Sources

- [Shopify product photography setup](https://help.shopify.com/en/manual/products/product-media/product-photography)
- [Shopify natural-light product photography](https://www.shopify.com/blog/natural-lighting-product-photography)
- [Ultralytics data collection and annotation](https://docs.ultralytics.com/guides/data-collection-and-annotation)
- [Ultralytics model testing and held-out data](https://docs.ultralytics.com/guides/model-testing)
