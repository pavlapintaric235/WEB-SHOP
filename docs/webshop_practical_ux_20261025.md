# WEB&SHOP practical UX demo — 25 October 2026

Isolated local demo: **http://localhost:8509**. Owner login `owner` / `owner123`; employee login `employee` / `employee123`. Buyer view uses the selected shop catalogue. The original release server on 8508 remains running.

The new database is `data/webshop_ux_demo/inventorylens.sqlite3`, copied from the existing fictional release demo with its registered reference mappings retained. Only the new copy's stock/history was cleared. The supplied workbook is copied to `demo_data/WEB_SHOP_mock_catalog_prices_shelves_map.xlsx`; it has **14 products and 8 shelves**. It has not been automatically imported into the demo.

## Demo actions

1. **Owner → Products → Import catalogue workbook / CSV.** Upload the supplied file or select “Preview supplied fictional workbook”. `Products_Import` row 5 maps SKU/Product_Name/Count_Unit/Price_EUR/Opening_Stock/Shelf_ID to the existing import schema. `Shelf_Locations` preserves shelf IDs, zone, aisle and map coordinates. Edit SKU mappings or omit rows; duplicate IDs, invalid prices/quantities and unknown shelves block import. Confirm the preview explicitly. Opening stock is optional and cannot overwrite an existing balance. New/provisional IDs remain pending with no reference images or fabricated embeddings. Existing SKU references are preserved.
2. **Employee → Fresh-photo SKU scan.** Keep the existing local/AWS capture check, then use LIVE detection or the clearly marked historical replay. The photo and stable numbered boxes sit beside SKU groups. Use +/−, Set quantity, Add missed product, or expand a group to inspect suggestions, correct boxes, remove false detections and split merged boxes. Split children require review. Unknown products can create a shop-owner registration request. Draft corrections, timestamps and actors are saved with the scan photo/JSON.
3. **Confirmation defaults to Review only.** Audit requires explicit SKU scope, including explicit zero decisions. ADD/SUBTRACT show before/after stock. Final approval is bound to the displayed action and remains idempotent. A posted draft is locked. Buyer scans have no stock action controls.
4. **Buyer.** Shopping lists and camera carts have editable quantities, current-shop euro price estimates and optional budgets. Unknown/unpriced items are excluded and totals marked incomplete. Add list items to highlight their shelf stops.
5. **Owner → Inventory → Edit 2D shop map.** Position existing shelf IDs in zero-based cells. Optionally author a rectangular `. # E C` walkable grid; shelves occupy blocked cells. A connected valid grid enables nearest-next-shelf ordering with BFS per segment, without claiming an optimal whole-shop tour. Otherwise the buyer sees an ordered shelf list. Progress advances manually.

## Changed implementation

- Import: `shop_catalog.py`, `catalog_import_ui.py`, `owner_ui.py`, additive tables/column in `db.py`, authorized operations in `inventory.py`.
- Review: `scan_review.py`, `grouped_scan_ui.py`, `scanner_ui.py`, status colors in `live_scan.py`.
- Buyer/map: `buyer_ui.py`, `shop_map.py`.
- Demo: copied sample workbook, `tools/prepare_webshop_ux_demo.py`, `tools/start_webshop_ux_demo.ps1`.
- Checks: `tests/test_practical_ux.py`, updated grouped-review regression in `tests/test_webshop_delivery.py`, `tools/check_release_tests.py`.

## Validation and remaining work

**72 tests passed**, with zero failures/errors/skips (165.373 s). Two final owner-upload/grouped-review UI checks also passed after the final confirmation guards. Run `.venv\Scripts\python.exe tools/check_release_tests.py` to repeat the release and UX regressions. Results are in `data/webshop_release_checks/tests.json`, `tests.log`, and `final_ux_checks.log`. Tests use temporary databases and explicitly mocked inference/AWS failures; they do not call cloud inference.

The actual isolated demo database also passed an AppTest navigation smoke: owner workbook preview without importing, employee fresh-scan screen, buyer list quantities and budget, and unchanged stock/history (`data/webshop_ux_demo/ux_navigation_smoke.json`). Both local servers returned HTTP 200 on their health endpoints. The new launcher reuses the existing private tunnel configuration and CPU thread settings.

Real package-to-SKU mappings still require owner confirmation. The workbook's illustrative shelf positions need owner confirmation and a walkable grid before real route guidance. A new real photo and a live AWS/GPU UX smoke have not been executed in this change. No model training, model downloads or GPU provisioning was performed.
