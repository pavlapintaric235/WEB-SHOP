# Proposed release changes

Remote: `pavlapintaric235/WEB-SHOP`. Work branch: `release/hackathon-20261025`. The fetched upstream README commit `87685fb` was fast-forwarded before editing; no history was rewritten. There were no local unpublished commits at the initial audit. Earlier working app changes were uncommitted/untracked and are included only where required by the release. Unrelated experiment changes stay outside these commits.

## Documentation and actual safe visuals

- `README.md`
- `docs/architecture.md`
- `docs/demo.md`
- `docs/deployment.md`
- `docs/evaluation.md`
- `docs/features.md`
- `docs/public/banner.svg`
- `docs/public/buyer-map.png`
- `docs/public/buyer-mobile.png`
- `docs/public/buyer.png`
- `docs/public/employee.png`
- `docs/public/owner.png`
- `docs/public/registration-mobile.png`
- `docs/public/registration.png`
- `docs/public/results.svg`
- `docs/release_changes.md`
- `docs/release_files.json`
- `docs/security.md`

## Publication hygiene

- `.claude/agents/project-overseer.md` ? obsolete assistant instruction removed
- `.claude/commands/security-review.md` ? obsolete assistant instruction removed
- `.github/workflows/security.yml`
- `.gitignore`
- `experiments/opencv5_guided_scan/aws_pilot/capture_config.example.json`
- `experiments/opencv5_guided_scan/aws_pilot/deployment_resources_20261020.json` ? remove from public tree; AWS configuration remains on disk

## Integrated app and isolated judge delivery

- `app.py`
- `app_8509.py`
- `app_judge.py`
- `auth.py`
- `buyer_8509.py`
- `buyer_ui.py`
- `catalog_import_ui.py`
- `db.py`
- `demo_data/WEB_SHOP_mock_catalog_prices_shelves_map.xlsx`
- `gallery_8509.py`
- `grouped_scan_ui.py`
- `images.py`
- `inventory.py`
- `judge_release.py`
- `live_scan.py`
- `owner_8509.py`
- `owner_test_8509.py`
- `owner_ui.py`
- `quality_8509.py`
- `registration_8509.py`
- `registration_wizard_8509.py`
- `requirements-dev.txt`
- `requirements-judge.txt`
- `requirements-live-scan.txt`
- `requirements-webshop-release.txt`
- `review_8509.py`
- `scan_review.py`
- `scanner_8509.py`
- `shop_catalog.py`
- `shop_map.py`
- `siglip_8509.py`
- `tests/test_fresh_scan_ui.py`
- `tests/test_judge_release.py`
- `tests/test_live_scan.py`
- `tests/test_practical_ux.py`
- `tests/test_registration_8509.py`
- `tests/test_webshop_delivery.py`
- `tools/browser_evidence.py`
- `tools/build_judge_preview.py`
- `tools/capture_judge_release.py`
- `tools/check_publication.py`
- `tools/check_release_tests.py`
- `tools/fetch_cloudflared.py`
- `tools/prepare_judge_demo.py`
- `tools/prepare_webshop_ux_demo.py`
- `tools/render_preview_audio.ps1`
- `tools/share_judge_demo.py`
- `tools/start_webshop_judge.ps1`
- `tools/start_webshop_ux_demo.ps1`
- `tools/tao_dino_worker.py`
- `vision.py`
- `visual_8509.py`

The complete final-tree allowlist is [release_files.json](release_files.json). Private working databases, photos, model files, videos, logs, access credentials, local connection configuration and unrelated benchmark outputs are excluded. Existing historical project metadata remains in Git history; no history cleanup or remote push is performed without the requested approval.
