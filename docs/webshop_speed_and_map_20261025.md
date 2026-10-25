# Recognition speed and buyer map

Use http://localhost:8509. Launch remains `./tools/start_webshop_ux_demo.ps1`.

## Measured speed

Actual pinned SigLIP2, the existing 55 registered vectors, and one fresh request to the existing RunPod worker were used. SKU decisions for all ten proposals matched the prior measured scene. No model, threshold, crop geometry or scoring policy changed.

| Work | Before / serial | After / cached |
|---|---:|---:|
| Repeated reference fingerprint | 3.908 s | 0.0127 s |
| Repeated photo preparation | Repeated full normalization/quality work | 0.0786 s |
| Gallery snapshot load | Full photo decoding | 0.0182 s |
| Matching ten actual crops | 14.3656 s, batch 1 | 7.5673 s, batch 4 |

Prepared LIVE recognition measured **27.5903 s**: DINO request **20.023 s** (detector core **14.612 s**) plus matching **7.5673 s**. Photo upload, capture analysis, initial model preparation and UI rendering are additional. This is one scene and a bounded timing check, not a broad benchmark or an independent accuracy evaluation. The previous 35.340 s acceptance ran under different startup/load conditions, so it is not a controlled overall-speed comparison.

Employee Scan → **Prepare scanner before recording → Prepare live scanner** loads local weights and references ahead of filming. Initial model preparation measured 24.0647 s and initial reference hashing 7.6612 s; keep the server running afterward. Every LIVE scan still calls real DINO. No predictions are replaced with replay.

The isolated old-loader timing probe crashed in Transformers' memory-mapped checkpoint reader before completing. The new reader loads the same image tensors through safetensors 0.8.0's ordinary-read backend into owned NumPy/PyTorch buffers, strictly validates weights and initializes the original nonpersistent position indices. An exact-output tiny-model test passed. Actual full-size inference subsequently preserved the existing scene assignments. This avoids the observed crash path; it is not a guarantee against every Windows memory-pressure failure.

Gallery snapshots contain actual stored vectors, keyed by shop, model revision, source content, selected mask content and active embedding content. File hashes are reused only while size, modification time, creation/change time and file identity remain unchanged. Registration changes still invalidate the gallery; price and stock changes do not. Images are decoded one product at a time when a snapshot needs building. Review reuses photo preparation and uses faster lossless PNG compression for saved annotations.

## Buyer map

**Buyer → Products:** choose quantities. Then **Shopping list & map** shows highlighted shelves, your list, totals and manual stop progression together. **Owner → Shelves** edits shelf blocks and an optional walkable grid. **Camera cart** opens the existing read-only scanner.

The isolated fictional demo had no saved shelf coordinates. Eight positions were copied from the user-supplied workbook's Shelf_Locations sheet; stock, prices, existing product shelf assignments and references were preserved. Products without a shelf assignment remain explicitly unlocated. No entrance, checkout or walkable aisles were invented. Without a valid connected grid, the display says ordered shelf list and makes no shortest-route claim.

Desktop map (local artifact: `webshop_8509_buyer_map.png`) · Mobile map (local artifact: `webshop_8509_buyer_map_mobile.png`)

Validation: 19 registration/map regressions passed; the corrected image-reader equivalence check, two gallery/buyer checks, and two final map-layout checks also passed. The real desktop/mobile browser check verified two lentil cans at €2.98, shelf A1 highlighted, no mobile overflow and unchanged stock. Evidence: `data/webshop_ux_demo/speed_after.json`, `speed_verified.log`, `map_browser_check.json`, and the targeted test logs. Original 8508 code and ledger were preserved.

Changed files: `gallery_8509.py`, `siglip_8509.py`, `scanner_8509.py`, `buyer_8509.py`, optional compact rendering in `shop_map.py`, focused tests and local verification tools. No training, model downloads, GPU starts or new cloud infrastructure.
