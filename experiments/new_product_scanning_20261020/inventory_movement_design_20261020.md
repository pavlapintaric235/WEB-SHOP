# Isolated stock movement design for Web&Shop

Design only, 2026-10-20. The existing app saves approved **snapshot replacement** counts. This document specifies a separate future **ADD/SUBTRACT** movement path after photo counts are matched to confirmed registered SKUs and approved by an authorized user. No production tables, inventory balances, or purchase records are changed here.

## Approval boundary

An OpenCV capture result, DINO box, SigLIP2 rank, or barcode decode may inform a draft scan. None posts stock. The reviewer must confirm each physical item and quantity, resolve merged/missed instances, select location, movement reason and operation, and explicitly approve a frozen scan revision. A shelf observation alone has no movement direction. Customer cart review does not subtract stock; an authorized finalized sale is a separate business event.

## Proposed append-only records

| Field | Purpose |
|---|---|
| `movement_id` | Unique immutable movement receipt |
| `approval_id` | Unique frozen user confirmation event and idempotency key |
| `scan_id`, `scan_revision`, `arrangement_id` | Source scan, reviewed version, and multiview grouping |
| `source_photo_sha256[]` | Exact photos supporting the approved count |
| `sku_id`, `location_id` | Confirmed registered product and scoped stock location |
| `quantity` | Positive integer physical units, aggregated once per SKU in the approval |
| `operation` | `ADD` or `SUBTRACT`; never inferred from pixels |
| `reason` | Receiving, authorized removal, sale, return, or approved adjustment |
| `actor_id`, `session_id`, `confirmed_at_utc` | Employee/owner identity, session and approval time |
| `source_event_id` | Optional receiving/order/sale reference |
| `reverses_movement_id` | Prior movement corrected by an opposite signed entry |
| `payload_sha256` | Detect a repeat approval ID with changed contents |

Posting is one transaction: verify authorization and confirmed SKU/location, lock or conditionally update the balance, insert movement lines with a unique `(approval_id, sku_id, location_id, operation)` key, and return a receipt. Repeating the same approval and payload returns the original receipt without changing stock; reusing its ID with a different payload is an error. A correction creates one linked reversal and a new approved movement, rather than editing or deleting the original. A unique reversal link prevents reversing the same movement twice. Negative-balance policy must be explicit at integration time.

The source review must assign one physical-instance ID per product. Confirmed unchanged camera views share an `arrangement_id`; their visible counts are reconciled into one approved quantity. Unknown products remain outside movements until the registry identity is confirmed. Keep this ledger separate from the current snapshot write path until migration and reconciliation are explicitly approved.
