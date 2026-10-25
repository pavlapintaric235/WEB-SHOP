# Architecture

WEB&SHOP is the application; InventoryLens supplies visual registration, recognition and inventory review. The integrated release is Python/Streamlit. The existing OpenCV 5 capture component executes on AWS in eu-north-1. DINO executes on an existing RunPod worker; SigLIP2 executes in the application process. Neither neural model is hosted by the AWS capture Lambda.

```mermaid
flowchart LR
  O[Owner: 3–5 reference photos] --> Q[OpenCV quality gate]
  Q --> A[Explicit image and identity approval]
  A --> E[Local pinned SigLIP2 embeddings]
  E --> C[(Shop-scoped catalogue and references)]
  P[Employee: distinct scene photo] --> S[Private S3 upload]
  S --> L[AWS Lambda: OpenCV 5]
  L --> G{Retake or proceed}
  G -->|Retake| P
  G -->|Proceed| D[Existing RunPod: Retail DINO]
  D --> R[Local SigLIP2 SKU suggestions]
  C --> R
  R --> V[OpenCV box geometry and interactive review]
  V --> F[Explicit action and before/after confirmation]
  F --> I[(Existing stock ledger)]
  C --> B[Buyer: calculator and shelf map]
```

```mermaid
flowchart TB
  Browser[Judge browser] --> Edge[Email-restricted HTTPS gateway]
  Edge --> Gate[30-minute application access-code gate]
  Gate --> App[Single Streamlit process / isolated judge database]
  App --> Role[Separate owner and employee credentials]
  App --> AWS[Existing AWS OpenCV capture]
  App --> Worker[Existing private RunPod connection]
```

The temporary HTTPS gateway reaches email sign-in for the operator; judge allowlisting and authenticated external acceptance remain pending. It must cover HTTP media routes and WebSocket connections as well as the landing page. The local app gate alone does not protect Streamlit media URLs. One server process avoids cross-replica SQLite and session-affinity problems. No public serving of the project directory is enabled. [Deployment details](deployment.md).

OpenCV measures resolution, focus and exposure before inference; local post-DINO geometry checks frame contact and intersection-over-union. Box overlap is not the percentage of a physical product visible. A blurred image requests a retake. Exposure-only overrides require a reason; AWS failure requires an explicit LOCAL choice. Neural suggestions do not approve stock.
