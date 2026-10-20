"""Optional, pretrained Hugging Face image encoders for isolated SKU retrieval tests."""

from __future__ import annotations

import os

import numpy as np

DINO_V2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


class HuggingFaceImageEncoder:
    """Feature-only DINOv2 or SigLIP2 encoder; loads local weights unless opted in."""

    def __init__(self, model_id: str, kind: str, device: str = "cpu",
                 local_files_only: bool = True, revision: str = "main"):
        if kind not in {"dinov2", "siglip2"}:
            raise ValueError(f"Unsupported Hugging Face image encoder: {kind}")
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError(
                "Install the optional requirements in experiments/hybrid_pipeline/requirements-encoders.txt."
            ) from error

        self.name = kind
        self.kind = kind
        self.model_id = model_id
        self.revision = revision
        self.torch = torch
        self.device = torch.device(device)
        cache_dir = os.environ.get("WEBSHOP_HF_CACHE")
        options = {"local_files_only": local_files_only, "revision": revision}
        if cache_dir:
            options["cache_dir"] = cache_dir
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False, **options)
            self.model = AutoModel.from_pretrained(model_id, **options)
        except Exception as error:
            mode = "local cache" if local_files_only else "model hub"
            raise RuntimeError(f"Could not load {model_id} from the {mode}: {error}") from error
        self.model = self.model.eval().requires_grad_(False).to(self.device)

    def encode(self, images, batch_size=8) -> np.ndarray:
        if not images:
            return np.empty((0, 384 if self.kind == "dinov2" else 768), dtype=np.float32)
        vectors = []
        with self.torch.inference_mode():
            for offset in range(0, len(images), batch_size):
                batch = [image.convert("RGB") for image in images[offset:offset + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                if self.kind == "dinov2":
                    output = self.model(**inputs)
                    features = output.last_hidden_state[:, 0]
                else:
                    features = self.model.get_image_features(**inputs)
                features = self.torch.nn.functional.normalize(features, dim=1)
                vectors.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors, axis=0)


class DinoV2SmallEncoder(HuggingFaceImageEncoder):
    name = "dinov2"

    def __init__(self, device="cpu", local_files_only=True):
        super().__init__("facebook/dinov2-small", "dinov2", device, local_files_only,
                         revision=DINO_V2_REVISION)


class SigLIP2BaseEncoder(HuggingFaceImageEncoder):
    name = "siglip2"

    def __init__(self, device="cpu", local_files_only=True):
        super().__init__("google/siglip2-base-patch16-224", "siglip2", device,
                         local_files_only, revision=SIGLIP2_REVISION)

    def encode(self, images, batch_size=1):
        return super().encode(images, batch_size=batch_size)
