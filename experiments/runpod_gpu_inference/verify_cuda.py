"""Fail closed unless PyTorch can execute a real operation on an NVIDIA CUDA GPU."""

from __future__ import annotations

import json
import sys


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Do not run inference on CPU by accident.")
    device = torch.device("cuda:0")
    left = torch.randn((2048, 1024), device=device)
    right = torch.randn((1024, 512), device=device)
    result = left @ right
    torch.cuda.synchronize(device)
    if result.device.type != "cuda" or not torch.isfinite(result).all().item():
        raise RuntimeError("The CUDA matrix-operation verification failed.")
    details = torch.cuda.get_device_properties(device)
    report = {
        "cuda_available": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device),
        "device_memory_gib": round(details.total_memory / (1024**3), 2),
        "matrix_test": "passed",
        "matrix_result_device": str(result.device),
    }
    print(json.dumps(report, indent=2))
    if "3090" not in report["device_name"]:
        print("WARNING: requested target is RTX 3090; detected a different CUDA GPU.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
