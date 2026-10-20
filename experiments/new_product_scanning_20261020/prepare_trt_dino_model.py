"""Prepare an isolated NVIDIA Retail DINO ONNX for ONNX Runtime TensorRT.

The original NGC ONNX is preserved in the verified retry-5 archive. This changes
only the custom operation domain required by ONNX Runtime's TensorRT provider.
It does not prove GPU compatibility or numerical equivalence to a prior engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import onnx


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ZIP = (ROOT / "data" / "recognition_checks" / "runpod_b300_20261019" /
              "inventorylens_gpu_benchmark_retry5_20261019.zip")
SOURCE_MEMBER = "models/ngc/retail_detector/retail_object_detection_binary_v2.2.2.3.onnx"
SOURCE_SHA256 = "70be1b2fcb4e898da6b0724555a8fd36c993bbb48deafdba7a25ab34b13a8e89"
CANDIDATE_SHA256 = "7abf51f62888433d27811caa6af4b486cccd6382fcfe83a2b9c3a4bcd244aafd"
NODE_NAME = "MultiscaleDeformableAttnPlugin_TRT"
OUTPUT = ROOT / ".venv" / "experiment_models" / "retail_detector_trt_plugins_20261020.onnx"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()

    with ZipFile(SOURCE_ZIP) as archive:
        source = archive.read(SOURCE_MEMBER)
    if sha256_bytes(source) != SOURCE_SHA256:
        raise RuntimeError("Source NVIDIA Retail DINO member hash changed.")

    model = onnx.load_model_from_string(source)
    custom_nodes = [node for node in model.graph.node if node.op_type == NODE_NAME]
    if len(custom_nodes) != 12 or any(node.domain != "nvidia" for node in custom_nodes):
        raise RuntimeError("Unexpected custom-op count or original domain.")
    if any(node.domain == "nvidia" and node.op_type != NODE_NAME for node in model.graph.node):
        raise RuntimeError("An additional NVIDIA-domain node needs separate review.")
    original_imports = [entry for entry in model.opset_import if entry.domain == "nvidia"]
    if len(original_imports) != 1 or original_imports[0].version != 17:
        raise RuntimeError("Unexpected original NVIDIA opset import.")

    for node in custom_nodes:
        node.domain = "trt.plugins"
    original_imports[0].domain = "trt.plugins"
    original_imports[0].version = 1
    onnx.checker.check_model(model)

    candidate = model.SerializeToString()
    digest = sha256_bytes(candidate)
    if digest != CANDIDATE_SHA256:
        raise RuntimeError(f"Candidate serialization hash changed: {digest}")
    if output.exists():
        if sha256_file(output) != CANDIDATE_SHA256:
            raise RuntimeError(f"Refusing to overwrite a different file: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial")
        if partial.exists():
            raise RuntimeError(f"Remove or inspect stale partial output first: {partial}")
        partial.write_bytes(candidate)
        if sha256_file(partial) != CANDIDATE_SHA256:
            raise RuntimeError("Candidate file hash failed after writing.")
        partial.replace(output)

    print(json.dumps({
        "status": "LOCAL_STRUCTURAL_CHECK_ONLY",
        "source_zip": str(SOURCE_ZIP),
        "source_member": SOURCE_MEMBER,
        "source_sha256": SOURCE_SHA256,
        "output": str(output),
        "output_bytes": len(candidate),
        "output_sha256": digest,
        "custom_nodes_remapped": len(custom_nodes),
        "from_domain": "nvidia",
        "to_domain": "trt.plugins",
        "gpu_inference_executed": False,
        "prior_remapped_model_sha256": "c6185e3b1cd01abfe9d83e7f6f13083a8485b6cd0ca409642f237a5c65c36f41",
    }, indent=2))


if __name__ == "__main__":
    main()
