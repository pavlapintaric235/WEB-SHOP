"""Project-local, explicit downloads for the photo recognition models."""

import hashlib
import json
import shutil
from pathlib import Path

import requests

MODEL_DIR = Path(__file__).resolve().parent / "data" / "models"
SAM_FILE = MODEL_DIR / "mobile_sam.pt"
ENCODER_FILE = MODEL_DIR / "resnet18-f37072fd.pth"
ASSETS = (
    (SAM_FILE, "https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt", None),
    (ENCODER_FILE, "https://download.pytorch.org/models/resnet18-f37072fd.pth", "f37072fd"),
)


def prepare_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(MODEL_DIR).free < 1024**3:
        raise RuntimeError("At least 1 GB of free space is required for model setup.")
    manifest = []
    for destination, url, expected_prefix in ASSETS:
        if not destination.exists():
            print(f"Downloading {destination.name}...", flush=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            try:
                with requests.get(url, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
                if temporary.stat().st_size < 1_000_000:
                    raise RuntimeError(f"Incomplete checkpoint: {destination.name}")
                if expected_prefix and not digest.startswith(expected_prefix):
                    raise RuntimeError(f"Checkpoint checksum failed: {destination.name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if expected_prefix and not digest.startswith(expected_prefix):
            raise RuntimeError(f"Checkpoint checksum failed: {destination.name}")
        manifest.append({"file": destination.name, "url": url, "sha256": digest})
        print(f"Ready: {destination.name} ({destination.stat().st_size / 1024**2:.1f} MB)", flush=True)
    (MODEL_DIR / "assets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    prepare_models()
