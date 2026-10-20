"""Package the isolated OpenCV 5 Lambda from pinned Linux wheels."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
DEPENDENCIES = ROOT / ".build_zip"
OUTPUT = ROOT / "capture_pilot_py313_linux_amd64.zip"
MAX_UNZIPPED_BYTES = 250 * 1024 * 1024
SOURCE_FILES = {
    "handler.py": ROOT / "handler.py",
    "capture_gate.py": PROJECT / "experiments/opencv5_guided_scan/capture_gate.py",
}


def main() -> None:
    if not (DEPENDENCIES / "cv2").is_dir() or not (DEPENDENCIES / "numpy").is_dir():
        raise RuntimeError("Install pinned Python 3.13 Linux wheels into .build_zip first")
    entries = {
        path.relative_to(DEPENDENCIES).as_posix(): path
        for path in DEPENDENCIES.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and "bin" not in path.relative_to(DEPENDENCIES).parts
    }
    entries.update(SOURCE_FILES)
    total = sum(path.stat().st_size for path in entries.values())
    if total >= MAX_UNZIPPED_BYTES:
        raise RuntimeError(f"Unzipped package {total} bytes exceeds Lambda's 250 MiB limit")
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, arcname=name)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"ZIP: {OUTPUT}")
    print(f"Files: {len(entries)}")
    print(f"Unzipped bytes: {total}")
    print(f"ZIP bytes: {OUTPUT.stat().st_size}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
