"""Collect outputs and downloaded weights into one artifact before Pod termination."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--archive", type=Path, default=ROOT / "inventorylens_runpod_results_and_models.zip")
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    archive = args.archive.resolve()
    if archive.exists():
        raise FileExistsError(f"Refusing to overwrite existing download archive: {archive}")
    include_roots = [output, ROOT / "models", ROOT / "project" / "data" / "models",
                     ROOT / "project" / "experiments" / "hybrid_pipeline" / "weights",
                     ROOT / "data" / "source"]
    files = []
    for base in include_roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not path.name.endswith(".part"):
                files.append(path)
    script_files = [ROOT / name for name in (
        "verify_cuda.py", "prepare_models.py", "prepare_reference_cache.py",
        "run_inference.py", "package_downloads.py", "README_RUNPOD.md",
        "requirements-runpod.txt", "bundle_manifest.json",
    ) if (ROOT / name).is_file()]
    unique = sorted(set(files + script_files))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as target:
        for path in unique:
            target.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(archive) as target:
        bad = target.testzip()
        if bad:
            raise RuntimeError(f"Generated download archive CRC failure: {bad}")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "included_files": len(unique),
                      "note": "Download this ZIP from JupyterLab before stopping or terminating the Pod."}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
