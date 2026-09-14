#!/usr/bin/env python3
"""Archive whatever accessory test outputs exist after an interrupted comparison."""

from pathlib import Path
import tarfile


root = Path("/content/lqaccessories_v1")
output = root / "test_output"
archive_path = Path("/content/lqaccessories_partial_results.tar.gz")
images = sorted(output.glob("*.png"))
if not images:
    raise RuntimeError("no partial test images are available")
archive_path.unlink(missing_ok=True)
with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(output, arcname="output")
    summary = root / "test_summary.json"
    if summary.is_file():
        archive.add(summary, arcname="test_summary.json")
print(f"Archived {len(images)} partial images to {archive_path}")
