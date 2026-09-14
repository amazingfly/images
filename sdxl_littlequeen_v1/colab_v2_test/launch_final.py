import runpy
import sys
from pathlib import Path


target = Path("/content/lqxl_v2_selected.safetensors")
parts = sorted(Path("/content").glob("lqxl_v2_selected.safetensors.part_*"))
if not parts:
    raise FileNotFoundError("no upload chunks found for selected LoRA")
with target.open("wb") as destination:
    for part in parts:
        destination.write(part.read_bytes())

sys.argv = ["remote_test.py", "--mode", "final"]
runpy.run_path("/content/remote_test_v2.py", run_name="__main__")
