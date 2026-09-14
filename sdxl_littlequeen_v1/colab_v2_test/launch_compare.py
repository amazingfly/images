import runpy
import sys
from pathlib import Path


def assemble(name: str) -> None:
    target = Path("/content") / name
    parts = sorted(Path("/content").glob(f"{name}.part_*"))
    if not parts:
        raise FileNotFoundError(f"no upload chunks found for {name}")
    with target.open("wb") as destination:
        for part in parts:
            destination.write(part.read_bytes())


assemble("lqxl_v2_step900.safetensors")
assemble("lqxl_v2_final1200.safetensors")

sys.argv = ["remote_test.py", "--mode", "compare"]
runpy.run_path("/content/remote_test_v2.py", run_name="__main__")
