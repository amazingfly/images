import runpy
import sys

sys.argv = ["remote_generate.py", "--reset"]
runpy.run_path("/content/remote_equipment_generate.py", run_name="__main__")
