import runpy
import sys

sys.argv = ["remote_generate.py"]
result = runpy.run_path("/content/remote_generate.py", run_name="__main__")
