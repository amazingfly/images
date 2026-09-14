import runpy

try:
    runpy.run_path("/content/remote_accessories_v4_region.py", run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
