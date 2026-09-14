import runpy

try:
    runpy.run_path("/content/remote_accessories_v3_bootstrap.py", run_name="__main__")
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
