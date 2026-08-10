# Runner: apply the QFP8WT env shim, then exec a probe script as __main__.
# Usage: python run_probe.py <probe_file.py> [probe args...]
import runpy
import sys

import _spyre_env_shim  # noqa: F401  (aliases ElementArrangement.QFP8WT before compile)

target = sys.argv[1]
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name="__main__")
