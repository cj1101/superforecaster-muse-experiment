import os, sys
# The workstream modules are flat modules inside the package directory
# (each was developed standalone); put that directory on sys.path so
# `from calibration import ...` / `from backtest import ...` resolve.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "superforecaster"))
