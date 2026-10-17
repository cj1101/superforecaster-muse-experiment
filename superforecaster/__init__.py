"""Superforecaster: an evidence-first forecasting system.

Priority order (per project charter): markets, prediction markets, crypto.
Every module is dependency-light (numpy / pandas / scipy / stdlib only) and
every shipped claim is backed by a walk-forward backtest in ../results/.
"""
from .engine import Superforecaster, shrink, fit_shrink_weight, log_odds_pool

__version__ = "1.0.0"
__all__ = ["Superforecaster", "shrink", "fit_shrink_weight", "log_odds_pool"]
