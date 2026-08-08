"""Reusable logic for the honey-yield predictive model.

The notebook is the narrative; this package is the code it calls. Anything that has to
be correct -- splits, baselines, leakage controls, metrics -- lives here so it can be
tested and reused instead of retyped per notebook cell.
"""

from honeymodel import data, evaluation, features, models  # noqa: F401

__all__ = ["data", "evaluation", "features", "models"]
