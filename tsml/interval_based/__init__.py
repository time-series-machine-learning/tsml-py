"""Interval-based estimators."""

__all__ = [
    "BaseIntervalForest",
    "CIFClassifier",
    "CIFRegressor",
    "DrCIFClassifier",
    "DrCIFRegressor",
    "IntervalForestClassifier",
    "IntervalForestRegressor",
    "RandomIntervalClassifier",
    "RandomIntervalRegressor",
    "SupervisedIntervalClassifier",
    "RISEClassifier",
    "RISERegressor",
    "STSFClassifier",
    "RSTSFClassifier",
    "TSFClassifier",
    "TSFRegressor",
]

from tsml.interval_based._base import BaseIntervalForest
from tsml.interval_based._cif import (
    CIFClassifier,
    CIFRegressor,
    DrCIFClassifier,
    DrCIFRegressor,
)
from tsml.interval_based._interval_forest import (
    IntervalForestClassifier,
    IntervalForestRegressor,
)
from tsml.interval_based._interval_pipelines import (
    RandomIntervalClassifier,
    RandomIntervalRegressor,
    SupervisedIntervalClassifier,
)
from tsml.interval_based._rise import RISEClassifier, RISERegressor
from tsml.interval_based._stsf import RSTSFClassifier, STSFClassifier
from tsml.interval_based._tsf import TSFClassifier, TSFRegressor
