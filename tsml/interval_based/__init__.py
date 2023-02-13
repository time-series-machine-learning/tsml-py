# -*- coding: utf-8 -*-
"""Interval based estimators."""


__all__ = [
    "CIFClassifier",
    "CIFRegressor",
    "DrCIFClassifier",
    "DrCIFRegressor",
    "IntervalForestClassifier",
    "IntervalForestRegressor",
    "RISEClassifier",
    "RISERegressor",
    "STSFClassifier",
    "TSFClassifier",
    "TSFRegressor",
]

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
from tsml.interval_based._rise import RISEClassifier, RISERegressor
from tsml.interval_based._stsf import STSFClassifier
from tsml.interval_based._tsf import TSFClassifier, TSFRegressor
