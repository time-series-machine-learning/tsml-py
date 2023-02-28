# -*- coding: utf-8 -*-
"""Interval based estimators."""

__all__ = [
    # "CIFClassifier",
    # "CIFRegressor",
    # "DrCIFClassifier",
    # "DrCIFRegressor",
    "IntervalForestClassifier",
    "IntervalForestRegressor",
    "RandomIntervalClassifier",
    "RandomIntervalRegressor",
    "SupervisedIntervalClassifier",
    # "RISEClassifier",
    # "RISERegressor",
    # "STSFClassifier",
    # "RSTSFClassifier",
    "TSFClassifier",
    "TSFRegressor",
]

from tsml.interval_based._interval_forest import (
    IntervalForestClassifier,
    IntervalForestRegressor,
)
from tsml.interval_based._interval_pipelines import (
    RandomIntervalClassifier,
    RandomIntervalRegressor,
    SupervisedIntervalClassifier,
)
from tsml.interval_based._tsf import TSFClassifier, TSFRegressor
