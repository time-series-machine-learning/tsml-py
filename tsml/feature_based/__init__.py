"""Feature-based estimators."""

__all__ = [
    "Catch22Classifier",
    "Catch22Regressor",
    "FPCAClassifier",
    "FPCARegressor",
]

from tsml.feature_based._catch22 import Catch22Classifier, Catch22Regressor
from tsml.feature_based._fpca import FPCAClassifier, FPCARegressor
