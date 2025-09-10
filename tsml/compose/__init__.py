"""Composable estimators."""

__all__ = [
    "ChannelEnsembleClassifier",
    "ChannelEnsembleRegressor",
    "SklearnToTsmlClassifier",
    "SklearnToTsmlClusterer",
    "SklearnToTsmlRegressor",
]

from tsml.compose._channel_ensemble import (
    ChannelEnsembleClassifier,
    ChannelEnsembleRegressor,
)
from tsml.compose._sklearn_to_tsml import (
    SklearnToTsmlClassifier,
    SklearnToTsmlClusterer,
    SklearnToTsmlRegressor,
)
