# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]
__all__ = ["SummaryFeatures"]

import numpy as np
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator


class SummaryFeatures(TransformerMixin, BaseTimeSeriesEstimator):
    """TODO."""

    def __init__(
        self,
        summary_function=None,
        quantiles=None,
    ):
        self.summary_function = summary_function
        self.quantiles = quantiles

    def fit(self, X, y=None):
        self._validate_data(X=X)
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X=X, reset=False)
        return np.zeros((X.shape[0], 1))

    def _more_tags(self):
        return {"stateless": True}
