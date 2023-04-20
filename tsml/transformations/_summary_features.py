# -*- coding: utf-8 -*-
"""Summary feature transformer."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["SevenNumberSummaryTransformer"]

import numpy as np
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.numba_functions.stats import (
    row_mean,
    row_numba_max,
    row_numba_min,
    row_quantile,
    row_std,
)


class SevenNumberSummaryTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """TODO.
    default
    percentiles
    bowley
    """

    def __init__(
        self,
        summary_stats="default",
    ):
        self.summary_stats = summary_stats

    def fit(self, X, y=None):
        self._validate_data(X=X)
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        if self.summary_stats == "default":
            functions = [
                row_mean,
                row_std,
                row_numba_min,
                row_numba_max,
                0.25,
                0.5,
                0.75,
            ]
        elif self.summary_stats == "percentiles":
            functions = [
                0.2,
                0.9,
                0.25,
                0.5,
                0.75,
                0.91,
                0.98,
            ]
        elif self.summary_stats == "bowley":
            functions = [
                row_numba_min,
                row_numba_max,
                0.1,
                0.25,
                0.50,
                0.75,
                0.9,
            ]
        else:
            raise ValueError(
                f"Summary function input {self.summary_stats} not " f"recognised."
            )

        n_instances, n_dims, _ = X.shape

        Xt = np.zeros((n_instances, 7 * n_dims))
        for i in range(n_instances):
            for n, f in enumerate(functions):
                idx = n * n_dims
                if isinstance(f, float):
                    Xt[i, idx : idx + n_dims] = row_quantile(X[i], f)
                else:
                    Xt[i, idx : idx + n_dims] = f(X[i])

        return Xt

    def _more_tags(self):
        return {"requires_fit": False}
