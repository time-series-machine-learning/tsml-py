# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]
__all__ = ["AutocorrelationFunctionTransformer"]

import numpy as np
from numba import njit
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator


class AutocorrelationFunctionTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    def __init__(
        self,
        lags=100,
        min_values=4,
    ):
        self.lags = lags
        self.min_values = min_values

        super(AutocorrelationFunctionTransformer, self).__init__()

    def fit(self, X, y=None):
        self._validate_data(X=X)
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        n_instances, n_channels, series_length = X.shape

        lags = self.lags
        if lags > series_length - self.min_values:
            lags = series_length - self.min_values
        if lags < 0:
            lags = 1

        Xt = np.zeros((n_instances, n_channels, lags))
        for n in range(n_channels):
            Xt[:, n, :] = self._acf_2d(X[:, n, :], lags)

        return Xt

    def _more_tags(self):
        return {"requires_fit": False, "optional_dependency": True}

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _acf_2d(X, max_lag):
        n_instances, length = X.shape

        X_t = np.zeros((n_instances, max_lag))
        for i, x in enumerate(X):
            for lag in range(1, max_lag + 1):
                # Do it ourselves to avoid zero variance warnings
                lag_length = length - lag
                x1, x2 = x[:-lag], x[lag:]
                s1 = np.sum(x1)
                s2 = np.sum(x2)
                m1 = s1 / lag_length
                m2 = s2 / lag_length
                ss1 = np.sum(x1 * x1)
                ss2 = np.sum(x2 * x2)
                v1 = ss1 - s1 * m1
                v2 = ss2 - s2 * m2
                v1_is_zero, v2_is_zero = v1 <= 1e-9, v2 <= 1e-9
                if v1_is_zero and v2_is_zero:  # Both zero variance,
                    # so must be 100% correlated
                    X_t[i][lag - 1] = 1
                elif v1_is_zero or v2_is_zero:  # One zero variance
                    # the other not
                    X_t[i][lag - 1] = 0
                else:
                    X_t[i][lag - 1] = np.sum((x1 - m1) * (x2 - m2)) / np.sqrt(v1 * v2)

        return X_t
