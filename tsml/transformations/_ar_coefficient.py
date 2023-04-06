# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]
__all__ = ["ARCoefficientTransformer"]

import math

import numpy as np
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class ARCoefficientTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    def __init__(
        self,
        lags=None,
        replace_nan=False,
    ):
        self.lags = lags
        self.replace_nan = replace_nan

        _check_optional_dependency("statsmodels", "statsmodels", self)

        super(ARCoefficientTransformer, self).__init__()

    def fit(self, X, y=None):
        self._validate_data(X=X)
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        lags = (
            int(12 * (X.shape[2] / 100.0) ** 0.25) if self.lags is None else self.lags
        )

        from statsmodels.regression.linear_model import burg

        Xt = np.zeros((X.shape[0], X.shape[1], lags))
        for i in range(X.shape[0]):
            for n in range(X.shape[1]):
                coefs, _ = burg(X[i, n], order=lags)
                Xt[i, n] = coefs

        if self.replace_nan:
            Xt[np.isnan(Xt)] = 0

        return Xt

    def _more_tags(self):
        return {"stateless": True, "optional_dependency": True}
