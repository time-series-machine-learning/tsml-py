# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]
__all__ = ["PeriodogramTransformer"]

import numpy as np
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class PeriodogramTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    def __init__(
        self,
        use_pyfftw=True,
    ):
        self.use_pyfftw = use_pyfftw

        super(PeriodogramTransformer, self).__init__()

    def fit(self, X, y=None):
        self._validate_data(X=X)
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        Xt = np.zeros((X.shape[0], X.shape[1], int(X.shape[2] / 2)))
        if self.use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)
            import pyfftw

            fft_object = pyfftw.builders.fft(X)
            per_X = np.abs(fft_object)
            per_X[:, : int(X.shape[2] / 2)]
        else:
            X_p = np.zeros(
                (
                    self.n_instances_,
                    self.n_dims_,
                    int(
                        math.pow(2, math.ceil(math.log(self.series_length_, 2)))
                        - self.series_length_
                    ),
                )
            )
            X_p = np.concatenate((X, X_p), axis=2)
            X_p = np.abs(np.fft.fft(X_p)[:, :, : int(X_p.shape[2] / 2)])

        return Xt

    def _more_tags(self):
        return {"stateless": True}
