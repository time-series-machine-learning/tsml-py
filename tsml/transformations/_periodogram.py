# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]
__all__ = ["PeriodogramTransformer"]

import math

import numpy as np
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class PeriodogramTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    def __init__(
        self,
        use_pyfftw=True,
        pad_series=True,
        pad_with="constant",
        constant_value=0,
        n_jobs=1,
    ):
        self.use_pyfftw = use_pyfftw
        self.pad_series = pad_series
        self.pad_with = pad_with
        self.constant_value = constant_value
        self.n_jobs = n_jobs

        if use_pyfftw:
            _check_optional_dependency("pyfftw", "pyfftw", self)

        super(PeriodogramTransformer, self).__init__()

    def fit(self, X, y=None):
        self._validate_data(X=X)
        return self

    def transform(self, X, y=None):
        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        threads_to_use = check_n_jobs(self.n_jobs)

        if self.pad_series:
            kwargs = {"mode": self.pad_with}
            if self.pad_with == "constant":
                kwargs["constant_values"] = self.constant_value

            X = np.pad(
                X,
                (
                    (0, 0),
                    (0, 0),
                    (
                        0,
                        int(
                            math.pow(2, math.ceil(math.log(X.shape[2], 2))) - X.shape[2]
                        ),
                    ),
                ),
                **kwargs,
            )

        if self.use_pyfftw:
            import pyfftw

            old_threads = pyfftw.config.NUM_THREADS
            pyfftw.config.NUM_THREADS = threads_to_use

            fft_object = pyfftw.builders.fft(X[:, :, :])
            Xt = np.abs(fft_object())
            Xt = Xt[:, :, : int(X.shape[2] / 2)]

            pyfftw.config.NUM_THREADS = old_threads
        else:
            Xt = np.abs(np.fft.fft(X)[:, :, : int(X.shape[2] / 2)])

        return Xt

    def _more_tags(self):
        return {"requires_fit": False, "optional_dependency": True}
