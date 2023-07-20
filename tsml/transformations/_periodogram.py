"""Periodogram transformer."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["PeriodogramTransformer"]

import math

import numpy as np
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency, check_n_jobs


class PeriodogramTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Periodogram transformer.

    This transformer converts a time series into its periodogram representation.

    Parameters
    ----------
    pad_series : bool, default=True
        Whether to pad the series to the next power of 2. If False, the series
        will be used as is.
    pad_with : str, default="constant"
        The type of padding to use. see the numpy.pad documentation mode parameter for
        options. By default, the series will be padded with zeros.
    constant_value : int, default=0
        The value to use when padding with a constant value.
    use_pyfftw : bool, default=True
        Whether to use the pyfftw library for FFT calculations. Requires the pyfftw
        package to be installed.
    n_jobs : int, default=1
        The number of threads to use for FFT calculations. Only used if use_pyfftw is
        True.

    Examples
    --------
    >>> from tsml.transformations import PeriodogramTransformer
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, _ = generate_3d_test_data(n_samples=4, n_channels=2, series_length=20,
    ...                              random_state=0)
    >>> tnf = PeriodogramTransformer()
    >>> tnf.fit(X)
    PeriodogramTransformer(...)
    >>> print(tnf.transform(X)[0])
    [[22.16456597 11.08122685  3.69018936  2.17665255  5.27387039  3.10598557
       6.311107    1.70468284  1.8269671   0.88838033  1.56747869  3.42037058
       1.67988661  1.71142437  3.49821716  1.25120108]
     [22.71382067  8.64933688  6.36412194  0.9298486   5.70358068  2.70669743
       4.33906385  0.36544821  2.28769936  3.67702091  1.45018642  1.26838712
       3.36395549  2.69146494  2.27041859  3.9023142 ]]
    """

    def __init__(
        self,
        pad_series=True,
        pad_with="constant",
        constant_value=0,
        use_pyfftw=True,
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

    def _more_tags(self) -> dict:
        return {"requires_fit": False, "optional_dependency": True}
