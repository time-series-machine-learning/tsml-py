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
    """Seven-number summary transformer.

    Transforms a time series into seven basic summary statistics.

    Parameters
    ----------
    summary_stats : ["default", "percentiles", "bowley", "tukey"], default="default"
        The summary statistics to compute.
        The options are as follows, with float denoting the percentile value extracted
        from the series:
            - "default": mean, std, min, max, 0.25, 0.5, 0.75
            - "percentiles": 0.215, 0.887, 0.25, 0.5, 0.75, 0.9113, 0.9785
            - "bowley": min, max, 0.1, 0.25, 0.5, 0.75, 0.9
            - "tukey": min, max, 0.125, 0.25, 0.5, 0.75, 0.875

    Examples
    --------
    >>> from tsml.transformations import SevenNumberSummaryTransformer
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, _ = generate_3d_test_data(n_samples=4, n_channels=2, series_length=10,
    ...                              random_state=0)
    >>> tnf = SevenNumberSummaryTransformer()
    >>> tnf.fit(X)
    SevenNumberSummaryTransformer(...)
    >>> print(tnf.transform(X)[0])
    [1.12176987 1.09468673 0.52340259 0.68084237 0.         0.04043679
     1.92732552 1.85119328 0.8542758  0.39514141 1.14764656 1.34620131
     1.39573111 1.64479229]
    """

    def __init__(
        self,
        summary_stats="default",
    ):
        self.summary_stats = summary_stats

        super(SevenNumberSummaryTransformer, self).__init__()

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
        elif self.summary_stats == "tukey":
            functions = [
                row_numba_min,
                row_numba_max,
                0.125,
                0.25,
                0.5,
                0.75,
                0.875,
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

    def _more_tags(self) -> dict:
        return {"requires_fit": False}
