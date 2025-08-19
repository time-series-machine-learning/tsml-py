"""AR coefficient feature transformer."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["ARCoefficientTransformer"]


import numpy as np
from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class ARCoefficientTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Autoreggression coefficient feature transformer.

    Coefficients of an autoregressive model using Burg's method. The Burg method
    fits a forward-backward autoregressive model to the data using least squares
    regression.

    Parameters
    ----------
    order : int or callable, default=100
        The order of the autoregression. If callable, the function should take a 3D
        numpy array of shape (n_instances, n_channels, n_timepoints) and return an
        integer.
    min_values : int, default=0
        Always transform at least this many values unless the series length is too
        short. This will reduce order if needed.
    replace_nan : bool, default=False
        If True, replace NaNs in output with 0s.
    """

    def __init__(
        self,
        order=100,
        min_values=0,
        replace_nan=False,
    ):
        self.order = order
        self.min_values = min_values
        self.replace_nan = replace_nan

        _check_optional_dependency("statsmodels", "statsmodels", self)

        super().__init__()

    def fit(self, X, y=None):
        self._validate_data(X=X, ensure_equal_length=True)
        return self

    def transform(self, X, y=None):
        from statsmodels.regression.linear_model import burg

        X = self._validate_data(X=X, reset=False, ensure_equal_length=True)
        X = self._convert_X(X)

        n_instances, n_channels, n_timepoints = X.shape

        order = self.order(X) if callable(self.order) else self.order
        if order > n_timepoints - self.min_values:
            order = n_timepoints - self.min_values
        if order < 0:
            order = 1

        if order > n_timepoints - 1:
            raise ValueError(
                f"order ({order}) must be smaller than n_timepoints - 1 "
                f"({n_timepoints - 1})."
            )

        Xt = np.zeros((n_instances, n_channels, order))
        for i in range(n_instances):
            for n in range(n_channels):
                coefs, _ = burg(X[i, n], order=order)
                Xt[i, n] = coefs

        if self.replace_nan:
            Xt[np.isnan(Xt)] = 0

        return Xt

    def _more_tags(self) -> dict:
        return {"requires_fit": False, "optional_dependency": True}

    @classmethod
    def get_test_params(cls, parameter_set: str | None = None) -> dict | list[dict]:
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return {
            "order": 4,
        }
