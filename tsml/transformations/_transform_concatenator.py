""""""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["TransformerConcatenator"]

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.utils._tags import _safe_tags


class TransformerConcatenator(TransformerMixin, BaseTimeSeriesEstimator):
    """ """

    def __init__(
        self,
        transformers,
        validate=True,
    ):
        self.transformers = transformers
        self.validate = validate

    def fit_transform(self, X, y=None):
        if self.validate:
            self._validate_data(X, ensure_min_series_length=1)
        else:
            self._check_n_features(X, True)

        self.transformers_ = [_clone_estimator(t) for t in self.transformers]

        arr = self.transformers_[0].fit_transform(X, y)
        for transformer in self.transformers_[1:]:
            arr = np.concatenate((arr, transformer.fit_transform(X, y)), axis=-1)

        return arr

    def fit(self, X, y=None):
        """Fit transformer by checking X.

        If ``validate`` is ``True``, ``X`` will be checked.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FunctionTransformer class instance.
        """
        if self.validate:
            if any([_safe_tags(t, key="requires_y") for t in self.transformers]):
                self._validate_data(X=X, y=y, ensure_min_series_length=1)
            else:
                self._validate_data(X=X, ensure_min_series_length=1)
        else:
            self._check_n_features(X, True)

        self.transformers_ = [_clone_estimator(t).fit(X, y) for t in self.transformers]

        return self

    def transform(self, X):
        """Transform X using the forward function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.

        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if any([_safe_tags(t, key="requires_fit") for t in self.transformers]):
            check_is_fitted(self)
            transformers = self.transformers_
        else:
            transformers = self.transformers

        if self.validate:
            X = self._validate_data(X=X, reset=False, ensure_min_series_length=1)

        arr = transformers[0].transform(X)
        for transformer in transformers[1:]:
            arr = np.concatenate((arr, transformer.transform(X)), axis=-1)

        return arr

    def _more_tags(self) -> dict:
        return {
            "no_validation": not self.validate,
            "requires_fit": any(
                [_safe_tags(t, key="requires_fit") for t in self.transformers]
            ),
            "requires_y": any(
                [_safe_tags(t, key="requires_y") for t in self.transformers]
            ),
            "X_types": ["3darray", "2darray", "np_list"],
            "_xfail_checks": {
                "check_parameters_default_constructible": "Intentional required "
                "parameter."
            },
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from tsml.transformations import Catch22Transformer

        return {
            "transformers": [
                Catch22Transformer(
                    features=["DN_HistogramMode_5", "DN_HistogramMode_10"]
                ),
                Catch22Transformer(features="CO_f1ecac"),
            ],
        }
