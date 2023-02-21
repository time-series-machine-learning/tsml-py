# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "BaseTimeSeriesEstimator",
    "clone_estimator",
]

from typing import Union

from numpy.random import RandomState
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble._base import _set_random_states
from sklearn.utils.validation import _check_y

from tsml.utils._tags import _DEFAULT_TAGS, _safe_tags
from tsml.utils.validation import _num_features, check_X, check_X_y


class BaseTimeSeriesEstimator(BaseEstimator):
    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        **check_params,
    ):
        """Validate input data and set or check the `n_features_in_` attribute.

        Uses the `scikit-learn` 1.2.1 `_validate_data` function as a base.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features), default='no validation'
            The input samples.
            If `'no_validation'`, no validation is performed on `X`. This is
            useful for meta-estimator which can delegate input validation to
            their underlying estimator(s). In that case `y` must be passed and
            the only accepted `check_params` are `multi_output` and
            `y_numeric`.

        y : array-like of shape (n_samples,), default='no_validation'
            The targets.

            - If `None`, `check_array` is called on `X`. If the estimator's
              requires_y tag is True, then an error will be raised.
            - If `'no_validation'`, `check_array` is called on `X` and the
              estimator's requires_y tag is ignored. This is a default
              placeholder and is never meant to be explicitly set. In that case
              `X` must be passed.
            - Otherwise, only `y` with `_check_y` or both `X` and `y` are
              checked with either `check_array` or `check_X_y` depending on
              `validate_separately`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.

        validate_separately : False or tuple of dicts, default=False
            Only used if y is not None.
            If False, call validate_X_y(). Else, it must be a tuple of kwargs
            to be used for calling check_array() on X and y respectively.

            `estimator=self` is automatically added to these dicts to generate
            more informative error message in case of invalid input data.

        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array` or
            :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

            `estimator=self` is automatically added to these params to generate
            more informative error message in case of invalid input data.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `y` are
            validated.
        """
        if y is None and self._get_tags()["requires_y"]:
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                "requires y to be passed, but the target y is None."
            )

        no_val_X = X is None or (isinstance(X, str) and X == "no_validation")
        no_val_y = y is None or (isinstance(y, str) and y == "no_validation")

        default_check_params = {"estimator": self}
        check_params = {**default_check_params, **check_params}

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")
        elif not no_val_X and no_val_y:
            out = check_X(X, **check_params)
        elif no_val_X and not no_val_y:
            out = _check_y(y, multi_output=False, **check_params)
        else:
            out = check_X_y(X, y, **check_params)

        if not no_val_X:
            self._check_n_features(
                out[0] if isinstance(out, tuple) else out, reset=reset
            )

        return out

    def _check_n_features(self, X, reset):
        """Set the `n_features_in_` attribute, or check against it.

        Uses the `scikit-learn` 1.2.1 `_check_n_features` function as a base.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        reset : bool
            If True, the `n_features_in_` attribute is set to `X.shape[1]`.
            If False and the attribute exists, then check that it is equal to
            `X.shape[1]`. If False and the attribute does *not* exist, then
            the check is skipped.
            .. note::
               It is recommended to call reset=True in `fit` and in the first
               call to `partial_fit`. All other methods that validate `X`
               should set `reset=False`.
        """
        try:
            n_features = _num_features(X)
        except TypeError as e:
            if not reset and hasattr(self, "n_features_in_"):
                raise ValueError(
                    "X does not contain any features to extract, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_[0]} dimensions as input."
                ) from e
            # If the number of features is not defined and reset=True,
            # then we skip this check
            return

        if reset:
            self.n_features_in_ = n_features
            return

        if not hasattr(self, "n_features_in_"):
            # Skip this check if the expected number of expected input features
            # was not recorded by calling fit first. This is typically the case
            # for stateless transformers.
            return

        if n_features[0] != self.n_features_in_[0]:
            raise ValueError(
                f"X has {n_features[0]} dimensions, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_[0]} dimensions as input."
            )

        tags = _safe_tags(self)
        if tags["equal_length_only"] and n_features[1] != self.n_features_in_[1]:
            raise ValueError(
                f"X has {n_features[1]} series length, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_[1]} series length as input."
            )

    def _more_tags(self):
        return _DEFAULT_TAGS

    @classmethod
    def get_test_params(cls, parameter_set=None):
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
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
        if parameter_set is None:
            # default parameters = empty dict
            return {}
        else:
            raise ValueError(
                f"No parameter set {parameter_set} defined for {cls.__name__}"
            )


def clone_estimator(
    base_estimator: BaseEstimator, random_state: Union[None, int, RandomState] = None
) -> BaseEstimator:
    estimator = clone(base_estimator)

    if random_state is not None:
        _set_random_states(estimator, random_state)

    return estimator
