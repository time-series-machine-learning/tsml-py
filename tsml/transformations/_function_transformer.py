""""""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["FunctionTransformer"]


from sklearn.base import TransformerMixin
from sklearn.preprocessing._function_transformer import _identity

from tsml.base import BaseTimeSeriesEstimator


class FunctionTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Constructs a transformer from an arbitrary callable.

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    Read more in the :ref:`User Guide <function_transformer>`.

    stripped down 1.2.2

    Parameters
    ----------
    func : callable, default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.
    validate : bool, default=False
        Indicate that the input X array should be checked before calling
        ``func``. The possibilities are:

        - If False, there is no input validation.
        - If True, then X will be converted to a 2-dimensional NumPy array or
          sparse matrix. If the conversion is not possible an exception is
          raised.
    kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to func.
    """

    def __init__(
        self,
        func=None,
        validate=True,
        kw_args=None,
    ):
        self.func = func
        self.validate = validate
        self.kw_args = kw_args

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
            self._validate_data(X, ensure_min_series_length=1)
        else:
            self._check_n_features(X, True)

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
        if self.validate:
            X = self._validate_data(X, reset=False, ensure_min_series_length=1)

        func = self.func if self.func is not None else _identity

        return func(X, **(self.kw_args if self.kw_args else {}))

    def _more_tags(self) -> dict:
        return {
            "no_validation": not self.validate,
            "requires_fit": False,
            "X_types": ["3darray", "2darray", "np_list"],
        }
