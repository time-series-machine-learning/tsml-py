"""Functional Principal Component Analysis."""

__author__ = ["dguijo", "MatthewMiddlehurst"]
__all__ = ["FPCATransformer"]

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class FPCATransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Apply FPCA on a set of time X to transform the X into a reduced dimension.

    Wraps the Functional Principal Component Analysis from scikit-fda.

    Parameters
    ----------
    n_components: int, default=10
        Number of principal components to keep from functional principal component
        analysis.
    centering: bool, default=True
        Set to ``False`` when the functional data is already known to be centered
        and there is no need to center it. Otherwise, the mean of the functional
        data object is calculated and the data centered before fitting.
    bspline: bool, default=False
        Set to ``True`` to use a B-spline basis for the functional principal
        component analysis.
    n_basis: int, default=None
        Number of functions in the basis. Only used if `bspline` is `True`.
    order: int, default=None
        Order of the splines. One greater than their degree. Only used if
        `bspline` is `True`.
    """

    def __init__(
        self,
        n_components=10,
        centering=True,
        bspline=False,
        n_basis=None,
        order=None,
    ):
        self.n_components = n_components
        self.centering = centering
        self.bspline = bspline
        self.n_basis = n_basis
        self.order = order

        _check_optional_dependency("scikit-fda", "skfda", self)

        super(FPCATransformer, self).__init__()

    def fit_transform(self, X, y=None):
        """
        Convert the X to its functional form.

        fit the transformer per dimension and transform the X based on the
        number of coefficients
        :param X: A set of time X with the shape N x L x D
        :return: transformed X with top n_components functional principal components
        """
        from skfda import FDataGrid
        from skfda.preprocessing.dim_reduction import FPCA
        from skfda.representation.basis import BSplineBasis

        X = self._fit_setup(X)

        X_t = np.zeros((self.n_instances_, self.n_dims_, self._n_components))
        for j in range(self.n_dims_):
            # represent the time X in functional form
            fd = FDataGrid(X[:, j, :], list(range(self.series_length_)))

            # smooth the X if needed
            if self.bspline:
                basis = BSplineBasis(n_basis=self._n_basis, order=self.order)
                fd = fd.to_basis(basis)

            individual_transformer = FPCA(
                n_components=self._n_components,
                centering=self.centering,
            )

            X_t[:, j, :] = individual_transformer.fit_transform(fd)
            self.transformers_.append(individual_transformer)

        return X_t

    def fit(self, X, y=None):
        from skfda import FDataGrid
        from skfda.preprocessing.dim_reduction import FPCA
        from skfda.representation.basis import BSplineBasis

        X = self._fit_setup(X)

        for j in range(self.n_dims_):
            # represent the time X in functional form
            fd = FDataGrid(X[:, j, :], list(range(self.series_length_)))

            # smooth the X if needed
            if self.bspline:
                basis = BSplineBasis(n_basis=self._n_basis, order=self.order)
                fd = fd.to_basis(basis)

            individual_transformer = FPCA(
                n_components=self._n_components,
                centering=self.centering,
            )

            individual_transformer.fit(fd)
            self.transformers_.append(individual_transformer)

        return self

    def _fit_setup(self, X):
        X = self._validate_data(X=X, ensure_min_samples=2)
        X = self._convert_X(X)

        if self.bspline:
            # n_basis has to be larger or equal to order
            self._n_basis = max(self.n_basis, self.order)
            # n_components has to be less than or equal to n_basis
            self._n_components = min(self.n_basis, self.n_components)
        else:
            self._n_components = self.n_components

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self.transformers_ = []

        return X

    def transform(self, X):
        """
        Transform the X based on the number of coefficients.

        :param X: A set of time X with the shape N x L x D
        :return: transformed X with top n_components functional
            principal components
        """
        from skfda import FDataGrid
        from skfda.representation.basis import BSplineBasis

        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        X_t = np.zeros((X.shape[0], self.n_dims_, self._n_components))
        for j in range(self.n_dims_):
            individual_transformer = self.transformers_[j]

            # represent the time X in functional form
            fd = FDataGrid(X[:, j, :], list(range(self.series_length_)))

            # smooth the X if needed
            if self.bspline:
                basis = BSplineBasis(n_basis=self._n_basis, order=self.order)
                fd = fd.to_basis(basis)

            X_t[:, j, :] = individual_transformer.transform(fd)

        return X_t

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return {
            "n_components": 5,
        }
