"""Functional Principal Component Analysis (FPCA) transformer.

The transformer is a port of the functional principal component analysis and B-spline
basis smoothing in scikit-fda 0.10.1 (BSD 3 clause, Grupo de Aprendizaje Automático -
Universidad Autónoma de Madrid), restricted to series observed at evenly spaced time
points. The quadrature weights, B-spline basis and centering follow the original so
that the transformed data matches it up to floating point error.
"""

__author__ = ["dguijo", "MatthewMiddlehurst"]
__all__ = ["FPCATransformer"]

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lstsq, solve_triangular
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator


class FPCATransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Functional Principal Component Analysis (FPCA) transformer.

    Each series is treated as a function observed at the time points
    ``0, 1, ..., series_length - 1`` and is transformed into its scores on the first
    ``n_components`` functional principal components [1]_. Channels are transformed
    independently.

    By default the series are used as observed, with the integrals of functions
    approximated using Simpson's rule. If ``bspline`` is ``True``, each series is first
    smoothed with a least squares fit to a B-spline basis with evenly spaced knots.

    Parameters
    ----------
    n_components: int, default=10
        Number of principal components to keep from functional principal component
        analysis. Cannot be greater than the number of cases, or the series length if
        `bspline` is `False`. If `bspline` is `True`, at most `n_basis` components are
        kept.
    centering: bool, default=True
        Set to ``False`` when the functional data is already known to be centered
        and there is no need to center it. Otherwise, the mean of the functional
        data object is calculated and the data centered before fitting.
    bspline: bool, default=False
        Set to ``True`` to use a B-spline basis for the functional principal
        component analysis.
    n_basis: int, default=None
        Number of functions in the basis. Only used if `bspline` is `True`, where it
        is required. Increased to `order` if smaller.
    order: int, default=None
        Order of the splines. One greater than their degree. Only used if
        `bspline` is `True`, where it is required.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_dims_ : int
        The number of dimensions per case in the training set.
    series_length_ : int
        The length of each series in the training set.
    components_ : ndarray of shape (n_dims_, n_components, n_features)
        The principal components of each dimension. These are the values of the
        component functions at each time point if `bspline` is `False`, and their
        B-spline basis coefficients otherwise. `n_features` is `series_length_` if
        `bspline` is `False` and the number of basis functions otherwise.
    mean_ : ndarray of shape (n_dims_, n_features)
        The mean function of each dimension in the training set, in the same form as
        `components_`.
    explained_variance_ : ndarray of shape (n_dims_, n_components)
        The amount of variance explained by each component.
    explained_variance_ratio_ : ndarray of shape (n_dims_, n_components)
        The proportion of variance explained by each component.
    singular_values_ : ndarray of shape (n_dims_, n_components)
        The singular values corresponding to each component.

    Notes
    -----
    The principal components are always those of the mean centered training data.
    `centering` only changes whether the training mean is subtracted from a series
    before its scores are calculated.

    References
    ----------
    .. [1] Ramsay, J. O. and Silverman, B. W. (2005). Functional Data Analysis.
       Springer, chapter 8.

    Examples
    --------
    >>> from tsml.transformations import FPCATransformer
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, _ = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> tnf = FPCATransformer(n_components=3)
    >>> tnf.fit_transform(X).shape
    (8, 1, 3)
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

        super().__init__()

    def fit(self, X, y=None):
        """Fit the functional principal components of each dimension.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : None
            Ignored.

        Returns
        -------
        self :
            Reference to self.
        """
        X = self._fit_setup(X)

        n_features = self._n_basis if self.bspline else self.series_length_
        self.components_ = np.zeros((self.n_dims_, self._n_components, n_features))
        self.mean_ = np.zeros((self.n_dims_, n_features))
        self.explained_variance_ = np.zeros((self.n_dims_, self._n_components))
        self.explained_variance_ratio_ = np.zeros((self.n_dims_, self._n_components))
        self.singular_values_ = np.zeros((self.n_dims_, self._n_components))

        if self.bspline:
            gram_cholesky = np.linalg.cholesky(self._gram)
        else:
            sqrt_weights = np.sqrt(self._weights)

        for j in range(self.n_dims_):
            pca = PCA(n_components=self._n_components)

            if self.bspline:
                # smooth the series, giving their basis coefficients
                coefs = lstsq(self._basis_values, X[:, j, :].T)[0].T

                self.mean_[j] = coefs.mean(axis=0)
                if self.centering:
                    coefs = coefs - self.mean_[j]

                # PCA using the inner product of the basis
                pca.fit(coefs @ gram_cholesky)
                self.components_[j] = solve_triangular(
                    gram_cholesky.T, pca.components_.T, lower=False
                ).T
            else:
                self.mean_[j] = X[:, j, :].mean(axis=0)

                # PCA using the inner product given by the quadrature weights, the
                # PCA centers the series whether centering is set or not
                pca.fit(X[:, j, :] * sqrt_weights)
                self.components_[j] = pca.components_ / sqrt_weights

            self.explained_variance_[j] = pca.explained_variance_
            self.explained_variance_ratio_[j] = pca.explained_variance_ratio_
            self.singular_values_[j] = pca.singular_values_

        return self

    def transform(self, X):
        """Transform X into its functional principal component scores.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The data to transform.

        Returns
        -------
        X_t : 3D np.ndarray of shape (n_instances, n_channels, n_components)
            The scores of each dimension on its top functional principal components.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False, ensure_equal_length=True)
        X = self._convert_X(X)

        X_t = np.zeros((X.shape[0], self.n_dims_, self._n_components))
        for j in range(self.n_dims_):
            if self.bspline:
                # smooth the series, giving their basis coefficients
                fd = lstsq(self._basis_values, X[:, j, :].T)[0].T
            else:
                fd = X[:, j, :]

            if self.centering:
                fd = fd - self.mean_[j]

            # inner product of each function with each component
            if self.bspline:
                X_t[:, j, :] = fd @ self._gram @ self.components_[j].T
            else:
                X_t[:, j, :] = (fd * self._weights) @ self.components_[j].T

        return X_t

    def _fit_setup(self, X):
        X = self._validate_data(X=X, ensure_min_samples=2, ensure_equal_length=True)
        X = self._convert_X(X)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape

        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1, got {self.n_components}."
            )

        if self.bspline:
            if self.n_basis is None or self.order is None:
                raise ValueError("n_basis and order must be set if bspline is True.")
            if self.n_basis < 1 or self.order < 1:
                raise ValueError(
                    f"n_basis and order must be at least 1, got {self.n_basis} and "
                    f"{self.order}."
                )

            # n_basis has to be larger or equal to order
            self._n_basis = max(self.n_basis, self.order)
            # n_components has to be less than or equal to n_basis
            self._n_components = min(self.n_basis, self.n_components)
        else:
            self._n_components = self.n_components

        if self._n_components > self.n_instances_:
            raise ValueError(
                f"n_components ({self._n_components}) cannot be greater than the "
                f"number of cases ({self.n_instances_})."
            )
        if not self.bspline and self._n_components > self.series_length_:
            raise ValueError(
                f"n_components ({self._n_components}) cannot be greater than the "
                f"series length ({self.series_length_})."
            )

        if self.bspline:
            degree = self.order - 1
            # evenly spaced knots, repeated at the ends as the splines are not
            # required to be zero there
            knots = np.linspace(0, self.series_length_ - 1, self._n_basis - degree + 1)
            padded_knots = np.pad(knots, degree, mode="edge")

            self._basis_values = BSpline.design_matrix(
                np.arange(self.series_length_, dtype=np.float64), padded_knots, degree
            ).toarray()

            # integrals of the products of each pair of basis functions, Gauss-Legendre
            # quadrature with order points between each pair of knots is exact
            nodes, node_weights = np.polynomial.legendre.leggauss(self.order)
            half_width = np.diff(knots)[:, np.newaxis] / 2
            points = (knots[:-1, np.newaxis] + half_width * (nodes + 1)).ravel()
            point_weights = (half_width * node_weights).ravel()
            values = BSpline.design_matrix(points, padded_knots, degree).toarray()
            self._gram = (values.T * point_weights) @ values
        else:
            # Simpson's rule weights for evenly spaced points. An even number of
            # points uses the correction for the final interval scipy has used since
            # 1.11, calculated here so the transform does not change with scipy.
            self._weights = np.full(self.series_length_, 2 / 3)
            self._weights[1::2] = 4 / 3
            self._weights[[0, -1]] = 1 / 3
            if self.series_length_ == 2:
                self._weights[:] = 1 / 2
            elif self.series_length_ % 2 == 0:
                self._weights[-3:] = [5 / 4, 1, 5 / 12]

        return X

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
