"""Random Dilated Shapelet Transform (RDST) estimators.

Random Dilated Shapelet Transform estimator pipelines that simply perform a random
shapelet dilated transform and build (by default) a RidgeCV on the output.
"""

__author__ = ["MatthewMiddlehurst", "baraline"]
__all__ = ["RDSTClassifier", "RDSTRegressor"]

import warnings
from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations import RandomDilatedShapeletTransformer
from tsml.utils.validation import check_n_jobs


class RDSTClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """A random dilated shapelet transform (RDST) classifier.

    Implementation of the random dilated shapelet transform classifier pipeline
    along the lines of [1][2]. Transforms the data using the
    `RandomDilatedShapeletTransform` and then builds a `RidgeClassifierCV` classifier
    with standard scalling.

    This is a duplicate of the original implementation in aeon, adapted for bugfixing
    and experimentation. All credit to the original author @baraline for the
    implementation.

    Parameters
    ----------
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn `BaseEstimator`. If
        `None` a default `RidgeClassifierCV` classifier is used with standard scalling.
    max_shapelets : int, default=10000
        The maximum number of shapelet to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array, default=None
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set. If None, the shapelets length will be equal to
        min(max(2,series_length//2),11).
    proba_normalization : float, default=0.8
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    threshold_percentiles : array, default=None
        The two perceniles used to select the threshold used to compute the Shapelet
        Occurrence feature. If None, the 5th and the 10th percentiles (i.e. [5,10])
        will be used.
    alpha_similarity : float, default=0.5
        The strenght of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5, and the maximum is
        1. Value above it have no effect for now.
    use_prime_dilations : bool, default=False
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed-up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    save_transformed_data : bool, default=False
        Save the data transformed in fit in ``transformed_data_``.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    class_dictionary_ : dict
        A dictionary mapping class labels to class indices in classes_.
    transformed_data_ : list of shape (n_estimators) of ndarray
        The transformed training dataset for all classifiers. Only saved when
        ``save_transformed_data`` is `True`.

    See Also
    --------
    RDSTRegressor
    RandomDilatedShapeletTransformer

    References
    ----------
    .. [1] Antoine Guillaume et al. "Random Dilated Shapelet Transform: A New Approach
       for Time Series Shapelets", Pattern Recognition and Artificial Intelligence.
       ICPRAI 2022.
    .. [2] Antoine Guillaume, "Time series classification with shapelets: Application
       to predictive maintenance on event logs", PhD Thesis, University of Orléans,
       2023.

    Examples
    --------
    >>> from tsml.shapelet_based import RDSTClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> clf = RDSTClassifier(random_state=0)
    >>> clf.fit(X, y)
    RDSTClassifier(...)
    >>> pred = clf.predict(X)
    """

    def __init__(
        self,
        max_shapelets=10000,
        shapelet_lengths=None,
        proba_normalization=0.8,
        threshold_percentiles=None,
        alpha_similarity=0.5,
        use_prime_dilations=False,
        estimator=None,
        save_transformed_data=False,
        n_jobs=1,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.shapelet_lengths = shapelet_lengths
        self.proba_normalization = proba_normalization
        self.threshold_percentiles = threshold_percentiles
        self.alpha_similarity = alpha_similarity
        self.use_prime_dilations = use_prime_dilations
        self.estimator = estimator
        self.save_transformed_data = save_transformed_data
        self.n_jobs = n_jobs
        self.random_state = random_state

        super(RDSTClassifier, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = RandomDilatedShapeletTransformer(
            max_shapelets=self.max_shapelets,
            shapelet_lengths=self.shapelet_lengths,
            proba_normalization=self.proba_normalization,
            threshold_percentiles=self.threshold_percentiles,
            alpha_similarity=self.alpha_similarity,
            use_prime_dilations=self.use_prime_dilations,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )

        self._estimator = _clone_estimator(
            make_pipeline(
                StandardScaler(with_mean=False),
                RidgeClassifierCV(alphas=np.logspace(-4, 4, 20)),
            )
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        if X_t.shape[1] == 0:
            warnings.warn("No shapelets found in training data.", stacklevel=2)
            self._no_atts = True
            self._majority_class = np.argmax(np.unique(y, return_counts=True)[1])
        else:
            self._no_atts = False
            self._estimator.fit(X_t, y)

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        if self._no_atts:
            return np.repeat([self.classes_[self._majority_class]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        X_t = self._transformer.transform(X)
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

        return self._estimator.predict(X_t)

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        if self._no_atts:
            p = np.zeros((X.shape[0], self.n_classes_))
            p[:, self._majority_class] = 1
            return p

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        X_t = self._transformer.transform(X)
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(X_t)
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(X_t)
            for i in range(0, X.shape[0]):
                dists[i, self.class_dictionary_[preds[i]]] = 1
            return dists

    def _more_tags(self) -> dict:
        return {"non_deterministic": True}

    @classmethod
    def get_test_params(
        cls, parameter_set: Union[str, None] = None
    ) -> Union[dict, List[dict]]:
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
            "max_shapelets": 10,
        }


class RDSTRegressor(RegressorMixin, BaseTimeSeriesEstimator):
    """A random dilated shapelet transform (RDST) regressor.

    Implementation of the random dilated shapelet transform pipeline
    along the lines of [1][2]. Transforms the data using the
    `RandomDilatedShapeletTransform` and then builds a `RidgeCV` regressor
    with standard scalling.

    Parameters
    ----------
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn `BaseEstimator`. If
        `None` a default `RidgeCV` regressor is used with standard scalling.
    max_shapelets : int, default=10000
        The maximum number of shapelet to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array, default=None
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set. If None, the shapelets length will be equal to
        min(max(2,series_length//2),11).
    proba_normalization : float, default=0.8
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    threshold_percentiles : array, default=None
        The two perceniles used to select the threshold used to compute the Shapelet
        Occurrence feature. If None, the 5th and the 10th percentiles (i.e. [5,10])
        will be used.
    alpha_similarity : float, default=0.5
        The strenght of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5, and the maximum is
        1. Value above it have no effect for now.
    use_prime_dilations : bool, default=False
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed-up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    save_transformed_data : bool, default=False
        Save the data transformed in fit in ``transformed_data_``.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    transformed_data_ : list of shape (n_estimators) of ndarray
        The transformed training dataset for all regressors. Only saved when
        ``save_transformed_data`` is `True`.

    See Also
    --------
    RDSTClassifier
    RandomDilatedShapeletTransformer

    References
    ----------
    .. [1] Antoine Guillaume et al. "Random Dilated Shapelet Transform: A New Approach
       for Time Series Shapelets", Pattern Recognition and Artificial Intelligence.
       ICPRAI 2022.
    .. [2] Antoine Guillaume, "Time series classification with shapelets: Application
       to predictive maintenance on event logs", PhD Thesis, University of Orléans,
       2023.

    Examples
    --------
    >>> from tsml.shapelet_based import RDSTRegressor
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10,
    ...                              regression_target=True, random_state=0)
    >>> reg = RDSTRegressor(random_state=0)
    >>> reg.fit(X, y)
    RDSTRegressor(...)
    >>> pred = reg.predict(X)
    """

    def __init__(
        self,
        max_shapelets=10000,
        shapelet_lengths=None,
        proba_normalization=0.8,
        threshold_percentiles=None,
        alpha_similarity=0.5,
        use_prime_dilations=False,
        estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.max_shapelets = max_shapelets
        self.shapelet_lengths = shapelet_lengths
        self.proba_normalization = proba_normalization
        self.threshold_percentiles = threshold_percentiles
        self.alpha_similarity = alpha_similarity
        self.use_prime_dilations = use_prime_dilations
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state

        super(RDSTRegressor, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The target labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2, y_numeric=True)
        X = self._convert_X(X)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transformer = RandomDilatedShapeletTransformer(
            max_shapelets=self.max_shapelets,
            shapelet_lengths=self.shapelet_lengths,
            proba_normalization=self.proba_normalization,
            threshold_percentiles=self.threshold_percentiles,
            alpha_similarity=self.alpha_similarity,
            use_prime_dilations=self.use_prime_dilations,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )

        self._estimator = _clone_estimator(
            make_pipeline(
                StandardScaler(with_mean=False),
                RidgeCV(alphas=np.logspace(-4, 4, 20)),
            )
            if self.estimator is None
            else self.estimator,
            self.random_state,
        )

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        X_t = self._transformer.fit_transform(X, y)
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

        if X_t.shape[1] == 0:
            warnings.warn("No shapelets found in training data.", stacklevel=2)
            self._no_atts = True
            self._y_mean = np.mean(y)
        else:
            self._no_atts = False
            self._estimator.fit(X_t, y)

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted target labels.
        """
        check_is_fitted(self)

        if self._no_atts:
            return np.full(X.shape[0], self._y_mean)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        X_t = self._transformer.transform(X)
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)

        return self._estimator.predict(X_t)

    def _more_tags(self) -> dict:
        return {"non_deterministic": True}

    @classmethod
    def get_test_params(
        cls, parameter_set: Union[str, None] = None
    ) -> Union[dict, List[dict]]:
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
            "max_shapelets": 10,
        }
