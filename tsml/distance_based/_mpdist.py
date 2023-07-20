"""Matrix Profile Distance 1-NN Classifier."""

__author__ = ["TonyBagnall", "patrickzib", "MatthewMiddlehurst"]
__all__ = ["MPDistClassifier"]

from typing import List, Union

import numpy as np
import stumpy
from sklearn.base import ClassifierMixin
from sklearn.metrics import pairwise
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import check_n_jobs


class MPDistClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Matrix Profile Distance 1-NN Classifier.

    Calculates the matrix profile distance to the training data for each case and
    returns the label of the nearest neighbour.

    Parameters
    ----------
    window : int or float, default=10
        Window size for the matrix profile. If float, will use a proportion of the
        series length.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    class_dictionary_ : dict
        A dictionary mapping class labels to class indices in classes_.

    References
    ----------
    .. [1] Gharghabi, Shaghayegh, et al. "Matrix profile xii: Mpdist: a novel time
        series distance measure to allow data mining in more challenging scenarios."
        2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.

    Examples
    --------
    >>> from tsml.distance_based import MPDistClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> clf = MPDistClassifier()
    >>> clf.fit(X, y)
    MPDistClassifier(...)
    >>> clf.predict(X)
    array([0, 1, 1, 0, 0, 1, 0, 1])
    """

    def __init__(self, window=10, n_jobs=1):
        self.window = window
        self.n_jobs = n_jobs

        super(MPDistClassifier, self).__init__()

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 2D np.ndarray of shape (n_instances, n_timepoints)
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

        self.n_instances_, self.n_timepoints_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._X_train = X.astype(np.float64)
        self._y_train = y

        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 2D np.array of shape (n_instances, n_timepoints)
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

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        window = (
            self.window if self.window >= 1 else int(self.window * self.n_timepoints_)
        )

        distance_matrix = pairwise.pairwise_distances(
            X.astype(np.float64),
            self._X_train,
            metric=(lambda x, y: stumpy.mpdist(x, y, window)),
            n_jobs=self._n_jobs,
        )

        return self._y_train[np.argmin(distance_matrix, axis=1)]

    def _more_tags(self) -> dict:
        return {
            "X_types": ["2darray"],
            "optional_dependency": True,
        }

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
            "window": 0.8,
        }
