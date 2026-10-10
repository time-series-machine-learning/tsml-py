"""Random Shapelet Forest (RSF) estimators."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["RandomShapeletForestClassifier", "RandomShapeletForestRegressor"]

import math
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.shapelet_based._rsf_numba import (
    ENTROPY,
    GINI,
    RAND_R_MAX,
    SQUARED_ERROR,
    _apply_tree,
    _build_tree,
)
from tsml.utils.validation import check_n_jobs


class _ShapeletTree:
    """A fitted shapelet tree, storing a copy of each split shapelet."""

    def __init__(self, X, left, right, threshold, shapelet_info, value):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.value = value

        self.shapelet_dim = np.maximum(shapelet_info[:, 1], 0)
        lengths = np.maximum(shapelet_info[:, 3], 0)
        self.shapelet_offset = np.zeros(len(left) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.shapelet_offset[1:])
        self.shapelets = np.empty(self.shapelet_offset[-1], dtype=np.float64)
        for i in np.flatnonzero(left != -1):
            index, dim, start, length = shapelet_info[i]
            self.shapelets[self.shapelet_offset[i] : self.shapelet_offset[i + 1]] = X[
                index, dim, start : start + length
            ]

    def apply(self, X):
        return _apply_tree(
            X,
            self.left,
            self.threshold,
            self.right,
            self.shapelet_dim,
            self.shapelet_offset,
            self.shapelets,
        )

    def predict(self, X):
        return self.value[self.apply(X)]


class _BaseRandomShapeletForest(BaseTimeSeriesEstimator):
    def __init__(
        self,
        n_estimators,
        n_shapelets,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_impurity_decrease,
        min_shapelet_size,
        max_shapelet_size,
        alpha,
        criterion,
        oob_score,
        bootstrap,
        random_state,
        n_jobs,
    ):
        self.n_estimators = n_estimators
        self.n_shapelets = n_shapelets
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.min_shapelet_size = min_shapelet_size
        self.max_shapelet_size = max_shapelet_size
        self.alpha = alpha
        self.criterion = criterion
        self.oob_score = oob_score
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

        super().__init__()

    def _fit_forest(self, X, y_cls, y_reg, sample_weight, n_outputs, criterion):
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        if not 0 <= self.min_shapelet_size <= 1:
            raise ValueError(
                f"min_shapelet_size must be in [0, 1], got {self.min_shapelet_size}."
            )
        if not 0 <= self.max_shapelet_size <= 1:
            raise ValueError(
                f"max_shapelet_size must be in [0, 1], got {self.max_shapelet_size}."
            )
        if self.min_shapelet_size > self.max_shapelet_size:
            raise ValueError(
                f"The min_shapelet_size parameter of {type(self).__name__} must be "
                "<= max_shapelet_size."
            )
        if self.alpha is not None and self.alpha == 0.0:
            raise ValueError("alpha must be None or != 0.")
        if self.oob_score and not self.bootstrap:
            raise ValueError("Out of bag estimation only available if bootstrap=True")

        n_cases, _, n_timepoints = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        self._max_shapelet_length = math.ceil(n_timepoints * self.max_shapelet_size)
        self._min_shapelet_length = math.ceil(n_timepoints * self.min_shapelet_size)
        if self._min_shapelet_length < 2:
            self._min_shapelet_length = 1 if n_timepoints < 2 else 2
        self._max_shapelet_length = max(
            self._max_shapelet_length, self._min_shapelet_length
        )

        if isinstance(self.n_shapelets, str) or callable(self.n_shapelets):
            if self._min_shapelet_length < self._max_shapelet_length:
                possible_shapelets = sum(
                    n_timepoints - length + 1
                    for length in range(
                        self._min_shapelet_length, self._max_shapelet_length
                    )
                )
            else:
                possible_shapelets = n_timepoints - self._min_shapelet_length + 1

            if self.n_shapelets == "log2":
                n_shapelets = int(np.log2(possible_shapelets))
            elif self.n_shapelets == "sqrt":
                n_shapelets = int(np.sqrt(possible_shapelets))
            elif callable(self.n_shapelets):
                n_shapelets = int(self.n_shapelets(possible_shapelets))
            else:
                raise ValueError(
                    "n_shapelets must be an int, 'log2', 'sqrt' or a callable, got "
                    f"{self.n_shapelets}."
                )
        else:
            n_shapelets = self.n_shapelets
        self._n_shapelets = max(1, n_shapelets)

        if sample_weight is None:
            sample_weight = np.ones(n_cases)

        tree_weights, tree_seeds = self._get_tree_weights_and_seeds(sample_weight)
        self.estimators_ = self._build_trees(
            X, y_cls, y_reg, tree_weights, tree_seeds, n_outputs, criterion
        )
        self._estimator_weights = tree_weights

    def _get_tree_weights_and_seeds(self, sample_weight):
        """Get the case weights (bootstrap counts) and random seed for each tree."""
        n_cases = sample_weight.shape[0]
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        tree_weights = []
        tree_seeds = []
        for seed in seeds:
            tree_rng = np.random.RandomState(seed)
            if self.bootstrap:
                indices = tree_rng.randint(0, n_cases, n_cases)
                weights = sample_weight * np.bincount(indices, minlength=n_cases)
            else:
                weights = sample_weight.copy()
            tree_weights.append(weights)
            tree_seeds.append(tree_rng.randint(0, RAND_R_MAX))

        return tree_weights, tree_seeds

    def _build_trees(
        self, X, y_cls, y_reg, tree_weights, tree_seeds, n_outputs, criterion
    ):
        max_depth = np.iinfo(np.int64).max if self.max_depth is None else self.max_depth
        alpha = 0.0 if self.alpha is None else self.alpha

        def _build(weights, seed):
            _, left, right, threshold, shapelet_info, value = _build_tree(
                X,
                y_cls,
                y_reg,
                weights,
                n_outputs,
                criterion,
                self._n_shapelets,
                alpha,
                self._min_shapelet_length,
                self._max_shapelet_length,
                max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                self.min_impurity_decrease,
                seed,
            )
            return _ShapeletTree(X, left, right, threshold, shapelet_info, value)

        return Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(_build)(weights, seed)
            for weights, seed in zip(tree_weights, tree_seeds)
        )

    def _predict_forest(self, X):
        predictions = Parallel(n_jobs=self._n_jobs, prefer="threads")(
            delayed(tree.predict)(X) for tree in self.estimators_
        )
        out = predictions[0].copy()
        for p in predictions[1:]:
            out += p
        return out / len(self.estimators_)

    def _oob_predictions(self, X):
        out = np.zeros((X.shape[0], self.estimators_[0].value.shape[1]))
        n_predictions = np.zeros(X.shape[0])
        for tree, weights in zip(self.estimators_, self._estimator_weights):
            mask = weights == 0
            if mask.any():
                out[mask] += tree.predict(X[mask])
                n_predictions[mask] += 1

        if (n_predictions == 0).any():
            warnings.warn(
                "Some inputs do not have OOB scores. This probably means too few "
                "estimators were used to compute any reliable oob estimates.",
                stacklevel=3,
            )
            n_predictions[n_predictions == 0] = 1

        return out / n_predictions[:, np.newaxis]


class RandomShapeletForestClassifier(ClassifierMixin, _BaseRandomShapeletForest):
    """Random Shapelet Forest (RSF) Classifier.

    An ensemble of randomised shapelet trees [1]_. Each tree is built from a bootstrap
    sample of the training data. At each node a number of shapelets are sampled at
    random from the cases at that node, and the shapelet and distance threshold
    which best split the data are kept. The distance between a shapelet and a series
    is the minimum Euclidean distance between the shapelet and all subsequences of
    the series of the same length. For multivariate data each shapelet is sampled
    from a single random channel.

    This is a port of the wildboar implementation [2]_ restricted to the Euclidean
    distance.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    n_shapelets : int, "log2", "sqrt" or callable, default=10
        The number of shapelets sampled at each node. "log2" and "sqrt" use the
        respective function of the number of possible shapelets. A callable is
        given the number of possible shapelets and must return an int.
    max_depth : int or None, default=None
        The maximum depth of each tree. If None, nodes are expanded until they are
        pure or contain fewer than min_samples_split cases.
    min_samples_split : int, default=2
        The minimum number of cases required to split a node.
    min_samples_leaf : int, default=1
        Nodes with fewer than 2 * min_samples_leaf cases are not split.
    min_impurity_decrease : float, default=0.0
        A node is only split if the weighted decrease in impurity is larger than
        this value.
    min_shapelet_size : float, default=0.0
        The minimum shapelet length as a fraction of the series length. Shapelets
        are at least 2 time points long.
    max_shapelet_size : float, default=1.0
        The maximum shapelet length as a fraction of the series length.
    alpha : float or None, default=None
        If not None, the number of shapelets sampled at a node depends on its depth.
        Positive values sample fewer shapelets near the root, negative values fewer
        near the leaves.
    criterion : {"entropy", "gini"}, default="entropy"
        The criterion used to evaluate the quality of a split.
    oob_score : bool, default=False
        Whether to estimate the accuracy of the forest using out-of-bag cases.
    bootstrap : bool, default=True
        Whether to build each tree from a bootstrap sample of the training data.
    class_weight : dict, "balanced" or None, default=None
        Weights associated with each class. "balanced" weights classes inversely
        proportional to their frequency.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for building the forest.
    n_jobs : int or None, default=None
        The number of threads used to build the trees and make predictions.
        ``-1`` uses all processors.

    Attributes
    ----------
    classes_ : np.ndarray
        The class labels.
    n_classes_ : int
        The number of classes.
    estimators_ : list
        The fitted trees.
    oob_score_ : float
        Out-of-bag accuracy, only set if oob_score is True.
    oob_decision_function_ : np.ndarray of shape (n_cases, n_classes)
        Out-of-bag class probabilities, only set if oob_score is True.

    References
    ----------
    .. [1] Karlsson, I., Papapetrou, P. and Boström, H., 2016. Generalized random
       shapelet forests. Data Mining and Knowledge Discovery, 30(5), pp.1053-1085.
    .. [2] Samsten, I., wildboar: Time series learning with Python.
       https://github.com/wildboar-foundation/wildboar

    Examples
    --------
    >>> from tsml.shapelet_based import RandomShapeletForestClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(random_state=0)
    >>> clf = RandomShapeletForestClassifier(n_estimators=10, random_state=0)
    >>> clf = clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    """

    def __init__(
        self,
        n_estimators=100,
        n_shapelets=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        criterion="entropy",
        oob_score=False,
        bootstrap=True,
        class_weight=None,
        random_state=None,
        n_jobs=None,
    ):
        self.class_weight = class_weight

        super().__init__(
            n_estimators=n_estimators,
            n_shapelets=n_shapelets,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            alpha=alpha,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray) -> object:
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
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_equal_length=True
        )
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_, y_cls = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        if self.criterion == "entropy":
            criterion = ENTROPY
        elif self.criterion == "gini":
            criterion = GINI
        else:
            raise ValueError(
                f"criterion must be 'entropy' or 'gini', got {self.criterion}."
            )

        sample_weight = (
            None
            if self.class_weight is None
            else compute_sample_weight(self.class_weight, y)
        )

        X = np.ascontiguousarray(X, dtype=np.float64)
        y_cls = y_cls.astype(np.int64)
        self._fit_forest(
            X, y_cls, np.zeros(0), sample_weight, self.n_classes_, criterion
        )

        if self.oob_score:
            self.oob_decision_function_ = self._oob_predictions(X)
            self.oob_score_ = accuracy_score(
                y_cls, np.argmax(self.oob_decision_function_, axis=1)
            )

        return self

    def predict(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
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

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
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

        X = self._validate_data(X=X, reset=False, ensure_equal_length=True)
        X = np.ascontiguousarray(self._convert_X(X), dtype=np.float64)

        return self._predict_forest(X)

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
            "n_estimators": 2,
        }


class RandomShapeletForestRegressor(RegressorMixin, _BaseRandomShapeletForest):
    """Random Shapelet Forest (RSF) Regressor.

    An ensemble of randomised shapelet trees [1]_. Each tree is built from a bootstrap
    sample of the training data. At each node a number of shapelets are sampled at
    random from the cases at that node, and the shapelet and distance threshold
    which best split the data are kept. The distance between a shapelet and a series
    is the minimum Euclidean distance between the shapelet and all subsequences of
    the series of the same length. For multivariate data each shapelet is sampled
    from a single random channel.

    This is a port of the wildboar implementation [2]_ restricted to the Euclidean
    distance.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    n_shapelets : int, "log2", "sqrt" or callable, default=10
        The number of shapelets sampled at each node. "log2" and "sqrt" use the
        respective function of the number of possible shapelets. A callable is
        given the number of possible shapelets and must return an int.
    max_depth : int or None, default=None
        The maximum depth of each tree. If None, nodes are expanded until they are
        pure or contain fewer than min_samples_split cases.
    min_samples_split : int, default=2
        The minimum number of cases required to split a node.
    min_samples_leaf : int, default=1
        Nodes with fewer than 2 * min_samples_leaf cases are not split.
    min_impurity_decrease : float, default=0.0
        A node is only split if the weighted decrease in impurity is larger than
        this value.
    min_shapelet_size : float, default=0.0
        The minimum shapelet length as a fraction of the series length. Shapelets
        are at least 2 time points long.
    max_shapelet_size : float, default=1.0
        The maximum shapelet length as a fraction of the series length.
    alpha : float or None, default=None
        If not None, the number of shapelets sampled at a node depends on its depth.
        Positive values sample fewer shapelets near the root, negative values fewer
        near the leaves.
    criterion : {"squared_error"}, default="squared_error"
        The criterion used to evaluate the quality of a split.
    oob_score : bool, default=False
        Whether to estimate the R^2 score of the forest using out-of-bag cases.
    bootstrap : bool, default=True
        Whether to build each tree from a bootstrap sample of the training data.
    random_state : int, RandomState instance or None, default=None
        Seed or random number generator used for building the forest.
    n_jobs : int or None, default=None
        The number of threads used to build the trees and make predictions.
        ``-1`` uses all processors.

    Attributes
    ----------
    estimators_ : list
        The fitted trees.
    oob_score_ : float
        Out-of-bag R^2 score, only set if oob_score is True.
    oob_prediction_ : np.ndarray of shape (n_cases,)
        Out-of-bag predictions, only set if oob_score is True.

    References
    ----------
    .. [1] Karlsson, I., Papapetrou, P. and Boström, H., 2016. Generalized random
       shapelet forests. Data Mining and Knowledge Discovery, 30(5), pp.1053-1085.
    .. [2] Samsten, I., wildboar: Time series learning with Python.
       https://github.com/wildboar-foundation/wildboar

    Examples
    --------
    >>> from tsml.shapelet_based import RandomShapeletForestRegressor
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(random_state=0, regression_target=True)
    >>> reg = RandomShapeletForestRegressor(n_estimators=10, random_state=0)
    >>> reg = reg.fit(X, y)
    >>> y_pred = reg.predict(X)
    """

    def __init__(
        self,
        n_estimators=100,
        n_shapelets=10,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        min_shapelet_size=0.0,
        max_shapelet_size=1.0,
        alpha=None,
        criterion="squared_error",
        oob_score=False,
        bootstrap=True,
        random_state=None,
        n_jobs=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            n_shapelets=n_shapelets,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            min_shapelet_size=min_shapelet_size,
            max_shapelet_size=max_shapelet_size,
            alpha=alpha,
            criterion=criterion,
            oob_score=oob_score,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray) -> object:
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
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_equal_length=True, y_numeric=True
        )
        X = self._convert_X(X)

        if self.criterion != "squared_error":
            raise ValueError(
                f"criterion must be 'squared_error', got {self.criterion}."
            )

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape

        X = np.ascontiguousarray(X, dtype=np.float64)
        y_reg = np.asarray(y, dtype=np.float64)
        self._fit_forest(X, np.zeros(0, dtype=np.int64), y_reg, None, 1, SQUARED_ERROR)

        if self.oob_score:
            self.oob_prediction_ = self._oob_predictions(X)[:, 0]
            self.oob_score_ = r2_score(y_reg, self.oob_prediction_)

        return self

    def predict(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
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

        X = self._validate_data(X=X, reset=False, ensure_equal_length=True)
        X = np.ascontiguousarray(self._convert_X(X), dtype=np.float64)

        return self._predict_forest(X)[:, 0]

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
            "n_estimators": 2,
        }
