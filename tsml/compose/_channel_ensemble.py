"""Channel ensemble estimator classes.

For multivariate time series machine learning. Builds estimators on each dimension
(channel) independently.
"""

__author__ = ["abostrom", "MatthewMiddlehurst"]
__all__ = ["ChannelEnsembleClassifier", "ChannelEnsembleRegressor"]

from abc import ABCMeta
from typing import List, Union

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_random_state

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator


class _BaseChannelEnsemble(BaseTimeSeriesEstimator, metaclass=ABCMeta):
    def __init__(self, estimators, remainder, random_state):
        self.estimators = estimators
        self.remainder = remainder
        self.random_state = random_state

        super(_BaseChannelEnsemble, self).__init__()

    _required_parameters = ["estimators"]

    def _validate_estimators(self, base_type, required_predict_method="predict"):
        self.estimators_ = (
            [self.estimators]
            if isinstance(self.estimators, tuple)
            and isinstance(self.estimators[0], str)
            else self.estimators
        )

        if self.estimators_ is None or len(self.estimators_) == 0:
            raise AttributeError(
                "Invalid estimators attribute, estimators should be a "
                "(string, estimator, dimensions) tuple or list of tuples"
            )

        names, estimators, channels = zip(*self.estimators_)

        self._check_names(names)

        # validate estimators
        self.estimators_ = []
        for i, t in enumerate(estimators):
            if t == "drop":
                continue
            elif (
                not t._estimator_type == base_type
                or not hasattr(t, "fit")
                or not hasattr(t, required_predict_method)
            ):
                raise TypeError(
                    f"All estimators should implement fit, {required_predict_method} "
                    "and be of the correct estimator type. They can be also be 'drop' "
                    f"specifiers. '{t}' (type {type(t)}) doesn't match."
                )
            elif channels[i] == "all-split":
                self.estimators_.extend(
                    [(names[i] + "-" + str(n), t, n) for n in range(self.n_channels_)]
                )
            else:
                self.estimators_.append((names[i], t, channels[i]))

    def _check_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError(f"Names provided are not unique: {list(names)}")

        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                "Estimator names conflict with constructor arguments: "
                f"{sorted(invalid_names)}"
            )

        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                f"Estimator names must not contain __: got f{invalid_names}"
            )

    def _validate_channels(self, X):
        """Convert callable channel specifications."""
        channels = []

        for _, _, channel in self.estimators_:
            if callable(channel):
                channel = channel(X)
            if channel == "all":
                channel = list(range(X[0].shape[0]))

            if not _check_key_type(channel):
                raise ValueError(
                    "Selected estimator channels must be a int, list/tuple of ints or "
                    "slice (or a callable resulting in one of the preceding)."
                )

            channels.append(channel)

        self._channels = channels

    def _validate_remainder(self, base_type):
        """Validate remainder and defines _remainder."""
        is_correct_estimator = (
            hasattr(self.remainder, "fit")
            and hasattr(self.remainder, "predict")
            and self.remainder._estimator_type == base_type
        )
        if self.remainder != "drop" and not is_correct_estimator:
            raise ValueError(
                "The remainder needs to be and valid estimator or the 'drop' keyword. "
                f"{self.remainder} was passed instead."
            )

        cols = []
        all_channels = np.arange(self.n_channels_)
        for channels in self._channels:
            if isinstance(channels, int):
                channels = [channels]
            cols.extend(all_channels[channels])
        remaining_idx = sorted(list(set(all_channels) - set(cols))) or None

        self._remainder = ("remainder", self.remainder, remaining_idx)

    def _get_estimators(self):
        """Generate (name, estimator, channel) tuples."""
        estimators = [
            (name, estimator, channel)
            for (name, estimator, _), channel in zip(self.estimators_, self._channels)
        ]

        # add tuple for remainder
        if self._remainder[2] is not None:
            estimators.append(self._remainder)

        for name, estimator, channel in estimators:
            if estimator == "drop" or _is_empty_channel_selection(channel):
                continue
            yield name, estimator, channel


class ChannelEnsembleClassifier(ClassifierMixin, _BaseChannelEnsemble):
    """Applies classifiers to selected chanels of the input data to form an ensemble.

    This estimator allows different channels or channel subsets of the input
    to be extracted and used to build different classifiers. These classifiers are
    ensembled to form a single output.

    Parameters
    ----------
    estimators : tuple or list of tuples
        Tuple or List of (name, estimator, channel(s)) tuples specifying the classifier
        objects to be applied to subsets of the data.

        name : string
            Name of the estimator and channel combination in the ensemble. Must be
            unique.
        estimator : ClassifierMixin or "drop"
            Estimator must support `fit` and `predict_proba`. Special-cased
            string 'drop' is accepted as well, to indicate to drop the columns.
        channels(s) : int, array-like of int, slice, "all", "all-split" or callable
            Channel(s) to be used with the estimator. If "all", all channels
            are used for the estimator. "all-split" will create a separate estimator
            for each channel. int, array-like of int and slice are used to select
            channels. A callable is passed the input data and should return
            the channel(s) to be used.
    remainder : ClassifierMixin or "drop", default="drop"
        By default, only the specified columns in `estimators` are
        used and combined in the output, and the non-specified
        columns are dropped.
        By setting `remainder` to be an estimator, the remaining
        non-specified columns will use the `remainder` estimator. The
        estimator must support `fit` and `predict`.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set. If input is a list, the length
        of the first series is used.
    n_classes_ : int
        Number of classes. Extracted from the data.
    classes_ : ndarray of shape (n_classes_)
        Holds the label for each class.
    class_dictionary_ : dict
        A dictionary mapping class labels to class indices in classes_.
    estimators_ : list of tuples
        List of (name, estimator, channel(s)) tuples specifying the ensemble
        classifiers.

    See Also
    --------
    ChannelEnsembleRegressor

    Examples
    --------
    >>> from tsml.compose import ChannelEnsembleClassifier
    >>> from tsml.interval_based import TSFClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10, random_state=0)
    >>> reg = ChannelEnsembleClassifier(
    ...     estimators=("tsf", TSFClassifier(n_estimators=2), "all-split"),
    ...     random_state=0,
    ... )
    >>> reg.fit(X, y)
    ChannelEnsembleClassifier(...)
    >>> reg.predict(X)
    array([0, 1, 1, 0, 0, 1, 0, 1])
    """

    def __init__(self, estimators, remainder="drop", random_state=None):
        super(ChannelEnsembleClassifier, self).__init__(
            estimators, remainder, random_state
        )

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
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

        self.n_instances_ = len(X)
        self.n_channels_, self.n_timepoints_ = X[0].shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        self._validate_estimators("classifier", required_predict_method="predict_proba")
        self._validate_channels(X)
        self._validate_remainder("classifier")

        rng = check_random_state(self.random_state)

        estimators_ = []
        for name, estimator, channel in self._get_estimators():
            estimator = _clone_estimator(estimator, random_state=rng)
            estimator.fit(_get_channel(X, channel), y)
            estimators_.append((name, estimator, channel))

        self.estimators_ = estimators_
        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                2D np.ndarray of shape (n_instances, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
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

        return np.array(
            [self.classes_[int(np.argmax(prob))] for prob in self.predict_proba(X)]
        )

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
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

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        probas = np.asarray(
            [
                estimator.predict_proba(_get_channel(X, channel))
                for (_, estimator, channel) in self.estimators_
            ]
        )

        return np.average(probas, axis=0)

    def _more_tags(self) -> dict:
        return {
            "X_types": ["np_list", "3darray"],
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
        from tsml.interval_based import TSFClassifier

        return {
            "estimators": [
                ("tsf1", TSFClassifier(n_estimators=2), 0),
                ("tsf2", TSFClassifier(n_estimators=2), 0),
            ]
        }


class ChannelEnsembleRegressor(RegressorMixin, _BaseChannelEnsemble):
    """Applies regressors to selected chanels of the input data to form an ensemble.

    This estimator allows different channels or channel subsets of the input
    to be extracted and used to build different regressors. These regressors are
    ensembled to form a single output.

    Parameters
    ----------
    estimators : tuple or list of tuples
        Tuple or List of (name, estimator, channel(s)) tuples specifying the regressor
        objects to be applied to subsets of the data.

        name : string
            Name of the estimator and channel combination in the ensemble. Must be
            unique.
        estimator : RegressorMixin or "drop"
            Estimator must support `fit` and `predict`. Special-cased
            string 'drop' is accepted as well, to indicate to drop the columns.
        channels(s) : int, array-like of int, slice, "all", "all-split" or callable
            Channel(s) to be used with the estimator. If "all", all channels
            are used for the estimator. "all-split" will create a separate estimator
            for each channel. int, array-like of int and slice are used to select
            channels. A callable is passed the input data and should return
            the channel(s) to be used.
    remainder : RegressorMixin or "drop", default="drop"
        By default, only the specified columns in `estimators` are
        used and combined in the output, and the non-specified
        columns are dropped.
        By setting `remainder` to be an estimator, the remaining
        non-specified columns will use the `remainder` estimator. The
        estimator must support `fit` and `predict`.

    Attributes
    ----------
    n_instances_ : int
        The number of train cases in the training set.
    n_channels_ : int
        The number of dimensions per case in the training set.
    n_timepoints_ : int
        The length of each series in the training set.
    estimators_ : list of tuples
        List of (name, estimator, channel(s)) tuples specifying the ensemble
        regressors.

    See Also
    --------
    ChannelEnsembleClassifier

    Examples
    --------
    >>> from tsml.compose import ChannelEnsembleRegressor
    >>> from tsml.interval_based import TSFRegressor
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(n_samples=8, series_length=10,
    ...                              regression_target=True, random_state=0)
    >>> reg = ChannelEnsembleRegressor(
    ...     estimators=("tsf", TSFRegressor(n_estimators=2), "all-split"),
    ...     random_state=0,
    ... )
    >>> reg.fit(X, y)
    ChannelEnsembleRegressor(...)
    >>> reg.predict(X)
    array([0.31798318, 1.41426301, 1.06414747, 0.6924721 , 0.56660146,
           1.26538944, 0.52324808, 1.0939405 ])
    """

    def __init__(self, estimators, remainder="drop", random_state=None):
        super(ChannelEnsembleRegressor, self).__init__(
            estimators, remainder, random_state
        )

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
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

        self.n_instances_ = len(X)
        self.n_channels_, self.n_timepoints_ = X[0].shape

        self._validate_estimators("regressor")
        self._validate_channels(X)
        self._validate_remainder("regressor")

        rng = check_random_state(self.random_state)

        estimators_ = []
        for name, estimator, channel in self._get_estimators():
            estimator = _clone_estimator(estimator, random_state=rng)
            estimator.fit(_get_channel(X, channel), y)
            estimators_.append((name, estimator, channel))

        self.estimators_ = estimators_
        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints) or
                list of size (n_instances) of 2D np.ndarray (n_channels,
                n_timepoints_i), where n_timepoints_i is length of series i
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted target labels.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        preds = np.asarray(
            [
                estimator.predict(_get_channel(X, channel))
                for (_, estimator, channel) in self.estimators_
            ]
        )

        return np.average(preds, axis=0)

    def _more_tags(self) -> dict:
        return {
            "X_types": ["np_list", "3darray"],
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
        from tsml.interval_based import TSFRegressor

        return {
            "estimators": [
                ("tsf1", TSFRegressor(n_estimators=2), 0),
                ("tsf2", TSFRegressor(n_estimators=2), 0),
            ]
        }


def _is_empty_channel_selection(channel):
    """Check if column selection is empty."""
    if hasattr(channel, "__len__"):
        return len(channel) == 0
    else:
        return False


def _get_channel(X, key):
    """Get time series channel(s) from input data X."""
    if isinstance(X, np.ndarray):
        return X[:, key]
    else:
        li = [x[key] for x in X]
        if li[0].ndim == 1:
            li = [x.reshape(1, -1) for x in li]
        return li


def _check_key_type(key):
    """
    Check that key is an int, list/tuple of ints or slice.

    Parameters
    ----------
    key : object
        The channel specification to check
    """
    if isinstance(key, int):
        return True
    elif isinstance(key, slice):
        return isinstance(key.start, (int, type(None))) and isinstance(
            key.stop, (int, type(None))
        )
    elif isinstance(key, (list, tuple)):
        return all(isinstance(x, int) for x in key)
    else:
        return False
