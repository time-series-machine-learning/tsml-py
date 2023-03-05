# -*- coding: utf-8 -*-
"""A shapelet transform classifier (STC).

Shapelet transform classifier pipeline that simply performs a (configurable) shapelet
transform then builds (by default) a rotation forest classifier on the output.
"""

__author__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator, _clone_estimator
from tsml.transformations.shapelet_transform import RandomShapeletTransform
from tsml.utils.validation import check_n_jobs
from tsml.vector import RotationForestClassifier


class ShapeletTransformClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """A shapelet transform classifier (STC).

    Implementation of the binary shapelet transform classifier pipeline along the lines
    of [1][2] but with random shapelet sampling. Transforms the data using the
    configurable `RandomShapeletTransform` and then builds a `RotationForest`
    classifier.

    As some implementations and applications contract the transformation solely,
    contracting is available for the transform only and both classifier and transform.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to ``<= max_shapelets``, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to ``n_classes_ / max_shapelets``. If `None`, uses the
        minimum between ``10 * n_instances_`` and `1000`.
    max_shapelet_length : int or None, default=None
        Lower bound on candidate shapelet lengths for the transform. If ``None``, no
        max length is used
    estimator : BaseEstimator or None, default=None
        Base estimator for the ensemble, can be supplied a sklearn `BaseEstimator`. If
        `None` a default `RotationForest` classifier is used.
    transform_limit_in_minutes : int, default=0
        Time contract to limit transform time in minutes for the shapelet transform,
        overriding `n_shapelet_samples`. A value of `0` means ``n_shapelet_samples``
        is used.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding ``n_shapelet_samples``
        and ``transform_limit_in_minutes``. The ``estimator`` will only be contracted if
        a ``time_limit_in_minutes parameter`` is present. Default of `0` means
        ``n_shapelet_samples`` or ``transform_limit_in_minutes`` is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when contracting the transform with
        ``transform_limit_in_minutes`` or ``time_limit_in_minutes``.
    save_transformed_data : bool, default=False
        Save the data transformed in fit in ``transformed_data_`` for use in
        ``_get_train_probs``.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        `-1` means using all processors.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets in the transform.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.

    Attributes
    ----------
    classes_ : list
        The unique class labels in the training set.
    n_classes_ : int
        The number of unique classes in the training set.
    fit_time_  : int
        The time (in milliseconds) for ``fit`` to run.
    n_instances_ : int
        The number of train cases in the training set.
    n_dims_ : int
        The number of dimensions per case in the training set.
    series_length_ : int
        The length of each series in the training set.
    transformed_data_ : list of shape (n_estimators) of ndarray
        The transformed training dataset for all classifiers. Only saved when
        ``save_transformed_data`` is `True`.

    See Also
    --------
    RandomShapeletTransform : The randomly sampled shapelet transform.
    RotationForest : The default rotation forest classifier used.

    Notes
    -----
    For the Java version, see
    `tsml <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/classifiers/shapelet_based/ShapeletTransformClassifier.java>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge Centered
       Systems, 32, 2017.
    """

    def __init__(
        self,
        n_shapelet_samples=10000,
        max_shapelets=None,
        max_shapelet_length=None,
        estimator=None,
        transform_limit_in_minutes=0,
        time_limit_in_minutes=0,
        contract_max_n_shapelet_samples=np.inf,
        save_transformed_data=False,
        n_jobs=1,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.max_shapelet_length = max_shapelet_length
        self.estimator = estimator

        self.transform_limit_in_minutes = transform_limit_in_minutes
        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples
        self.save_transformed_data = save_transformed_data

        self.random_state = random_state
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        super(ShapeletTransformClassifier, self).__init__()

    def fit(self, X, y):
        """Fit ShapeletTransformClassifier to training data.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The training data.
        y : array-like, shape = [n_instances]
            The class labels.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_".
        """
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, classVal in enumerate(self.classes_):
            self.class_dictionary_[classVal] = index

        if len(self.classes_) == 1:
            return self

        self._n_jobs = check_n_jobs(self.n_jobs)

        self._transform_limit_in_minutes = 0
        if self.time_limit_in_minutes > 0:
            # contracting 2/3 transform (with 1/5 of that taken away for final
            # transform), 1/3 classifier
            third = self.time_limit_in_minutes / 3
            self._classifier_limit_in_minutes = third
            self._transform_limit_in_minutes = (third * 2) / 5 * 4
        elif self.transform_limit_in_minutes > 0:
            self._transform_limit_in_minutes = self.transform_limit_in_minutes

        self._transformer = RandomShapeletTransform(
            n_shapelet_samples=self.n_shapelet_samples,
            max_shapelets=self.max_shapelets,
            max_shapelet_length=self.max_shapelet_length,
            time_limit_in_minutes=self._transform_limit_in_minutes,
            contract_max_n_shapelet_samples=self.contract_max_n_shapelet_samples,
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

        self._estimator = _clone_estimator(
            RotationForestClassifier() if self.estimator is None else self.estimator,
            self.random_state,
        )

        if isinstance(self._estimator, RotationForestClassifier):
            self._estimator.save_transformed_data = self.save_transformed_data

        m = getattr(self._estimator, "n_jobs", None)
        if m is not None:
            self._estimator.n_jobs = self._n_jobs

        m = getattr(self._estimator, "time_limit_in_minutes", None)
        if m is not None and self.time_limit_in_minutes > 0:
            self._estimator.time_limit_in_minutes = self._classifier_limit_in_minutes

        X_t = self._transformer.fit_transform(X, y)

        if self.save_transformed_data:
            self.transformed_data_ = X_t

        self._estimator.fit(X_t, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predictions for.

        Returns
        -------
        y : array-like, shape = [n_instances]
            Predicted class labels.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)

        return self._estimator.predict(self._transformer.transform(X))

    def predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_min_series_length=3)

        m = getattr(self._estimator, "predict_proba", None)
        if callable(m):
            return self._estimator.predict_proba(self._transformer.transform(X))
        else:
            dists = np.zeros((X.shape[0], self.n_classes_))
            preds = self._estimator.predict(self._transformer.transform(X))
            for i in range(0, X.shape[0]):
                dists[i, self.class_dictionary_[preds[i]]] = 1
            return dists

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
        from sklearn.ensemble import RandomForestClassifier

        if parameter_set == "results_comparison":
            return {
                "estimator": RandomForestClassifier(n_estimators=5),
                "n_shapelet_samples": 50,
                "max_shapelets": 10,
                "batch_size": 10,
            }
        else:
            return {
                "estimator": RotationForestClassifier(n_estimators=2),
                "n_shapelet_samples": 10,
                "max_shapelets": 3,
                "batch_size": 5,
                "save_transformed_data": True,
            }
