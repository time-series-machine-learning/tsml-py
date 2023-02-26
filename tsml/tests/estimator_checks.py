# -*- coding: utf-8 -*-
"""Checks for all estimators in tsml."""

__author__ = ["MatthewMiddlehurst"]

import warnings
from functools import partial

from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import SkipTestWarning
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_parameters_default_constructible,
    check_set_params,
)
from sklearn.utils.validation import has_fit_parameter

import tsml.tests._sklearn_checks as patched_checks
from tsml.utils._tags import _safe_tags
from tsml.utils.validation import is_clusterer, is_transformer


def _yield_all_time_series_checks(estimator):
    name = estimator.__class__.__name__
    tags = _safe_tags(estimator)

    if "3darray" not in tags["X_types"]:
        warnings.warn(
            "Can't test estimator {} which requires input  of type {}".format(
                name, tags["X_types"]
            ),
            SkipTestWarning,
        )
        return

    if tags["_skip_test"]:
        warnings.warn(
            f"Explicit SKIP via _skip_test tag for estimator {name}.",
            SkipTestWarning,
        )
        return

    for check in _yield_checks(estimator):
        yield check

    if is_classifier(estimator):
        for check in _yield_classifier_checks(estimator):
            yield check

    if is_regressor(estimator):
        for check in _yield_regressor_checks(estimator):
            yield check

    if is_transformer(estimator):
        for check in _yield_transformer_checks(estimator):
            yield check

    if is_clusterer(estimator):
        for check in _yield_clustering_checks(estimator):
            yield check


def _yield_checks(estimator):
    """sklearn"""
    tags = _safe_tags(estimator)

    yield patched_checks.check_no_attributes_set_in_init
    yield patched_checks.check_estimators_dtypes
    yield patched_checks.check_fit_score_takes_y
    yield patched_checks.check_estimators_fit_returns_self
    yield partial(
        patched_checks.check_estimators_fit_returns_self, readonly_memmap=True
    )
    yield patched_checks.check_pipeline_consistency
    yield patched_checks.check_estimators_overwrite_params
    yield patched_checks.check_estimators_pickle
    yield patched_checks.check_estimator_get_tags_default_keys
    yield check_parameters_default_constructible
    yield patched_checks.check_fit3d_1sample
    yield patched_checks.check_fit3d_1feature
    yield check_get_params_invariance
    yield check_set_params
    yield patched_checks.check_dict_unchanged
    yield patched_checks.check_dont_overwrite_parameters
    yield patched_checks.check_fit_idempotent
    yield patched_checks.check_fit_check_is_fitted

    if has_fit_parameter(estimator, "sample_weight"):
        yield patched_checks.check_sample_weights_not_an_array
        yield patched_checks.check_sample_weights_list
        if not tags["pairwise"]:
            yield patched_checks.check_sample_weights_shape
            yield patched_checks.check_sample_weights_not_overwritten
            yield partial(patched_checks.check_sample_weights_invariance, kind="ones")
            yield partial(patched_checks.check_sample_weights_invariance, kind="zeros")

    if not tags["no_validation"]:
        yield patched_checks.check_complex_data
        yield patched_checks.check_dtype_object
        yield patched_checks.check_estimators_empty_data_messages
        yield patched_checks.check_n_features_in
        yield patched_checks.check_fit1d
        yield patched_checks.check_fit3d_predict1d
        if tags["requires_y"]:
            yield patched_checks.check_requires_y_none

    if not tags["allow_nan"] and not tags["no_validation"]:
        yield patched_checks.check_estimators_nan_inf

    if not tags["non_deterministic"]:
        yield patched_checks.check_methods_sample_order_invariance
        yield patched_checks.check_methods_subset_invariance

    if tags["requires_positive_X"]:
        yield patched_checks.check_fit_non_negative

    if tags["pairwise"]:
        yield patched_checks.check_nonsquare_error

    if not tags["univariate_only"]:
        _yield_multivariate_checks(estimator)

    if not tags["equal_length_only"]:
        _yield_unequal_length_checks(estimator)


def _yield_classifier_checks(classifier):
    tags = _safe_tags(classifier)

    yield patched_checks.check_classifier_data_not_an_array
    yield patched_checks.check_classifiers_one_label
    yield patched_checks.check_classifiers_one_label_sample_weights
    yield patched_checks.check_classifiers_classes
    yield patched_checks.check_estimators_partial_fit_n_features
    yield patched_checks.check_classifiers_train
    yield partial(patched_checks.check_classifiers_train, readonly_memmap=True)
    yield partial(
        patched_checks.check_classifiers_train, readonly_memmap=True, X_dtype="float32"
    )
    yield patched_checks.check_classifiers_regression_target
    yield patched_checks.check_non_transformer_estimators_n_iter
    yield patched_checks.check_decision_proba_consistency

    if tags["multioutput"]:
        yield patched_checks.check_classifier_multioutput

    if tags["multilabel"]:
        yield patched_checks.check_classifiers_multilabel_representation_invariance
        yield patched_checks.check_classifiers_multilabel_output_format_predict
        yield patched_checks.check_classifiers_multilabel_output_format_predict_proba
        yield patched_checks.check_classifiers_multilabel_output_format_decision_function

    if not tags["no_validation"]:
        yield patched_checks.check_supervised_y_no_nan
        if not tags["multioutput_only"]:
            yield patched_checks.check_supervised_y_2d

    if tags["requires_fit"]:
        yield patched_checks.check_estimators_unfitted

    if "class_weight" in classifier.get_params().keys():
        yield patched_checks.check_class_weight_classifiers


def _yield_regressor_checks(regressor):
    tags = _safe_tags(regressor)

    yield patched_checks.check_regressors_train
    yield partial(patched_checks.check_regressors_train, readonly_memmap=True)
    yield partial(
        patched_checks.check_regressors_train, readonly_memmap=True, X_dtype="float32"
    )
    yield patched_checks.check_regressor_data_not_an_array
    yield patched_checks.check_estimators_partial_fit_n_features
    yield patched_checks.check_regressors_no_decision_function
    yield patched_checks.check_regressors_int
    yield patched_checks.check_non_transformer_estimators_n_iter

    if tags["multioutput"]:
        yield patched_checks.check_regressor_multioutput

    if not tags["no_validation"]:
        yield patched_checks.check_supervised_y_no_nan
        if not tags["multioutput_only"]:
            yield patched_checks.check_supervised_y_2d

    if tags["requires_fit"]:
        yield patched_checks.check_estimators_unfitted


def _yield_transformer_checks(transformer):
    tags = _safe_tags(transformer)

    yield patched_checks.check_transformer_general
    yield partial(patched_checks.check_transformer_general, readonly_memmap=True)
    yield patched_checks.check_transformer_n_iter

    if not tags["no_validation"]:
        yield patched_checks.check_transformer_data_not_an_array

    if tags["preserves_dtype"]:
        yield patched_checks.check_transformer_preserve_dtypes

    if not tags["stateless"]:
        yield patched_checks.check_transformers_unfitted


def _yield_clustering_checks(clusterer):
    yield patched_checks.check_clusterer_compute_labels_predict
    yield patched_checks.check_clustering
    yield partial(patched_checks.check_clustering, readonly_memmap=True)
    yield patched_checks.check_estimators_partial_fit_n_features

    if not hasattr(clusterer, "transform"):
        yield patched_checks.check_non_transformer_estimators_n_iter


def _yield_multivariate_checks(estimator):
    pass


def _yield_unequal_length_checks(estimator):
    pass
