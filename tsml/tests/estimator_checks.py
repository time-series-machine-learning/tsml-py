# -*- coding: utf-8 -*-

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

from tsml.tests._sklearn_checks import (
    check_class_weight_classifiers,
    check_classifier_data_not_an_array,
    check_classifier_multioutput,
    check_classifiers_classes,
    check_classifiers_multilabel_output_format_decision_function,
    check_classifiers_multilabel_output_format_predict,
    check_classifiers_multilabel_output_format_predict_proba,
    check_classifiers_multilabel_representation_invariance,
    check_classifiers_one_label,
    check_classifiers_one_label_sample_weights,
    check_classifiers_regression_target,
    check_classifiers_train,
    check_clusterer_compute_labels_predict,
    check_clustering,
    check_complex_data,
    check_decision_proba_consistency,
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_dtype_object,
    check_estimator_get_tags_default_keys,
    check_estimators_dtypes,
    check_estimators_empty_data_messages,
    check_estimators_fit_returns_self,
    check_estimators_nan_inf,
    check_estimators_overwrite_params,
    check_estimators_partial_fit_n_features,
    check_estimators_pickle,
    check_estimators_unfitted,
    check_fit1d,
    check_fit3d_1feature,
    check_fit3d_1sample,
    check_fit3d_predict1d,
    check_fit_check_is_fitted,
    check_fit_idempotent,
    check_fit_non_negative,
    check_fit_score_takes_y,
    check_methods_sample_order_invariance,
    check_methods_subset_invariance,
    check_n_features_in,
    check_no_attributes_set_in_init,
    check_non_transformer_estimators_n_iter,
    check_nonsquare_error,
    check_pipeline_consistency,
    check_regressor_data_not_an_array,
    check_regressor_multioutput,
    check_regressors_int,
    check_regressors_no_decision_function,
    check_regressors_train,
    check_requires_y_none,
    check_sample_weights_invariance,
    check_sample_weights_list,
    check_sample_weights_not_an_array,
    check_sample_weights_not_overwritten,
    check_sample_weights_shape,
    check_supervised_y_2d,
    check_supervised_y_no_nan,
    check_transformer_data_not_an_array,
    check_transformer_general,
    check_transformer_n_iter,
    check_transformer_preserve_dtypes,
    check_transformers_unfitted,
)
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

    yield check_no_attributes_set_in_init
    yield check_estimators_dtypes
    yield check_fit_score_takes_y
    yield check_estimators_fit_returns_self
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)
    yield check_pipeline_consistency
    yield check_estimators_overwrite_params
    yield check_estimators_pickle
    yield check_estimator_get_tags_default_keys
    yield check_parameters_default_constructible
    yield check_fit3d_1sample
    yield check_fit3d_1feature
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters
    yield check_fit_idempotent
    yield check_fit_check_is_fitted

    if has_fit_parameter(estimator, "sample_weight"):
        yield check_sample_weights_not_an_array
        yield check_sample_weights_list
        if not tags["pairwise"]:
            yield check_sample_weights_shape
            yield check_sample_weights_not_overwritten
            yield partial(check_sample_weights_invariance, kind="ones")
            yield partial(check_sample_weights_invariance, kind="zeros")

    if not tags["no_validation"]:
        yield check_complex_data
        yield check_dtype_object
        yield check_estimators_empty_data_messages
        yield check_n_features_in
        yield check_fit1d
        yield check_fit3d_predict1d
        if tags["requires_y"]:
            yield check_requires_y_none

    if not tags["allow_nan"] and not tags["no_validation"]:
        yield check_estimators_nan_inf

    if not tags["non_deterministic"]:
        yield check_methods_sample_order_invariance
        yield check_methods_subset_invariance

    if tags["requires_positive_X"]:
        yield check_fit_non_negative

    if tags["pairwise"]:
        yield check_nonsquare_error

    if not tags["univariate_only"]:
        _yield_multivariate_checks(estimator)

    if not tags["equal_length_only"]:
        _yield_unequal_length_checks(estimator)


def _yield_classifier_checks(classifier):
    tags = _safe_tags(classifier)

    yield check_classifier_data_not_an_array
    yield check_classifiers_one_label
    yield check_classifiers_one_label_sample_weights
    yield check_classifiers_classes
    yield check_estimators_partial_fit_n_features
    yield check_classifiers_train
    yield partial(check_classifiers_train, readonly_memmap=True)
    yield partial(check_classifiers_train, readonly_memmap=True, X_dtype="float32")
    yield check_classifiers_regression_target
    yield check_non_transformer_estimators_n_iter
    yield check_decision_proba_consistency

    if tags["multioutput"]:
        yield check_classifier_multioutput

    if tags["multilabel"]:
        yield check_classifiers_multilabel_representation_invariance
        yield check_classifiers_multilabel_output_format_predict
        yield check_classifiers_multilabel_output_format_predict_proba
        yield check_classifiers_multilabel_output_format_decision_function

    if not tags["no_validation"]:
        yield check_supervised_y_no_nan
        if not tags["multioutput_only"]:
            yield check_supervised_y_2d

    if tags["requires_fit"]:
        yield check_estimators_unfitted

    if "class_weight" in classifier.get_params().keys():
        yield check_class_weight_classifiers


def _yield_regressor_checks(regressor):
    tags = _safe_tags(regressor)

    yield check_regressors_train
    yield partial(check_regressors_train, readonly_memmap=True)
    yield partial(check_regressors_train, readonly_memmap=True, X_dtype="float32")
    yield check_regressor_data_not_an_array
    yield check_estimators_partial_fit_n_features
    yield check_regressors_no_decision_function
    yield check_regressors_int
    yield check_non_transformer_estimators_n_iter

    if tags["multioutput"]:
        yield check_regressor_multioutput

    if not tags["no_validation"]:
        yield check_supervised_y_no_nan
        if not tags["multioutput_only"]:
            yield check_supervised_y_2d

    if tags["requires_fit"]:
        yield check_estimators_unfitted


def _yield_transformer_checks(transformer):
    tags = _safe_tags(transformer)

    yield check_transformer_general
    yield partial(check_transformer_general, readonly_memmap=True)
    yield check_transformer_n_iter

    if not tags["no_validation"]:
        yield check_transformer_data_not_an_array

    if tags["preserves_dtype"]:
        yield check_transformer_preserve_dtypes

    if not tags["stateless"]:
        yield check_transformers_unfitted


def _yield_clustering_checks(clusterer):
    yield check_clusterer_compute_labels_predict
    yield check_clustering
    yield partial(check_clustering, readonly_memmap=True)
    yield check_estimators_partial_fit_n_features

    if not hasattr(clusterer, "transform"):
        yield check_non_transformer_estimators_n_iter


def _yield_multivariate_checks(estimator):
    pass


def _yield_unequal_length_checks(estimator):
    pass
