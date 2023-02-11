# -*- coding: utf-8 -*-
import warnings

from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin, TransformerMixin
from sklearn.exceptions import SkipTestWarning
from sklearn.utils._tags import _safe_tags
from sklearn.utils.estimator_checks import (
    _yield_checks,
    _yield_classifier_checks,
    _yield_clustering_checks,
    _yield_regressor_checks,
    _yield_transformer_checks,
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_fit1d,
    check_fit2d_1feature,
    check_fit2d_1sample,
    check_fit2d_predict1d,
    check_fit_check_is_fitted,
    check_fit_idempotent,
    check_fit_non_negative,
    check_get_params_invariance,
    check_methods_sample_order_invariance,
    check_methods_subset_invariance,
    check_n_features_in,
    check_parameters_default_constructible,
    check_requires_y_none,
    check_set_params,
)


def _yield_all_time_series_checks(estimator):
    name = estimator.__class__.__name__
    tags = _safe_tags(estimator)
    if "2darray" not in tags["X_types"]:
        warnings.warn(
            "Can't test estimator {} which requires input  of type {}".format(
                name, tags["X_types"]
            ),
            SkipTestWarning,
        )
        return
    if tags["_skip_test"]:
        warnings.warn(
            "Explicit SKIP via _skip_test tag for estimator {}.".format(name),
            SkipTestWarning,
        )
        return

    for check in _yield_checks(estimator):
        yield check
    if isinstance(estimator, ClassifierMixin):
        for check in _yield_classifier_checks(estimator):
            yield check
    if isinstance(estimator, RegressorMixin):
        for check in _yield_regressor_checks(estimator):
            yield check
    if isinstance(estimator, TransformerMixin):
        for check in _yield_transformer_checks(estimator):
            yield check
    if isinstance(estimator, ClusterMixin):
        for check in _yield_clustering_checks(estimator):
            yield check
    yield check_parameters_default_constructible
    if not tags["non_deterministic"]:
        yield check_methods_sample_order_invariance
        yield check_methods_subset_invariance
    yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters
    yield check_fit_idempotent
    yield check_fit_check_is_fitted
    if not tags["no_validation"]:
        yield check_n_features_in
        yield check_fit1d
        yield check_fit2d_predict1d
        if tags["requires_y"]:
            yield check_requires_y_none
    if tags["requires_positive_X"]:
        yield check_fit_non_negative
