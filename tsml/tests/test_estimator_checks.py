"""Checks for all estimators in tsml."""

__author__ = ["MatthewMiddlehurst"]

import inspect
import warnings
from functools import partial

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import SkipTestWarning
from sklearn.utils._testing import SkipTest, assert_allclose, ignore_warnings
from sklearn.utils.estimator_checks import (
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

import tsml.tests.test_estimators_sklearn as patched_checks
import tsml.utils.testing as test_utils
from tsml.base import _clone_estimator
from tsml.utils._tags import _safe_tags
from tsml.utils.validation import is_clusterer, is_transformer


def _yield_all_time_series_checks(estimator):
    name = estimator.__class__.__name__
    tags = _safe_tags(estimator)

    if tags["_skip_test"]:
        warnings.warn(
            f"Explicit SKIP via _skip_test tag for estimator {name}.",
            SkipTestWarning,
            stacklevel=2,
        )
        return

    yield from _yield_checks(estimator)

    if is_classifier(estimator):
        yield from _yield_classifier_checks(estimator)

    if is_regressor(estimator):
        yield from _yield_regressor_checks(estimator)

    if is_transformer(estimator):
        yield from _yield_transformer_checks(estimator)

    if is_clusterer(estimator):
        yield from _yield_clustering_checks(estimator)


def _yield_checks(estimator):
    tags = _safe_tags(estimator)

    yield check_no_attributes_set_in_init
    yield patched_checks.check_estimators_dtypes
    yield patched_checks.check_fit_score_takes_y
    yield patched_checks.check_estimators_fit_returns_self
    yield partial(
        patched_checks.check_estimators_fit_returns_self, readonly_memmap=True
    )
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
    yield patched_checks.check_fit_check_is_fitted
    yield check_fit3d_predict2d

    if not tags["no_validation"]:
        yield check_estimator_input_types
        yield patched_checks.check_complex_data
        yield patched_checks.check_dtype_object
        yield patched_checks.check_estimators_empty_data_messages
        yield patched_checks.check_n_features_in
        yield patched_checks.check_fit1d
        yield patched_checks.check_fit3d_predict1d
        if tags["requires_y"]:
            yield patched_checks.check_requires_y_none
        if not tags["equal_length_only"]:
            yield check_n_features_unequal

    if not tags["allow_nan"] and not tags["no_validation"]:
        yield patched_checks.check_estimators_nan_inf

    if not tags["non_deterministic"]:
        yield patched_checks.check_pipeline_consistency
        yield patched_checks.check_fit_idempotent
        yield patched_checks.check_methods_sample_order_invariance
        yield patched_checks.check_methods_subset_invariance

    if not tags["univariate_only"]:
        yield check_estimator_handles_multivariate_data
    else:
        yield check_estimator_cannot_handle_multivariate_data

    if not tags["equal_length_only"]:
        yield check_estimator_handles_unequal_data
    else:
        yield check_estimator_cannot_handle_unequal_data


def _yield_classifier_checks(classifier):
    tags = _safe_tags(classifier)

    yield patched_checks.check_estimator_data_not_an_array
    yield patched_checks.check_classifiers_one_label
    yield patched_checks.check_classifiers_classes
    yield patched_checks.check_classifiers_train
    yield partial(patched_checks.check_classifiers_train, readonly_memmap=True)
    yield partial(
        patched_checks.check_classifiers_train, readonly_memmap=True, X_dtype="float32"
    )
    yield patched_checks.check_decision_proba_consistency

    if not tags["no_validation"]:
        yield patched_checks.check_classifiers_regression_target
        yield patched_checks.check_supervised_y_no_nan
        yield patched_checks.check_supervised_y_2d

    if tags["requires_fit"]:
        yield patched_checks.check_estimators_unfitted


def _yield_regressor_checks(regressor):
    tags = _safe_tags(regressor)

    yield patched_checks.check_estimator_data_not_an_array
    yield patched_checks.check_regressors_train
    yield partial(patched_checks.check_regressors_train, readonly_memmap=True)
    yield partial(
        patched_checks.check_regressors_train, readonly_memmap=True, X_dtype="float32"
    )
    yield patched_checks.check_regressors_no_decision_function
    yield patched_checks.check_regressors_int

    if not tags["no_validation"]:
        yield patched_checks.check_supervised_y_no_nan
        yield patched_checks.check_supervised_y_2d

    if tags["requires_fit"]:
        yield patched_checks.check_estimators_unfitted


def _yield_transformer_checks(transformer):
    tags = _safe_tags(transformer)

    yield patched_checks.check_transformer_general
    yield partial(patched_checks.check_transformer_general, readonly_memmap=True)

    if not tags["no_validation"]:
        yield patched_checks.check_transformer_data_not_an_array

    if tags["preserves_dtype"]:
        yield patched_checks.check_transformer_preserve_dtypes

    if tags["requires_fit"]:
        yield patched_checks.check_estimators_unfitted

    if not tags["requires_y"]:
        yield check_transformer_fit_no_y


def _yield_clustering_checks(clusterer):
    yield patched_checks.check_clusterer_compute_labels_predict
    yield patched_checks.check_clustering
    yield partial(patched_checks.check_clustering, readonly_memmap=True)


def check_estimator_input_types(name, estimator_orig):
    """Check estimators with more than one input type returns the same results.

    Validates type tag.
    """
    valid_types = ["3darray", "2darray", "np_list"]
    type_tag = _safe_tags(estimator_orig, key="X_types")
    for t in type_tag:
        assert t in valid_types, f"Invalid X_types tag value {t} for {name}"

    # If there is only one input type other tests will cover it
    if len(type_tag) == 1:
        return

    if _safe_tags(estimator_orig, key="non_deterministic"):
        raise SkipTest(name + " is non deterministic")

    # test a single function with this priority
    def _get_func(est):
        if hasattr(est, "predict_proba"):
            return est.predict_proba
        elif hasattr(est, "predict"):
            return est.predict
        elif hasattr(est, "transform"):
            return est.transform

    X, y = test_utils.generate_3d_test_data()
    first_result = None

    if "3darray" in type_tag:
        estimator = _clone_estimator(estimator_orig, 1)

        estimator.fit(X, y)
        current_result = _get_func(estimator)(X)

        # no results to compare against yet
        first_result = current_result

    if "2darray" in type_tag:
        estimator = _clone_estimator(estimator_orig, 1)
        X_t = X.reshape((X.shape[0], -1))

        func = _get_func(estimator)
        estimator.fit(X_t, y)
        current_result = func(X_t)

        if first_result is None:
            first_result = current_result
        else:
            # if series to series transform reshape to match first result
            if first_result.ndim == 3 and current_result.ndim == 2:
                current_result = current_result.reshape(first_result.shape)

            msg = (
                f"Results for {name} differ between input types using function {func}."
            )
            assert_allclose(first_result, current_result, err_msg=msg)

    if "np_list" in type_tag:
        estimator = _clone_estimator(estimator_orig, 1)
        X_t = [x for x in X]

        func = _get_func(estimator)
        estimator.fit(X_t, y)
        current_result = func(X_t)

        # we must already have a first result if we reach here
        msg = f"Results for {name} differ between input types using function {func}."
        assert_allclose(first_result, current_result, err_msg=msg)


@ignore_warnings(category=FutureWarning)
def check_fit3d_predict2d(name, estimator_orig):
    """Check that fitting on 3D and predicting on 2D produces correct output size."""
    X, y = test_utils.generate_3d_test_data()
    X_2d = X.reshape((X.shape[0], -1))  # Convert 3D to 2D

    estimator = _clone_estimator(estimator_orig)
    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        predictions = estimator.predict(X_2d)

        assert predictions.shape[0] == X_2d.shape[0], (
            f"Predict output size {predictions.shape[0]} does not match "
            f"input size {X_2d.shape[0]} for estimator {name}."
        )
    if hasattr(estimator, "transform"):
        tform = estimator.transform(X)

        assert len(tform) == len(X), (
            f"Transform output size {len(tform)} does not match "
            f"input size {len(X)} for estimator {name}."
        )


@ignore_warnings(category=FutureWarning)
def check_estimator_cannot_handle_multivariate_data(name, estimator_orig):
    """Check univariate_only estimators raise an error for multivariate data."""
    X, y = test_utils.generate_3d_test_data(n_channels=2)
    estimator = _clone_estimator(estimator_orig)

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if (
            "not all arrays are univariate" not in str(e).lower()
            and "but it is not univariate" not in str(e).lower()
        ):
            raise AssertionError(
                f"Estimator {name} raised an error, but it does not mention "
                f"'not all arrays are univariate' or 'but it is not univariate'."
            )
    else:
        raise AssertionError(
            f"Estimator {name} did not raise an error for multivariate data."
        )


@ignore_warnings(category=FutureWarning)
def check_estimator_handles_multivariate_data(name, estimator_orig):
    """Check equal_length_only estimators raise an error for unequal data."""
    X, y = test_utils.generate_3d_test_data(n_channels=2)
    estimator = _clone_estimator(estimator_orig)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        estimator.predict_proba(X)
    if hasattr(estimator, "transform"):
        estimator.transform(X)

    tags = _safe_tags(estimator_orig)
    if not tags["no_validation"]:
        X2, _ = test_utils.generate_3d_test_data(n_channels=4)

        if hasattr(estimator, "predict"):
            try:
                estimator.predict(X2)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"Estimator {name} did not raise an error for different channels."
                )
        if hasattr(estimator, "predict_proba"):
            try:
                estimator.predict_proba(X2)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"Estimator {name} did not raise an error for different channels."
                )
        if hasattr(estimator, "transform"):
            try:
                estimator.transform(X2)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"Estimator {name} did not raise an error for different channels."
                )


@ignore_warnings(category=FutureWarning)
def check_estimator_cannot_handle_unequal_data(name, estimator_orig):
    """Check equal_length_only estimators raise an error for unequal data."""
    X, y = test_utils.generate_unequal_test_data()
    estimator = _clone_estimator(estimator_orig)

    try:
        estimator.fit(X, y)
    except ValueError as e:
        if "not all arrays have the same series length" not in str(e).lower():
            raise AssertionError(
                f"Estimator {name} raised an error, but it does not mention "
                f"'not all arrays have the same series length'."
            )
    else:
        raise AssertionError(
            f"Estimator {name} did not raise an error for unequal length data."
        )


@ignore_warnings(category=FutureWarning)
def check_estimator_handles_unequal_data(name, estimator_orig):
    """Check estimators can handle unequal length data."""
    X, y = test_utils.generate_unequal_test_data()
    estimator = _clone_estimator(estimator_orig)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)
    if hasattr(estimator, "predict_proba"):
        estimator.predict_proba(X)
    if hasattr(estimator, "transform"):
        estimator.transform(X)


@ignore_warnings(category=FutureWarning)
def check_n_features_unequal(name, estimator_orig):
    """Check that estimators handling unequal length data do not have n_features_in_."""
    X, y = test_utils.generate_unequal_test_data()
    estimator = _clone_estimator(estimator_orig)

    estimator.fit(X, y)

    lengths = [x.shape[1] for x in X]
    assert estimator.n_features_in_[0] == X[0].shape[0]
    assert estimator.n_features_in_[1] == np.min(lengths)
    assert estimator.n_features_in_[2] == np.max(lengths)


@ignore_warnings(category=FutureWarning)
def check_transformer_fit_no_y(name, estimator_orig):
    """Check that transformers treat y=None as valid input."""
    X, y = test_utils.generate_3d_test_data(n_samples=5)

    estimator = _clone_estimator(estimator_orig)
    estimator.fit(X)

    estimator = _clone_estimator(estimator_orig)
    estimator.fit_transform(X)

    assert "y" in inspect.getfullargspec(estimator.fit).args
