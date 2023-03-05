# -*- coding: utf-8 -*-
"""Patched estimator checks originating from scikit-learn"""

__author__ = ["MatthewMiddlehurst"]

import pickle
import warnings
from copy import deepcopy
from inspect import signature

import joblib
import numpy as np
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)
from scipy.stats import rankdata
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.datasets import make_multilabel_classification, make_regression
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from sklearn.utils import IS_PYPY, shuffle
from sklearn.utils._testing import (
    SkipTest,
    _get_args,
    assert_allclose,
    assert_allclose_dense_sparse,
    assert_raise_message,
    create_memmap_backed_data,
    ignore_warnings,
    raises,
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _choose_check_classifiers_labels,
    _enforce_estimator_tags_X,
    _enforce_estimator_tags_y,
    _is_pairwise_metric,
    _is_public_parameter,
    _NotAnArray,
    _regression_dataset,
    check_estimators_data_not_an_array,
)
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _num_samples, check_is_fitted, has_fit_parameter

import tsml.utils.testing as test_utils
from tsml.utils._tags import _DEFAULT_TAGS, _safe_tags


@ignore_warnings(category=FutureWarning)
def check_supervised_y_no_nan(name, estimator_orig):
    """
    Checks that the Estimator targets are not NaN.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    X, _ = test_utils.generate_test_data()

    for value in [np.nan, np.inf]:
        y = np.full(10, value)
        y = _enforce_estimator_tags_y(estimator, y)

        module_name = estimator.__module__
        if module_name.startswith("sklearn.") and not (
            "test_" in module_name or module_name.endswith("_testing")
        ):
            # In scikit-learn we want the error message to mention the input
            # name and be specific about the kind of unexpected value.
            if np.isinf(value):
                match = (
                    r"Input (y|Y) contains infinity or a value too large for"
                    r" dtype\('float64'\)."
                )
            else:
                match = r"Input (y|Y) contains NaN."
        else:
            # Do not impose a particular error message to third-party libraries.
            match = None
        err_msg = (
            f"Estimator {name} should have raised error on fitting array y with inf"
            " value."
        )
        with raises(ValueError, match=match, err_msg=err_msg):
            estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_sample_weights_not_an_array(name, estimator_orig):
    """
    check that estimators will accept a 'sample_weight' parameter of type _NotAnArray
    in the 'fit' function.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    estimator = clone(estimator_orig)
    X = _NotAnArray(_enforce_estimator_tags_X(estimator_orig, X))
    y = _NotAnArray(y)
    weights = _NotAnArray([1] * 12)
    if _safe_tags(estimator, key="multioutput_only"):
        y = _NotAnArray(y.data.reshape(-1, 1))
    estimator.fit(X, y, sample_weight=weights)


@ignore_warnings(category=FutureWarning)
def check_sample_weights_list(name, estimator_orig):
    """
    check that estimators will accept a 'sample_weight' parameter of
    type list in the 'fit' function.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    estimator = clone(estimator_orig)
    rnd = np.random.RandomState(0)
    n_samples = 30
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator, y)
    sample_weight = [3] * n_samples
    # Test that estimators don't raise any exception
    estimator.fit(X, y, sample_weight=sample_weight)


@ignore_warnings(category=FutureWarning)
def check_sample_weights_shape(name, estimator_orig):
    """
    check that estimators raise an error if sample_weight
    shape mismatches the input

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y, sample_weight=np.ones(len(y)))

    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones(2 * len(y)))

    with raises(ValueError):
        estimator.fit(X, y, sample_weight=np.ones((len(y), 2)))


@ignore_warnings(category=FutureWarning)
def check_sample_weights_invariance(name, estimator_orig, kind="ones"):
    """
    For kind="ones" check that the estimators yield same results for
    unit weights and no weights
    For kind="zeros" check that setting sample_weight to 0 is equivalent
    to removing corresponding samples.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X1, y1 = test_utils.generate_test_data()

    estimator1 = clone(estimator_orig)
    estimator2 = clone(estimator_orig)
    set_random_state(estimator1, random_state=0)
    set_random_state(estimator2, random_state=0)

    if kind == "ones":
        X2 = X1
        y2 = y1
        sw2 = np.ones(shape=len(y1))
        err_msg = (
            f"For {name} sample_weight=None is not equivalent to sample_weight=ones"
        )
    elif kind == "zeros":
        # Construct a dataset that is very different to (X, y) if weights
        # are disregarded, but identical to (X, y) given weights.
        X2 = np.vstack([X1, X1 + 1])
        y2 = np.hstack([y1, 3 - y1])
        sw2 = np.ones(shape=len(y1) * 2)
        sw2[len(y1) :] = 0
        X2, y2, sw2 = shuffle(X2, y2, sw2, random_state=0)

        err_msg = (
            f"For {name}, a zero sample_weight is not equivalent to removing the sample"
        )
    else:  # pragma: no cover
        raise ValueError

    y1 = _enforce_estimator_tags_y(estimator1, y1)
    y2 = _enforce_estimator_tags_y(estimator2, y2)

    estimator1.fit(X1, y=y1, sample_weight=None)
    estimator2.fit(X2, y=y2, sample_weight=sw2)

    for method in ["predict", "predict_proba", "decision_function", "transform"]:
        if hasattr(estimator_orig, method):
            X_pred1 = getattr(estimator1, method)(X1)
            X_pred2 = getattr(estimator2, method)(X1)
            assert_allclose_dense_sparse(X_pred1, X_pred2, err_msg=err_msg)


def check_sample_weights_not_overwritten(name, estimator_orig):
    """
    check that estimators don't override the passed sample_weight parameter

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    estimator = clone(estimator_orig)
    set_random_state(estimator, random_state=0)

    y = _enforce_estimator_tags_y(estimator, y)

    sample_weight_original = np.ones(y.shape[0])
    sample_weight_original[0] = 10.0

    sample_weight_fit = sample_weight_original.copy()

    estimator.fit(X, y, sample_weight=sample_weight_fit)

    err_msg = f"{name} overwrote the original `sample_weight` given during fit"
    assert_allclose(sample_weight_fit, sample_weight_original, err_msg=err_msg)


@ignore_warnings(category=(FutureWarning, UserWarning))
def check_dtype_object(name, estimator_orig):
    """
    check that estimators treat dtype object as numeric if possible

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)
    X = X.astype(object)
    tags = _safe_tags(estimator_orig)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        estimator.predict(X)

    if hasattr(estimator, "transform"):
        estimator.transform(X)

    with raises(Exception, match="Unknown label type", may_pass=True):
        estimator.fit(X, y.astype(object))

    if "string" not in tags["X_types"]:
        X[0, 0] = {"foo": "bar"}
        msg = "argument must be a string.* number"
        with raises(TypeError, match=msg):
            estimator.fit(X, y)
    else:
        # Estimators supporting string will not call np.asarray to convert the
        # data to numeric and therefore, the error will not be raised.
        # Checking for each element dtype in the input array will be costly.
        # Refer to #11401 for full discussion.
        estimator.fit(X, y)


def check_complex_data(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    rng = np.random.RandomState()
    # check that estimators raise an exception on providing complex data
    X = rng.uniform(size=10) + 1j * rng.uniform(size=10)
    X = X.reshape(-1, 1)

    # Something both valid for classification and regression
    y = rng.randint(low=0, high=2, size=10) + 1j
    estimator = clone(estimator_orig)
    set_random_state(estimator, random_state=0)
    with raises(ValueError, match="Complex data not supported"):
        estimator.fit(X, y)


@ignore_warnings
def check_dict_unchanged(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    if hasattr(estimator, "n_components"):
        estimator.n_components = 1

    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    if hasattr(estimator, "n_best"):
        estimator.n_best = 1

    set_random_state(estimator, 1)

    estimator.fit(X, y)
    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert estimator.__dict__ == dict_before, (
                "Estimator changes __dict__ during %s" % method
            )


@ignore_warnings(category=FutureWarning)
def check_dont_overwrite_parameters(name, estimator_orig):
    """
    check that fit method only changes or sets private attributes

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    estimator = clone(estimator_orig)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [
        key for key in dict_after_fit.keys() if _is_public_parameter(key)
    ]

    attrs_added_by_fit = [
        key for key in public_keys_after_fit if key not in dict_before_fit.keys()
    ]

    # check that fit doesn't add any public attribute
    assert not attrs_added_by_fit, (
        "Estimator adds public attribute(s) during"
        " the fit method."
        " Estimators are only allowed to add private attributes"
        " either started with _ or ended"
        " with _ but %s added" % ", ".join(attrs_added_by_fit)
    )

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
    ]

    assert not attrs_changed_by_fit, (
        "Estimator changes public attribute(s) during"
        " the fit method. Estimators are only allowed"
        " to change attributes started"
        " or ended with _, but"
        " %s changed" % ", ".join(attrs_changed_by_fit)
    )


@ignore_warnings(category=FutureWarning)
def check_fit3d_predict1d(name, estimator_orig):
    """
    check by fitting a 3d array and predicting with a 1d array

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            assert_raise_message(
                ValueError, "Reshape your data", getattr(estimator, method), X[0][0]
            )


@ignore_warnings(category=FutureWarning)
def check_methods_subset_invariance(name, estimator_orig):
    """Check smaller batches of data for predict methods does not impact results.

    Check that method gives invariant results if applied on mini batches or the whole
    set

    Modified version of the scikit-learn 1.2.1 function with the name for time series
    data.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:
        msg = f"{method} of {name} is not invariant when applied to a subset."

        if hasattr(estimator, method):
            func = getattr(estimator, method)
            result_full = func(X)
            result_by_batch = [func(batch.reshape(1, 1, X.shape[2])) for batch in X]

            # func can output tuple (e.g. score_samples)
            if type(result_full) == tuple:
                result_full = result_full[0]
                result_by_batch = list(map(lambda x: x[0], result_by_batch))

            assert_allclose(
                np.ravel(result_full), np.ravel(result_by_batch), atol=1e-7, err_msg=msg
            )


@ignore_warnings(category=FutureWarning)
def check_methods_sample_order_invariance(name, estimator_orig):
    """
    check that method gives invariant results if applied
    on a subset with different sample order

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 2

    set_random_state(estimator, 1)
    estimator.fit(X, y)

    idx = np.random.permutation(X.shape[0])

    for method in [
        "predict",
        "transform",
        "decision_function",
        "score_samples",
        "predict_proba",
    ]:
        msg = (
            "{method} of {name} is not invariant when applied to a dataset"
            "with different sample order."
        ).format(method=method, name=name)

        if hasattr(estimator, method):
            assert_allclose_dense_sparse(
                getattr(estimator, method)(X)[idx],
                getattr(estimator, method)(X[idx]),
                atol=1e-9,
                err_msg=msg,
            )


@ignore_warnings
def check_fit3d_1sample(name, estimator_orig):
    """Check for fitting an estimator with only 1 sample.

    Check that fitting a 3d array with only one sample either works or
    returns an informative message. The error message should either mention
    the number of samples or the number of classes.

    Modified version of the scikit-learn 1.2.1 function with the name for time series
    data.
    """
    X, y = test_utils.generate_test_data(n_samples=1)

    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator, 1)

    msgs = [
        "1 sample",
        "n_samples = 1",
        "n_samples=1",
        "one sample",
        "1 class",
        "one class",
    ]
    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings
def check_fit3d_1feature(name, estimator_orig):
    """Check for fitting an estimator with only 1 series length.

    Check fitting a 3d array with only 1 feature either works or returns
    informative message

    Modified version of the scikit-learn 1.2.1 function with the name for time series
    data.
    """
    X, y = test_utils.generate_test_data(series_length=1)

    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)

    msgs = ["series length 1", "series length = 1", "series length=1"]
    with raises(ValueError, match=msgs, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings
def check_fit1d(name, estimator_orig):
    """Check fitting 1d X array raises a ValueError.

    Modified version of the scikit-learn 1.2.1 function with the name for time series
    data.
    """
    rnd = np.random.RandomState()
    X = 3 * rnd.uniform(size=(20))
    y = X.astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator, 1)
    with raises(ValueError):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_general(name, transformer, readonly_memmap=False):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(transformer, X)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    _check_transformer(name, transformer, X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_data_not_an_array(name, transformer):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(transformer, X)
    this_X = _NotAnArray(X)
    this_y = _NotAnArray(np.asarray(y))
    _check_transformer(name, transformer, this_X, this_y)
    # try the same with some list
    _check_transformer(name, transformer, X.tolist(), y.tolist())


@ignore_warnings(category=FutureWarning)
def check_transformers_unfitted(name, transformer):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, _ = test_utils.generate_test_data()

    transformer = clone(transformer)
    with raises(
        (AttributeError, ValueError),
        err_msg=(
            "The unfitted "
            f"transformer {name} does not raise an error when "
            "transform is called. Perhaps use "
            "check_is_fitted in transform."
        ),
    ):
        transformer.transform(X)


def _check_transformer(name, transformer_orig, X, y):
    n_samples, n_dimensions, n_features = np.asarray(X).shape
    transformer = clone(transformer_orig)
    set_random_state(transformer)

    # fit

    transformer.fit(X, y)
    # fit_transform method should work on non fitted estimator
    transformer_clone = clone(transformer)
    X_pred = transformer_clone.fit_transform(X, y=y)

    if isinstance(X_pred, tuple):
        for x_pred in X_pred:
            assert x_pred.shape[0] == n_samples
    else:
        # check for consistent n_samples
        assert X_pred.shape[0] == n_samples

    if hasattr(transformer, "transform"):
        X_pred2 = transformer.transform(X)
        X_pred3 = transformer.fit_transform(X, y=y)

        if _safe_tags(transformer_orig, key="non_deterministic"):
            msg = name + " is non deterministic"
            raise SkipTest(msg)
        if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
            for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred2,
                    atol=1e-2,
                    err_msg="fit_transform and transform outcomes not consistent in %s"
                    % transformer,
                )
                assert_allclose_dense_sparse(
                    x_pred,
                    x_pred3,
                    atol=1e-2,
                    err_msg="consecutive fit_transform outcomes not consistent in %s"
                    % transformer,
                )
        else:
            assert_allclose_dense_sparse(
                X_pred,
                X_pred2,
                err_msg="fit_transform and transform outcomes not consistent in %s"
                % transformer,
                atol=1e-2,
            )
            assert_allclose_dense_sparse(
                X_pred,
                X_pred3,
                atol=1e-2,
                err_msg="consecutive fit_transform outcomes not consistent in %s"
                % transformer,
            )
            assert _num_samples(X_pred2) == n_samples
            assert _num_samples(X_pred3) == n_samples

        # raises error on malformed input for transform
        if (
            hasattr(X, "shape")
            and not _safe_tags(transformer, key="stateless")
            and X.ndim == 2
            and X.shape[1] > 1
        ):
            # If it's not an array, it does not have a 'T' property
            with raises(
                ValueError,
                err_msg=(
                    f"The transformer {name} does not raise an error "
                    "when the number of features in transform is different from "
                    "the number of features in fit."
                ),
            ):
                transformer.transform(X[:, :-1])


@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    if _safe_tags(estimator_orig, key="non_deterministic"):
        msg = name + " is non deterministic"
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X = _enforce_estimator_tags_X(estimator_orig, X, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    funcs = ["score", "fit_transform"]

    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


@ignore_warnings
def check_fit_score_takes_y(name, estimator_orig):
    """
    check that all estimators accept an optional y in fit and score so they can be used
    in pipelines

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                # if_delegate_has_method makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert args[1] in ["y", "Y"], (
                "Expected y or Y as second argument for method "
                "%s of %s. Got arguments: %r."
                % (func_name, type(estimator).__name__, args)
            )


@ignore_warnings
def check_estimators_dtypes(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X_train_32 = X.astype(np.float32)
    X_train_32 = _enforce_estimator_tags_X(estimator_orig, X_train_32)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)
    y = _enforce_estimator_tags_y(estimator_orig, y)

    methods = ["predict", "transform", "decision_function", "predict_proba"]

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        estimator.fit(X_train, y)

        for method in methods:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)


def check_transformer_preserve_dtypes(name, transformer_orig):
    """
    check that dtype are preserved meaning if input X is of some dtype
    X_transformed should be from the same dtype.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(transformer_orig, X)

    for dtype in _safe_tags(transformer_orig, key="preserves_dtype"):
        X_cast = X.astype(dtype)
        transformer = clone(transformer_orig)
        set_random_state(transformer)
        X_trans1 = transformer.fit_transform(X_cast, y)
        X_trans2 = transformer.fit(X_cast, y).transform(X_cast)

        for Xt, method in zip([X_trans1, X_trans2], ["fit_transform", "transform"]):
            if isinstance(Xt, tuple):
                # cross-decompostion returns a tuple of (x_scores, y_scores)
                # when given y with fit_transform; only check the first element
                Xt = Xt[0]

            # check that the output dtype is preserved
            assert Xt.dtype == dtype, (
                f"{name} (method={method}) does not preserve dtype. "
                f"Original/Expected dtype={dtype.__name__}, got dtype={Xt.dtype}."
            )


@ignore_warnings(category=FutureWarning)
def check_estimators_empty_data_messages(name, estimator_orig):
    """Check the error message for estimators trained on empty data.

    Modified version of the scikit-learn 1.2.1 function with the name for time series
    data.
    """
    e = clone(estimator_orig)

    X_zero_samples = np.empty(0).reshape((0, 1, 8))
    msg = ["0 sample\(s\)", "n_samples=0", "n_samples = 0"]
    with raises(ValueError, match=msg):
        e.fit(X_zero_samples, [])

    X_zero_features = np.empty(0).reshape((12, 1, 0))
    # the following y should be accepted by both classifiers and regressors
    # and ignored by unsupervised models
    y = _enforce_estimator_tags_y(e, np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
    msg = ["series length 0", "series length=0", "series length = 0"]
    with raises(ValueError, match=msg):
        e.fit(X_zero_features, y)


@ignore_warnings(category=FutureWarning)
def check_estimators_nan_inf(name, estimator_orig):
    """
    Checks that Estimator X's do not contain NaN or inf.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    rnd = np.random.RandomState(0)
    X_train_finite = _enforce_estimator_tags_X(estimator_orig, X)
    X_train_nan = rnd.uniform(size=(10, 1, 3))
    X_train_nan[0, 0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, 1, 3))
    X_train_inf[0, 0, 0] = np.inf
    y = _enforce_estimator_tags_y(estimator_orig, y)
    error_string_fit = f"Estimator {name} doesn't check for NaN and inf in fit."
    error_string_predict = f"Estimator {name} doesn't check for NaN and inf in predict."
    error_string_transform = (
        f"Estimator {name} doesn't check for NaN and inf in transform."
    )
    for X_train in [X_train_nan, X_train_inf]:
        # catch deprecation warnings
        with ignore_warnings(category=FutureWarning):
            estimator = clone(estimator_orig)
            set_random_state(estimator, 1)
            # try to fit
            with raises(ValueError, match=["inf", "NaN"], err_msg=error_string_fit):
                estimator.fit(X_train, y)
            # actually fit
            estimator.fit(X_train_finite, y)

            # predict
            if hasattr(estimator, "predict"):
                with raises(
                    ValueError,
                    match=["inf", "NaN"],
                    err_msg=error_string_predict,
                ):
                    estimator.predict(X_train)

            # transform
            if hasattr(estimator, "transform"):
                with raises(
                    ValueError,
                    match=["inf", "NaN"],
                    err_msg=error_string_transform,
                ):
                    estimator.transform(X_train)


@ignore_warnings
def check_nonsquare_error(name, estimator_orig):
    """
    Test that error is thrown when non-square data provided.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    estimator = clone(estimator_orig)

    with raises(
        ValueError,
        err_msg=(
            f"The pairwise estimator {name} does not raise an error on non-square data"
        ),
    ):
        estimator.fit(X, y)


@ignore_warnings
def check_estimators_pickle(name, estimator_orig):
    """
    Test that we can pickle all estimators

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    X = _enforce_estimator_tags_X(estimator_orig, X, kernel=rbf_kernel)

    tags = _safe_tags(estimator_orig)
    # include NaN values when the estimator should deal with them
    if tags["allow_nan"]:
        # set randomly 10 elements to np.nan
        rng = np.random.RandomState(42)
        mask = rng.choice(X.size, 10, replace=False)
        X.reshape(-1)[mask] = np.nan

    estimator = clone(estimator_orig)

    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)
    estimator.fit(X, y)

    # pickle and unpickle!
    pickled_estimator = pickle.dumps(estimator)
    module_name = estimator.__module__
    if module_name.startswith("sklearn.") and not (
        "test_" in module_name or module_name.endswith("_testing")
    ):
        # strict check for sklearn estimators that are not implemented in test
        # modules.
        assert b"version" in pickled_estimator
    unpickled_estimator = pickle.loads(pickled_estimator)

    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)


@ignore_warnings(category=FutureWarning)
def check_estimators_partial_fit_n_features(name, estimator_orig):
    """
    check if number of features changes between calls to partial_fit.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    if not hasattr(estimator_orig, "partial_fit"):
        return
    estimator = clone(estimator_orig)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator_orig, y)

    try:
        if is_classifier(estimator):
            classes = np.unique(y)
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    except NotImplementedError:
        return

    with raises(
        ValueError,
        err_msg=(
            f"The estimator {name} does not raise an error when the "
            "number of features changes between calls to partial_fit."
        ),
    ):
        estimator.partial_fit(X[:, :-1], y)


@ignore_warnings(category=FutureWarning)
def check_classifier_multioutput(name, estimator):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    n_samples, n_labels, n_classes = 42, 5, 3
    tags = _safe_tags(estimator)
    estimator = clone(estimator)
    X, y = make_multilabel_classification(
        random_state=42, n_samples=n_samples, n_labels=n_labels, n_classes=n_classes
    )
    X = X.reshape(n_samples, 1, -1)
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    assert y_pred.shape == (n_samples, n_classes), (
        "The shape of the prediction for multioutput data is "
        "incorrect. Expected {}, got {}.".format((n_samples, n_labels), y_pred.shape)
    )
    assert y_pred.dtype.kind == "i"

    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(X)
        assert isinstance(decision, np.ndarray)
        assert decision.shape == (n_samples, n_classes), (
            "The shape of the decision function output for "
            "multioutput data is incorrect. Expected {}, got {}.".format(
                (n_samples, n_classes), decision.shape
            )
        )

        dec_pred = (decision > 0).astype(int)
        dec_exp = estimator.classes_[dec_pred]
        assert_array_equal(dec_exp, y_pred)

    if hasattr(estimator, "predict_proba"):
        y_prob = estimator.predict_proba(X)

        if isinstance(y_prob, list) and not tags["poor_score"]:
            for i in range(n_classes):
                assert y_prob[i].shape == (n_samples, 2), (
                    "The shape of the probability for multioutput data is"
                    " incorrect. Expected {}, got {}.".format(
                        (n_samples, 2), y_prob[i].shape
                    )
                )
                assert_array_equal(
                    np.argmax(y_prob[i], axis=1).astype(int), y_pred[:, i]
                )
        elif not tags["poor_score"]:
            assert y_prob.shape == (n_samples, n_classes), (
                "The shape of the probability for multioutput data is"
                " incorrect. Expected {}, got {}.".format(
                    (n_samples, n_classes), y_prob.shape
                )
            )
            assert_array_equal(y_prob.round().astype(int), y_pred)

    if hasattr(estimator, "decision_function") and hasattr(estimator, "predict_proba"):
        for i in range(n_classes):
            y_proba = estimator.predict_proba(X)[:, i]
            y_decision = estimator.decision_function(X)
            assert_array_equal(rankdata(y_proba), rankdata(y_decision[:, i]))


@ignore_warnings(category=FutureWarning)
def check_regressor_multioutput(name, estimator):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator)
    n_samples = n_features = 10

    if not _is_pairwise_metric(estimator):
        n_samples = n_samples + 1

    X, y = make_regression(
        random_state=42, n_targets=5, n_samples=n_samples, n_features=n_features
    )
    X = X.reshape(n_samples, 1, -1)
    X = _enforce_estimator_tags_X(estimator, X)

    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    assert y_pred.dtype == np.dtype("float64"), (
        "Multioutput predictions by a regressor are expected to be"
        " floating-point precision. Got {} instead".format(y_pred.dtype)
    )
    assert y_pred.shape == y.shape, (
        "The shape of the prediction for multioutput data is incorrect."
        " Expected {}, got {}."
    )


@ignore_warnings(category=FutureWarning)
def check_clustering(name, clusterer_orig, readonly_memmap=False):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    clusterer = clone(clusterer_orig)
    rng = np.random.RandomState(7)
    X_noise = np.concatenate(
        [X, rng.uniform(low=-3, high=3, size=(5, X.shape[1], X.shape[2]))]
    )

    if readonly_memmap:
        X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

    n_samples, n_dims, series_length = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert pred.shape == (n_samples,)
    if _safe_tags(clusterer, key="non_deterministic"):
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert pred.dtype in [np.dtype("int32"), np.dtype("int64")]
    assert pred2.dtype in [np.dtype("int32"), np.dtype("int64")]

    # Add noise to X to test the possible values of the labels
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(
        labels_sorted, np.arange(labels_sorted[0], labels_sorted[-1] + 1)
    )

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert labels_sorted[0] in [0, -1]
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, "n_clusters"):
        n_clusters = getattr(clusterer, "n_clusters")
        assert n_clusters - 1 >= labels_sorted[-1]
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=FutureWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """
    Check that predict is invariant of compute_labels.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    clusterer = clone(clusterer_orig)
    set_random_state(clusterer)

    if hasattr(clusterer, "compute_labels"):
        # MiniBatchKMeans
        X_pred1 = clusterer.fit(X).predict(X)
        clusterer.set_params(compute_labels=False)
        X_pred2 = clusterer.fit(X).predict(X)
        assert_array_equal(X_pred1, X_pred2)


@ignore_warnings(category=FutureWarning)
def check_classifiers_one_label(name, classifier_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    error_string_fit = "Classifier can't train when only one class is present."
    error_string_predict = "Classifier can't predict when only one class is present."
    rnd = np.random.RandomState(0)
    X_train = rnd.uniform(size=(10, 1, 8))
    X_test = rnd.uniform(size=(10, 1, 8))
    y = np.ones(10)
    # catch deprecation warnings
    with ignore_warnings(category=FutureWarning):
        classifier = clone(classifier_orig)
        with raises(
            ValueError, match="class", may_pass=True, err_msg=error_string_fit
        ) as cm:
            classifier.fit(X_train, y)

        if cm.raised_and_matched:
            # ValueError was raised with proper error message
            return

        assert_array_equal(classifier.predict(X_test), y, err_msg=error_string_predict)


@ignore_warnings(category=FutureWarning)
def check_classifiers_one_label_sample_weights(name, classifier_orig):
    """
    Check that classifiers accepting sample_weight fit or throws a ValueError with
    an explicit message if the problem is reduced to one class.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    error_fit = (
        f"{name} failed when fitted on one label after sample_weight trimming. Error "
        "message is not explicit, it should have 'class'."
    )
    error_predict = f"{name} prediction results should only output the remaining class."
    rnd = np.random.RandomState(0)
    # X should be square for test on SVC with precomputed kernel
    X_train = rnd.uniform(size=(10, 1, 8))
    X_test = rnd.uniform(size=(10, 1, 8))
    y = np.arange(10) % 2
    sample_weight = y.copy()  # select a single class
    classifier = clone(classifier_orig)

    if has_fit_parameter(classifier, "sample_weight"):
        match = [r"\bclass(es)?\b", error_predict]
        err_type, err_msg = (AssertionError, ValueError), error_fit
    else:
        match = r"\bsample_weight\b"
        err_type, err_msg = (TypeError, ValueError), None

    with raises(err_type, match=match, may_pass=True, err_msg=err_msg) as cm:
        classifier.fit(X_train, y, sample_weight=sample_weight)
        if cm.raised_and_matched:
            # raise the proper error type with the proper error message
            return
        # for estimators that do not fail, they should be able to predict the only
        # class remaining during fit
        assert_array_equal(
            classifier.predict(X_test), np.ones(10), err_msg=error_predict
        )


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(
    name, classifier_orig, readonly_memmap=False, X_dtype="float64"
):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X_m, y_m = test_utils.generate_test_data(n_samples=15, n_labels=3)

    X_m = X_m.astype(X_dtype)
    # generate binary problem from multi-class one
    y_b = y_m.copy()
    y_b[y_b == 2] = 0
    X_b = X_m.copy()
    X_b[y_b == 2, 0, 0] = 0

    if readonly_memmap:
        X_m, y_m, X_b, y_b = create_memmap_backed_data([X_m, y_m, X_b, y_b])

    problems = [(X_b, y_b)]
    tags = _safe_tags(classifier_orig)
    if not tags["binary_only"]:
        problems.append((X_m, y_m))

    for X, y in problems:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, _, n_features = X.shape
        classifier = clone(classifier_orig)
        X = _enforce_estimator_tags_X(classifier, X)
        y = _enforce_estimator_tags_y(classifier, y)

        # raises error on malformed input for fit
        if not tags["no_validation"]:
            with raises(
                ValueError,
                err_msg=(
                    f"The classifier {name} does not raise an error when "
                    "incorrect/malformed input data for fit is passed. The number "
                    "of training examples is not the same as the number of "
                    "labels. Perhaps use check_X_y in fit."
                ),
            ):
                classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert y_pred.shape == (n_samples,)

        # raises error on malformed input for predict
        msg_pairwise = (
            "The classifier {} does not raise an error when shape of X in "
            " {} is not equal to (n_test_samples, n_training_samples)"
        )
        msg = (
            "The classifier {} does not raise an error when the number of "
            "features in {} is different from the number of features in "
            "fit."
        )

        if not tags["no_validation"]:
            if tags["pairwise"]:
                with raises(
                    ValueError,
                    err_msg=msg_pairwise.format(name, "predict"),
                ):
                    classifier.predict(X.reshape(-1, 1))
            else:
                with raises(ValueError, err_msg=msg.format(name, "predict")):
                    classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    if not tags["multioutput_only"]:
                        assert decision.shape == (n_samples,)
                    else:
                        assert decision.shape == (n_samples, 1)
                    dec_pred = (decision.ravel() > 0).astype(int)
                    assert_array_equal(dec_pred, y_pred)
                else:
                    assert decision.shape == (n_samples, n_classes)
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if not tags["no_validation"]:
                    if tags["pairwise"]:
                        with raises(
                            ValueError,
                            err_msg=msg_pairwise.format(name, "decision_function"),
                        ):
                            classifier.decision_function(X.reshape(-1, 1))
                    else:
                        with raises(
                            ValueError,
                            err_msg=msg.format(name, "decision_function"),
                        ):
                            classifier.decision_function(X.T)
            except NotImplementedError:
                pass

        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert y_prob.shape == (n_samples, n_classes)
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1), np.ones(n_samples))
            if not tags["no_validation"]:
                # raises error on malformed input for predict_proba
                if tags["pairwise"]:
                    with raises(
                        ValueError,
                        err_msg=msg_pairwise.format(name, "predict_proba"),
                    ):
                        classifier.predict_proba(X.reshape(-1, 1))
                else:
                    with raises(
                        ValueError,
                        err_msg=msg.format(name, "predict_proba"),
                    ):
                        classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_representation_invariance(name, classifier_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    n_samples, test_size, n_outputs = 20, 10, 4
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=8,
        n_classes=n_outputs,
        n_labels=3,
        length=15,
        allow_unlabeled=True,
    )
    X = X.reshape(X.shape[0], 1, -1)
    X = scale(X)

    X_train, X_test = X[:-test_size], X[-test_size:]
    (y_train,) = y[:-test_size]

    y_train_list_of_lists = y_train.tolist()
    y_train_list_of_arrays = list(y_train)

    classifier = clone(classifier_orig)
    set_random_state(classifier)

    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    y_pred_list_of_lists = classifier.fit(X_train, y_train_list_of_lists).predict(
        X_test
    )

    y_pred_list_of_arrays = classifier.fit(X_train, y_train_list_of_arrays).predict(
        X_test
    )

    assert_array_equal(y_pred, y_pred_list_of_arrays)
    assert_array_equal(y_pred, y_pred_list_of_lists)

    assert y_pred.dtype == y_pred_list_of_arrays.dtype
    assert y_pred.dtype == y_pred_list_of_lists.dtype
    assert type(y_pred) == type(y_pred_list_of_arrays)
    assert type(y_pred) == type(y_pred_list_of_lists)


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_predict(name, classifier_orig):
    """
    Check the output of the `predict` method for classifiers supporting
    multilabel-indicator targets.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    n_samples, test_size, n_outputs = 30, 10, 4
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=15,
        allow_unlabeled=True,
    )
    X = X.reshape(X.shape[0], 1, -1)
    X = scale(X)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    classifier.fit(X_train, y_train)

    response_method_name = "predict"
    predict_method = getattr(classifier, response_method_name, None)
    if predict_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    y_pred = predict_method(X_test)

    # y_pred.shape -> y_test.shape with the same dtype
    assert isinstance(y_pred, np.ndarray), (
        f"{name}.predict is expected to output a NumPy array. Got "
        f"{type(y_pred)} instead."
    )
    assert y_pred.shape == y_test.shape, (
        f"{name}.predict outputs a NumPy array of shape {y_pred.shape} "
        f"instead of {y_test.shape}."
    )
    assert y_pred.dtype == y_test.dtype, (
        f"{name}.predict does not output the same dtype than the targets. "
        f"Got {y_pred.dtype} instead of {y_test.dtype}."
    )


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_predict_proba(name, classifier_orig):
    """
    Check the output of the `predict_proba` method for classifiers supporting
    multilabel-indicator targets.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    n_samples, test_size, n_outputs = 30, 10, 4
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=15,
        allow_unlabeled=True,
    )
    X = X.reshape(X.shape[0], 1, -1)
    X = scale(X)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = y[:-test_size]
    classifier.fit(X_train, y_train)

    response_method_name = "predict_proba"
    predict_proba_method = getattr(classifier, response_method_name, None)
    if predict_proba_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    y_pred = predict_proba_method(X_test)

    # y_pred.shape -> 2 possibilities:
    # - list of length n_outputs of shape (n_samples, 2);
    # - ndarray of shape (n_samples, n_outputs).
    # dtype should be floating
    if isinstance(y_pred, list):
        assert len(y_pred) == n_outputs, (
            f"When {name}.predict_proba returns a list, the list should "
            "be of length n_outputs and contain NumPy arrays. Got length "
            f"of {len(y_pred)} instead of {n_outputs}."
        )
        for pred in y_pred:
            assert pred.shape == (test_size, 2), (
                f"When {name}.predict_proba returns a list, this list "
                "should contain NumPy arrays of shape (n_samples, 2). Got "
                f"NumPy arrays of shape {pred.shape} instead of "
                f"{(test_size, 2)}."
            )
            assert pred.dtype.kind == "f", (
                f"When {name}.predict_proba returns a list, it should "
                "contain NumPy arrays with floating dtype. Got "
                f"{pred.dtype} instead."
            )
            # check that we have the correct probabilities
            err_msg = (
                f"When {name}.predict_proba returns a list, each NumPy "
                "array should contain probabilities for each class and "
                "thus each row should sum to 1 (or close to 1 due to "
                "numerical errors)."
            )
            assert_allclose(pred.sum(axis=1), 1, err_msg=err_msg)
    elif isinstance(y_pred, np.ndarray):
        assert y_pred.shape == (test_size, n_outputs), (
            f"When {name}.predict_proba returns a NumPy array, the "
            f"expected shape is (n_samples, n_outputs). Got {y_pred.shape}"
            f" instead of {(test_size, n_outputs)}."
        )
        assert y_pred.dtype.kind == "f", (
            f"When {name}.predict_proba returns a NumPy array, the "
            f"expected data type is floating. Got {y_pred.dtype} instead."
        )
        err_msg = (
            f"When {name}.predict_proba returns a NumPy array, this array "
            "is expected to provide probabilities of the positive class "
            "and should therefore contain values between 0 and 1."
        )
        assert_array_less(0, y_pred, err_msg=err_msg)
        assert_array_less(y_pred, 1, err_msg=err_msg)
    else:
        raise ValueError(
            f"Unknown returned type {type(y_pred)} by {name}."
            "predict_proba. A list or a Numpy array is expected."
        )


@ignore_warnings(category=FutureWarning)
def check_classifiers_multilabel_output_format_decision_function(name, classifier_orig):
    """
    Check the output of the `decision_function` method for classifiers supporting
    multilabel-indicator targets.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    classifier = clone(classifier_orig)
    set_random_state(classifier)

    n_samples, test_size, n_outputs = 30, 10, 4
    X, y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=2,
        n_classes=n_outputs,
        n_labels=3,
        length=15,
        allow_unlabeled=True,
    )
    X = X.reshape(X.shape[0], 1, -1)
    X = scale(X)

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train = y[:-test_size]
    classifier.fit(X_train, y_train)

    response_method_name = "decision_function"
    decision_function_method = getattr(classifier, response_method_name, None)
    if decision_function_method is None:
        raise SkipTest(f"{name} does not have a {response_method_name} method.")

    y_pred = decision_function_method(X_test)

    # y_pred.shape -> y_test.shape with floating dtype
    assert isinstance(y_pred, np.ndarray), (
        f"{name}.decision_function is expected to output a NumPy array."
        f" Got {type(y_pred)} instead."
    )
    assert y_pred.shape == (test_size, n_outputs), (
        f"{name}.decision_function is expected to provide a NumPy array "
        f"of shape (n_samples, n_outputs). Got {y_pred.shape} instead of "
        f"{(test_size, n_outputs)}."
    )
    assert y_pred.dtype.kind == "f", (
        f"{name}.decision_function is expected to output a floating dtype."
        f" Got {y_pred.dtype} instead."
    )


@ignore_warnings(category=FutureWarning)
def check_get_feature_names_out_error(name, estimator_orig):
    """
    Check the error raised by get_feature_names_out when called before fit.

    Unfitted estimators with get_feature_names_out should raise a NotFittedError.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    err_msg = (
        f"Estimator {name} should have raised a NotFitted error when fit is called"
        " before get_feature_names_out"
    )
    with raises(NotFittedError, err_msg=err_msg):
        estimator.get_feature_names_out()


@ignore_warnings(category=FutureWarning)
def check_estimators_fit_returns_self(name, estimator_orig, readonly_memmap=False):
    """
    Check if self is returned when calling fit.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()
    X = _enforce_estimator_tags_X(estimator_orig, X)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator


@ignore_warnings
def check_estimators_unfitted(name, estimator_orig):
    """
    Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise a NotFittedError.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    estimator = clone(estimator_orig)
    for method in (
        "decision_function",
        "predict",
        "predict_proba",
        "predict_log_proba",
    ):
        if hasattr(estimator, method):
            with raises(NotFittedError):
                getattr(estimator, method)(X)


@ignore_warnings(category=FutureWarning)
def check_supervised_y_2d(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    tags = _safe_tags(estimator_orig)
    n_samples = 30
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    set_random_state(estimator)
    # Check that when a 2D y is given, a DataConversionWarning is
    # raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = "expected 1 DataConversionWarning, got: %s" % ", ".join(
        [str(w_x) for w_x in w]
    )
    if not tags["multioutput"]:
        # check that we warned if we don't support multi-output
        assert len(w) > 0, msg
        assert (
            "DataConversionWarning('A column-vector y"
            " was passed when a 1d array was expected" in msg
        )
    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())


def check_classifiers_classes(name, classifier_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X_multiclass, y_multiclass = test_utils.generate_test_data(n_labels=3)
    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass, random_state=7)

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    X_multiclass = _enforce_estimator_tags_X(classifier_orig, X_multiclass)
    X_binary = _enforce_estimator_tags_X(classifier_orig, X_binary)

    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    problems = [(X_binary, y_binary, y_names_binary)]
    if not _safe_tags(classifier_orig, key="binary_only"):
        problems.append((X_multiclass, y_multiclass, y_names_multiclass))

    for X, y, y_names in problems:
        for y_names_i in [y_names, y_names.astype("O")]:
            y_ = _choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = _choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)


@ignore_warnings
def check_classifiers_predictions(X, y, name, classifier_orig):
    classes = np.unique(y)
    classifier = clone(classifier_orig)
    if name == "BernoulliNB":
        X = X > X.mean()
    set_random_state(classifier)

    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    if hasattr(classifier, "decision_function"):
        decision = classifier.decision_function(X)
        assert isinstance(decision, np.ndarray)
        if len(classes) == 2:
            dec_pred = (decision.ravel() > 0).astype(int)
            dec_exp = classifier.classes_[dec_pred]
            assert_array_equal(
                dec_exp,
                y_pred,
                err_msg=(
                    "decision_function does not match "
                    "classifier for %r: expected '%s', got '%s'"
                )
                % (
                    classifier,
                    ", ".join(map(str, dec_exp)),
                    ", ".join(map(str, y_pred)),
                ),
            )
        elif getattr(classifier, "decision_function_shape", "ovr") == "ovr":
            decision_y = np.argmax(decision, axis=1).astype(int)
            y_exp = classifier.classes_[decision_y]
            assert_array_equal(
                y_exp,
                y_pred,
                err_msg=(
                    "decision_function does not match "
                    "classifier for %r: expected '%s', got '%s'"
                )
                % (
                    classifier,
                    ", ".join(map(str, y_exp)),
                    ", ".join(map(str, y_pred)),
                ),
            )

    assert_array_equal(
        classes,
        classifier.classes_,
        err_msg="Unexpected classes_ attribute for %r: expected '%s', got '%s'"
        % (
            classifier,
            ", ".join(map(str, classes)),
            ", ".join(map(str, classifier.classes_)),
        ),
    )


@ignore_warnings(category=FutureWarning)
def check_regressors_int(name, regressor_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(regressor_orig, X)
    y = _enforce_estimator_tags_y(regressor_orig, y)
    # separate estimators to control random seeds
    regressor_1 = clone(regressor_orig)
    regressor_2 = clone(regressor_orig)
    set_random_state(regressor_1)
    set_random_state(regressor_2)

    # fit
    regressor_1.fit(X, y)
    pred1 = regressor_1.predict(X)
    regressor_2.fit(X, y.astype(float))
    pred2 = regressor_2.predict(X)
    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = X.astype(X_dtype)
    y = scale(y)  # X is already scaled
    regressor = clone(regressor_orig)
    X = _enforce_estimator_tags_X(regressor, X)
    y = _enforce_estimator_tags_y(regressor, y)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    # raises error on malformed input for fit
    with raises(
        ValueError,
        err_msg=(
            f"The classifier {name} does not raise an error when "
            "incorrect/malformed input data for fit is passed. The number of "
            "training examples is not the same as the number of labels. Perhaps "
            "use check_X_y in fit."
        ),
    ):
        regressor.fit(X, y[:-1])

    # fit
    set_random_state(regressor)
    regressor.fit(X, y)
    regressor.fit(X.tolist(), y.tolist())
    y_pred = regressor.predict(X)
    assert y_pred.shape == y.shape


@ignore_warnings
def check_regressors_no_decision_function(name, regressor_orig):
    """
    check that regressors don't have a decision_function, predict_proba, or
    predict_log_proba method.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    regressor = clone(regressor_orig)

    X = _enforce_estimator_tags_X(regressor_orig, X)
    y = _enforce_estimator_tags_y(regressor, y)

    regressor.fit(X, y)
    funcs = ["decision_function", "predict_proba", "predict_log_proba"]
    for func_name in funcs:
        assert not hasattr(regressor, func_name)


@ignore_warnings(category=FutureWarning)
def check_class_weight_classifiers(name, classifier_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    if _safe_tags(classifier_orig, key="binary_only"):
        problems = [2]
    else:
        problems = [2, 3]

    for n_classes in problems:
        # create a very noisy dataset
        X, y = test_utils.generate_test_data(n_samples=15, n_labels=n_classes)
        rng = np.random.RandomState(0)
        X += 20 * rng.uniform(size=X.shape)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0
        )

        # can't use gram_if_pairwise() here, setting up gram matrix manually
        if _safe_tags(classifier_orig, key="pairwise"):
            X_test = rbf_kernel(X_test, X_train)
            X_train = rbf_kernel(X_train, X_train)

        n_classes = len(np.unique(y_train))

        if n_classes == 2:
            class_weight = {0: 1000, 1: 0.0001}
        else:
            y[-1] = 2
            class_weight = {0: 1000, 1: 0.0001, 2: 0.0001}

        classifier = clone(classifier_orig).set_params(class_weight=class_weight)

        set_random_state(classifier)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        if not _safe_tags(classifier_orig, key="poor_score"):
            assert np.mean(y_pred == 0) > 0.75


@ignore_warnings(category=FutureWarning)
def check_estimators_overwrite_params(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    # Fit the model
    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]

        # We should never change or mutate the internal state of input
        # parameters by default. To check this we use the joblib.hash function
        # that introspects recursively any subobjects to compute a checksum.
        # The only exception to this rule of immutable constructor parameters
        # is possible RandomState instance but in this check we explicitly
        # fixed the random_state params recursively to be integer seeds.
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate "
            " the parameter %s from %s to %s during fit."
            % (name, param_name, original_value, new_value)
        )


@ignore_warnings(category=FutureWarning)
def check_no_attributes_set_in_init(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    try:
        # Clone fails if the estimator does not store
        # all parameters as an attribute during init
        estimator = clone(estimator_orig)
    except AttributeError:
        raise AttributeError(
            f"Estimator {name} should store all parameters as an attribute during init."
        )

    if hasattr(type(estimator).__init__, "deprecated_original"):
        return

    init_params = _get_args(type(estimator).__init__)
    if IS_PYPY:
        # __init__ signature has additional objects in PyPy
        for key in ["obj"]:
            if key in init_params:
                init_params.remove(key)
    parents_init_params = [
        param
        for params_parent in (_get_args(parent) for parent in type(estimator).__mro__)
        for param in params_parent
    ]

    # Test for no setting apart from parameters during init
    invalid_attr = set(vars(estimator)) - set(init_params) - set(parents_init_params)
    assert not invalid_attr, (
        "Estimator %s should not set any attribute apart"
        " from parameters during init. Found attributes %s."
        % (name, sorted(invalid_attr))
    )


@ignore_warnings(category=FutureWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray"]:
        check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type)


@ignore_warnings(category=FutureWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    """

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray"]:
        check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type)


@ignore_warnings(category=FutureWarning)
def check_non_transformer_estimators_n_iter(name, estimator_orig):
    """
    Test that estimators that are not transformers with a parameter
    max_iter, return the attribute of n_iter_ at least 1.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        X, y = test_utils.generate_test_data()
        y = _enforce_estimator_tags_y(estimator, y)

        set_random_state(estimator, 0)

        X = _enforce_estimator_tags_X(estimator_orig, X)

        estimator.fit(X, y)

        assert np.all(estimator.n_iter_ >= 1)


@ignore_warnings(category=FutureWarning)
def check_transformer_n_iter(name, estimator_orig):
    """
    Test that transformers with a parameter max_iter, return the
    attribute of n_iter_ at least 1.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    if hasattr(estimator, "max_iter"):
        X, y = test_utils.generate_test_data()
        X = _enforce_estimator_tags_X(estimator_orig, X)
        set_random_state(estimator, 0)
        estimator.fit(X, y)

        assert estimator.n_iter_ >= 1


@ignore_warnings(category=FutureWarning)
def check_classifiers_regression_target(name, estimator_orig):
    """
    Check if classifier throws an exception when fed regression targets

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = _regression_dataset()
    X.reshape((X.shape[0], 1, -1))

    X = _enforce_estimator_tags_X(estimator_orig, X)
    e = clone(estimator_orig)
    msg = "Unknown label type: "
    if not _safe_tags(e, key="no_validation"):
        with raises(ValueError, match=msg):
            e.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_decision_proba_consistency(name, estimator_orig):
    """
    Check whether an estimator having both decision_function and
    predict_proba methods has outputs with perfect rank correlation.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X, y = test_utils.generate_test_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    estimator = clone(estimator_orig)

    if hasattr(estimator, "decision_function") and hasattr(estimator, "predict_proba"):
        estimator.fit(X_train, y_train)
        # Since the link function from decision_function() to predict_proba()
        # is sometimes not precise enough (typically expit), we round to the
        # 10th decimal to avoid numerical issues: we compare the rank
        # with deterministic ties rather than get platform specific rank
        # inversions in case of machine level differences.
        a = estimator.predict_proba(X_test)[:, 1].round(decimals=10)
        b = estimator.decision_function(X_test).round(decimals=10)

        rank_proba, rank_score = rankdata(a), rankdata(b)
        try:
            assert_array_almost_equal(rank_proba, rank_score)
        except AssertionError:
            # Sometimes, the rounding applied on the probabilities will have
            # ties that are not present in the scores because it is
            # numerically more precise. In this case, we relax the test by
            # grouping the decision function scores based on the probability
            # rank and check that the score is monotonically increasing.
            grouped_y_score = np.array(
                [b[rank_proba == group].mean() for group in np.unique(rank_proba)]
            )
            sorted_idx = np.argsort(grouped_y_score)
            assert_array_equal(sorted_idx, np.arange(len(sorted_idx)))


def check_fit_non_negative(name, estimator_orig):
    """
    Check that proper warning is raised for negative X
    when tag requires_positive_X is present

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    X = np.array([[[-1.0, 1], [-1.0, 1]]])
    y = np.array([1, 2])
    estimator = clone(estimator_orig)
    with raises(ValueError):
        estimator.fit(X, y)


def check_fit_idempotent(name, estimator_orig):
    """
    Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    check that the estimated parameters during training (e.g. coefs_) are
    the same, but having a universal comparison function for those
    attributes is difficult and full of edge cases. So instead we check that
    predict(), predict_proba(), decision_function() and transform() return
    the same results.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]
    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params().keys():
        estimator.set_params(warm_start=False)

    X, y = test_utils.generate_test_data(n_samples=15)
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)

    train, test = next(ShuffleSplit(test_size=0.2, random_state=rng).split(X))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # Fit for the first time
    estimator.fit(X_train, y_train)

    result = {
        method: getattr(estimator, method)(X_test)
        for method in check_methods
        if hasattr(estimator, method)
    }

    # Fit again
    set_random_state(estimator)
    estimator.fit(X_train, y_train)

    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            if np.issubdtype(new_result.dtype, np.floating):
                tol = 2 * np.finfo(new_result.dtype).eps
            else:
                tol = 2 * np.finfo(np.float64).eps
            assert_allclose_dense_sparse(
                result[method],
                new_result,
                atol=max(tol, 1e-9),
                rtol=max(tol, 1e-7),
                err_msg="Idempotency check failed for method {}".format(method),
            )


def check_fit_check_is_fitted(name, estimator_orig):
    """
    Make sure that estimator doesn't pass check_is_fitted before calling fit
    and that passes check_is_fitted once it's fit.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    X, y = test_utils.generate_test_data(n_samples=15)
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)

    if not _safe_tags(estimator).get("stateless", False):
        # stateless estimators (such as FunctionTransformer) are always "fit"!
        try:
            check_is_fitted(estimator)
            raise AssertionError(
                f"{estimator.__class__.__name__} passes check_is_fitted before being"
                " fit!"
            )
        except NotFittedError:
            pass
    estimator.fit(X, y)
    try:
        check_is_fitted(estimator)
    except NotFittedError as e:
        raise NotFittedError(
            "Estimator fails to pass `check_is_fitted` even though it has been fit."
        ) from e


def check_n_features_in(name, estimator_orig):
    """
    Make sure that n_features_in_ attribute doesn't exist until fit is
    called, and that its value is correct.

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    X, y = test_utils.generate_test_data()
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)

    assert not hasattr(estimator, "n_features_in_")
    estimator.fit(X, y)
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == (X.shape[1], X.shape[2], X.shape[2])


def check_requires_y_none(name, estimator_orig):
    """
    Make sure that an estimator with requires_y=True fails gracefully when
    given y=None

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    X, _ = test_utils.generate_test_data(n_samples=15)
    X = _enforce_estimator_tags_X(estimator, X)

    expected_err_msgs = (
        "requires y to be passed, but the target y is None",
        "Expected array-like (array or non-string sequence), got None",
        "y should be a 1d array",
    )

    try:
        estimator.fit(X, None)
    except ValueError as ve:
        if not any(msg in str(ve) for msg in expected_err_msgs):
            raise ve


def check_estimator_get_tags_default_keys(name, estimator_orig):
    """
    check that if _get_tags is implemented, it contains all keys from
    _DEFAULT_KEYS

    Modified version of the scikit-learn 1.2.1 function with the name for time series.
    """
    estimator = clone(estimator_orig)
    if not hasattr(estimator, "_get_tags"):
        return

    tags_keys = set(estimator._get_tags().keys())
    default_tags_keys = set(_DEFAULT_TAGS.keys())
    assert tags_keys.intersection(default_tags_keys) == default_tags_keys, (
        f"{name}._get_tags() is missing entries for the following default tags"
        f": {default_tags_keys - tags_keys.intersection(default_tags_keys)}"
    )
