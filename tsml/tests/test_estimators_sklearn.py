"""Patched estimator checks originating from scikit-learn."""

__author__ = ["MatthewMiddlehurst"]

import pickle
import warnings
from copy import deepcopy
from inspect import signature

import joblib
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy.stats import rankdata
from sklearn import clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.utils._testing import (
    SkipTest,
    assert_allclose,
    create_memmap_backed_data,
    ignore_warnings,
    raises,
    set_random_state,
)
from sklearn.utils.estimator_checks import _is_public_parameter, _NotAnArray
from sklearn.utils.validation import _num_samples, check_is_fitted

import tsml.utils.testing as test_utils
from tsml.utils._tags import _DEFAULT_TAGS, _safe_tags


@ignore_warnings(category=FutureWarning)
def check_supervised_y_no_nan(name, estimator_orig):
    """Check that the Estimator targets are not NaN.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, _ = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    for value in [np.nan, np.inf]:
        y = np.full(10, value)

        if np.isinf(value):
            match = (
                r"Input (y|Y) contains infinity or a value too large for"
                r" dtype\('float64'\)."
            )
        else:
            match = r"Input (y|Y) contains NaN."

        err_msg = (
            f"Estimator {name} should have raised error on fitting array y with inf"
            " value."
        )
        with raises(ValueError, match=match, err_msg=err_msg):
            estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_dtype_object(name, estimator_orig):
    """Check that estimators treat dtype object as numeric if possible.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()
    X = X.astype(object)

    estimator = clone(estimator_orig)

    estimator.fit(X, y)

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
    ]:
        if hasattr(estimator, method):
            getattr(estimator, method)(X)

    with raises(Exception, match="Unknown label type", may_pass=True):
        estimator.fit(X, y.astype(object))

    X[0, 0] = {"foo": "bar"}
    msg = "argument must be a string.* number"
    with raises(TypeError, match=msg):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_complex_data(name, estimator_orig):
    """Check that estimators raise an exception on complex data.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    # check that estimators raise an exception on providing complex data
    X = X + 1j
    y = y + 1j

    estimator = clone(estimator_orig)

    with raises(ValueError, match="Complex data not supported"):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_dict_unchanged(name, estimator_orig):
    """Check that estimator dict is not modified by predict/transform methods.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    estimator.fit(X, y)

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
    ]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert (
                estimator.__dict__ == dict_before
            ), f"Estimator changes __dict__ during {method}"


@ignore_warnings(category=FutureWarning)
def check_dont_overwrite_parameters(name, estimator_orig):
    """Check that fit method only changes or sets private attributes.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

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
        "Estimator adds public attribute(s) during the fit method. Estimators are only "
        "allowed to add private attributes either started with _ or ended with _ but "
        f"{', '.join(attrs_added_by_fit)} were added."
    )

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
    ]

    assert not attrs_changed_by_fit, (
        "Estimator changes public attribute(s) during the fit method. Estimators are "
        "only allowed to change attributes started or ended with _, but "
        f"{', '.join(attrs_changed_by_fit)} were changed."
    )


@ignore_warnings(category=FutureWarning)
def check_fit3d_predict1d(name, estimator_orig):
    """Check by fitting a 3d array and predicting with a 1d array.

    Modified version of the scikit-learn 1.2.1 function with a similar name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    estimator.fit(X, y)

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
    ]:
        if hasattr(estimator, method):
            with raises(ValueError, match="Reshape your data"):
                getattr(estimator, method)(X[0][0])


@ignore_warnings(category=FutureWarning)
def check_methods_subset_invariance(name, estimator_orig):
    """Check smaller batches of data for predict methods does not impact results.

    Check that method gives invariant results if applied on mini batches or the whole
    set.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)
    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
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
                np.ravel(result_full), np.ravel(result_by_batch), err_msg=msg
            )


@ignore_warnings(category=FutureWarning)
def check_methods_sample_order_invariance(name, estimator_orig):
    """Test sample order invariance.

    Check that method gives invariant results if applied on a subset with different
    sample order.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)
    set_random_state(estimator, 1)
    estimator.fit(X, y)

    idx = np.random.permutation(X.shape[0])

    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
    ]:
        msg = (
            f"{method} of {name} is not invariant when applied to a dataset "
            "with different sample order."
        )

        if hasattr(estimator, method):
            assert_allclose(
                getattr(estimator, method)(X)[idx],
                getattr(estimator, method)(X[idx]),
                err_msg=msg,
            )


@ignore_warnings(category=FutureWarning)
def check_fit3d_1sample(name, estimator_orig):
    """Check for fitting an estimator with only 1 sample.

    Check that fitting a 3d array with only one sample either works or
    returns an informative message. The error message should either mention
    the number of samples or the number of classes.

    Modified version of the scikit-learn 1.2.1 function with a similar name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data(n_samples=1)

    estimator = clone(estimator_orig)

    msg = [
        "1 sample",
        "n_samples = 1",
        "n_samples=1",
        "one sample",
        "1 class",
        "one class",
    ]
    with raises(ValueError, match=msg, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_fit3d_1feature(name, estimator_orig):
    """Check for fitting an estimator with only 1 series length.

    Check fitting a 3d array with only 1 feature either works or returns
    informative message

    Modified version of the scikit-learn 1.2.1 function with a similar name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data(series_length=1)

    estimator = clone(estimator_orig)

    msg = ["1 series length", "series length 1", "series length = 1", "series length=1"]
    with raises(ValueError, match=msg, may_pass=True):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_fit1d(name, estimator_orig):
    """Check fitting 1d X array raises a ValueError.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    rnd = np.random.RandomState()
    X = 3 * rnd.uniform(size=10)
    y = X.astype(int)

    estimator = clone(estimator_orig)

    with raises(ValueError):
        estimator.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_general(name, transformer, readonly_memmap=False):
    """Check transformer adheres to sklearn-like conventions.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    _check_transformer(name, transformer, X, y)


@ignore_warnings(category=FutureWarning)
def check_transformer_data_not_an_array(name, transformer):
    """Check transformer works with non-array input.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    this_X = _NotAnArray(X)
    this_y = _NotAnArray(np.asarray(y))

    _check_transformer(name, transformer, this_X, this_y)
    # try the same with some list
    _check_transformer(name, transformer, X.tolist(), y.tolist())


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

    assert hasattr(transformer, "transform")

    X_pred2 = transformer.transform(X)
    X_pred3 = transformer.fit_transform(X, y=y)

    if _safe_tags(transformer_orig, key="non_deterministic"):
        raise SkipTest(name + " is non deterministic")

    if isinstance(X_pred, tuple) and isinstance(X_pred2, tuple):
        for x_pred, x_pred2, x_pred3 in zip(X_pred, X_pred2, X_pred3):
            assert_allclose(
                x_pred,
                x_pred2,
                atol=1e-2,
                err_msg="fit_transform and transform outcomes not consistent in %s"
                % transformer,
            )
            assert_allclose(
                x_pred,
                x_pred3,
                atol=1e-2,
                err_msg="consecutive fit_transform outcomes not consistent in %s"
                % transformer,
            )
    else:
        assert_allclose(
            X_pred,
            X_pred2,
            err_msg="fit_transform and transform outcomes not consistent in %s"
            % transformer,
            atol=1e-2,
        )
        assert_allclose(
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
        and _safe_tags(transformer, key="requires_fit")
        and X.ndim == 3
        and X.shape[2] > 1
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
            transformer.transform(X[:, :, -1])


@ignore_warnings(category=FutureWarning)
def check_pipeline_consistency(name, estimator_orig):
    """Check estimators and pipelines created from an estimator give same results.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)

    estimator.fit(X, y)
    pipeline.fit(X, y)

    # check that make_pipeline(est) gives same score as est
    funcs = ["score", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose(result, result_pipe)


@ignore_warnings(category=FutureWarning)
def check_fit_score_takes_y(name, estimator_orig):
    """Check that all estimators accept an optional y in fit and score.

    This is so they can be used in pipelines.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    for func_name in ["fit", "score", "fit_predict", "fit_transform"]:
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


@ignore_warnings(category=FutureWarning)
def check_estimators_dtypes(name, estimator_orig):
    """Check that estimators work with different dtypes.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    X_train_32 = X.astype(np.float32)
    X_train_64 = X_train_32.astype(np.float64)
    X_train_int_64 = X_train_32.astype(np.int64)
    X_train_int_32 = X_train_32.astype(np.int32)

    for X_train in [X_train_32, X_train_64, X_train_int_64, X_train_int_32]:
        estimator = clone(estimator_orig)

        estimator.fit(X_train, y)

        for method in [
            "predict",
            "predict_proba",
            "decision_function",
            "transform",
        ]:
            if hasattr(estimator, method):
                getattr(estimator, method)(X_train)


@ignore_warnings(category=FutureWarning)
def check_transformer_preserve_dtypes(name, transformer_orig):
    """Check that dtype are preserved meaning if input X is of some dtype.

    X_transformed should be from the same dtype.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

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

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    estimator = clone(estimator_orig)

    X_zero_samples = np.empty(0).reshape((0, 1, 8))
    msg = ["0 sample\(s\)", "n_samples=0", "n_samples = 0"]  # noqa: W605
    with raises(ValueError, match=msg):
        estimator.fit(X_zero_samples, [])

    X_zero_features = np.empty(0).reshape((12, 1, 0))
    # the following y should be accepted by both classifiers and regressors
    # and ignored by unsupervised models
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    msg = ["0 series length", "series length 0", "series length=0", "series length = 0"]
    with raises(ValueError, match=msg):
        estimator.fit(X_zero_features, y)


@ignore_warnings(category=FutureWarning)
def check_estimators_nan_inf(name, estimator_orig):
    """Check that Estimator X's do not contain NaN or inf.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()
    X_nan = X.copy()
    X_nan[0, 0, 0] = np.nan
    X_inf = X.copy()
    X_inf[0, 0, 0] = np.inf

    error_string_fit = f"Estimator {name} doesn't check for NaN and inf in fit."
    for X_train in [X_nan, X_inf]:
        estimator = clone(estimator_orig)

        # try to fit
        with raises(ValueError, match=["inf", "NaN"], err_msg=error_string_fit):
            estimator.fit(X_train, y)
        # actually fit
        estimator.fit(X, y)

        for method in [
            "predict",
            "predict_proba",
            "decision_function",
            "transform",
        ]:
            if hasattr(estimator, method):
                with raises(
                    ValueError,
                    match=["inf", "NaN"],
                    err_msg=(
                        f"Estimator {name} doesn't check for NaN and inf in {method}."
                    ),
                ):
                    getattr(estimator, method)(X_train)


@ignore_warnings(category=FutureWarning)
def check_estimators_pickle(name, estimator_orig):
    """Test that we can pickle all estimators.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    # include NaN values when the estimator should deal with them
    if _safe_tags(estimator_orig, key="allow_nan"):
        # set randomly 10 elements to np.nan
        rng = np.random.RandomState()
        mask = rng.choice(X.size, 10, replace=False)
        X.reshape(-1)[mask] = np.nan

    estimator = clone(estimator_orig)

    estimator.fit(X, y)

    # pickle and unpickle!
    pickled_estimator = pickle.dumps(estimator)
    unpickled_estimator = pickle.loads(pickled_estimator)

    result = dict()
    for method in [
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
    ]:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose(result[method], unpickled_result)


@ignore_warnings(category=FutureWarning)
def check_clustering(name, clusterer_orig, readonly_memmap=False):
    """Check clusterer adheres to sklearn-like conventions.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

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
        assert clusterer.n_clusters - 1 >= labels_sorted[-1]
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=FutureWarning)
def check_clusterer_compute_labels_predict(name, clusterer_orig):
    """Check that predict is invariant of compute_labels.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    clusterer = clone(clusterer_orig)
    set_random_state(clusterer)

    if hasattr(clusterer, "compute_labels"):
        X_pred1 = clusterer.fit(X).predict(X)
        clusterer.set_params(compute_labels=False)
        X_pred2 = clusterer.fit(X).predict(X)
        assert_array_equal(X_pred1, X_pred2)


@ignore_warnings(category=FutureWarning)
def check_classifiers_one_label(name, classifier_orig):
    """Check classifier outputs suitable error or correct output for single class input.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X_train, _ = test_utils.generate_3d_test_data()
    X_test, _ = test_utils.generate_3d_test_data()
    y = np.ones(10)

    classifier = clone(classifier_orig)

    with raises(
        ValueError,
        match="single class",
        may_pass=True,
        err_msg="Classifier can't train when only one class is present.",
    ) as cm:
        classifier.fit(X_train, y)

    if cm.raised_and_matched:
        # ValueError was raised with proper error message
        return

    assert_array_equal(
        classifier.predict(X_test),
        y,
        "Classifier can't predict when only one class is present.",
    )


@ignore_warnings(category=FutureWarning)
def check_classifiers_train(
    name, classifier_orig, readonly_memmap=False, X_dtype="float64"
):
    """Check classifier adheres to sklearn-like conventions.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X_m, y_m = test_utils.generate_3d_test_data(n_labels=3)

    X_m = X_m.astype(X_dtype)
    # generate binary problem from multi-class one
    y_b = y_m.copy()
    y_b[y_b == 2] = 0
    X_b = X_m.copy()
    X_b[y_b == 2, 0, 0] = 0

    if readonly_memmap:
        X_m, y_m, X_b, y_b = create_memmap_backed_data([X_m, y_m, X_b, y_b])

    problems = [(X_b, y_b), (X_m, y_m)]
    tags = _safe_tags(classifier_orig)

    for X, y in problems:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, _, n_features = X.shape
        classifier = clone(classifier_orig)

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
        msg = (
            "The classifier {} does not raise an error when the number of "
            "features in {} is different from the number of features in "
            "fit."
        )

        if not tags["no_validation"]:
            with raises(ValueError, err_msg=msg.format(name, "predict")):
                classifier.predict(X.T)

        if hasattr(classifier, "decision_function"):
            # decision_function agrees with predict
            decision = classifier.decision_function(X)
            if n_classes == 2:
                assert decision.shape == (n_samples,)
                dec_pred = (decision.ravel() > 0).astype(int)
                assert_array_equal(dec_pred, y_pred)
            else:
                assert decision.shape == (n_samples, n_classes)
                assert_array_equal(np.argmax(decision, axis=1), y_pred)

            # raises error on malformed input for decision_function
            if not tags["no_validation"]:
                with raises(
                    ValueError,
                    err_msg=msg.format(name, "decision_function"),
                ):
                    classifier.decision_function(X.T)

        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert y_prob.shape == (n_samples, n_classes)
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1), np.ones(n_samples))

            if not tags["no_validation"]:
                # raises error on malformed input for predict_proba
                with raises(
                    ValueError,
                    err_msg=msg.format(name, "predict_proba"),
                ):
                    classifier.predict_proba(X.T)


@ignore_warnings(category=FutureWarning)
def check_estimators_fit_returns_self(name, estimator_orig, readonly_memmap=False):
    """Check if self is returned when calling fit.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    assert estimator.fit(X, y) is estimator


@ignore_warnings(category=FutureWarning)
def check_estimators_unfitted(name, estimator_orig):
    """Check that predict raises an exception in an unfitted estimator.

    Unfitted estimators should raise a NotFittedError.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)
    for method in (
        "predict",
        "predict_proba",
        "decision_function",
        "transform",
    ):
        if hasattr(estimator, method):
            with raises(
                NotFittedError,
                err_msg=(
                    f"The unfitted estimator {name} does not raise an error when "
                    f"{method} is called. Perhaps use check_is_fitted."
                ),
            ):
                getattr(estimator, method)(X)


@ignore_warnings(category=FutureWarning)
def check_supervised_y_2d(name, estimator_orig):
    """Check that when a 2D y is given, a DataConversionWarning is raised.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)
    set_random_state(estimator)

    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

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

    assert len(w) > 0, msg
    assert (
        "DataConversionWarning('A column-vector y"
        " was passed when a 1d array was expected" in msg
    )

    if _safe_tags(estimator_orig, key="non_deterministic"):
        raise SkipTest(name + " is non deterministic")

    assert_allclose(y_pred.ravel(), y_pred_2d.ravel())


@ignore_warnings(category=FutureWarning)
def check_classifiers_classes(name, classifier_orig):
    """Check classifier can handle binary and multiclass data with different y types.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X_multiclass, y_multiclass = test_utils.generate_3d_test_data(n_labels=3)

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    problems = [
        (X_binary, y_names_binary),
        (X_multiclass, y_names_multiclass),
    ]

    for X, y_names in problems:
        for y_names_i in [y_names, y_names.astype("O")]:
            _check_classifiers_predictions(X, y_names_i, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    _check_classifiers_predictions(X_binary, y_names_binary, classifier_orig)


@ignore_warnings
def _check_classifiers_predictions(X, y, classifier_orig):
    classes = np.unique(y)
    classifier = clone(classifier_orig)

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
    """Check that regressor labels return the same result as float or int labels.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

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

    if _safe_tags(regressor_orig, key="non_deterministic"):
        raise SkipTest(name + " is non deterministic")

    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    """Check regressor adheres to sklearn-like conventions.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data(regression_target=True)

    X = X.astype(X_dtype)
    y = scale(y)  # X is already scaled

    regressor = clone(regressor_orig)

    if readonly_memmap:
        X, y = create_memmap_backed_data([X, y])

    # raises error on malformed input for fit
    if not _safe_tags(regressor_orig, key="no_validation"):
        with raises(
            ValueError,
            err_msg=(
                f"The regressor {name} does not raise an error when "
                "incorrect/malformed input data for fit is passed. The number of "
                "training examples is not the same as the number of labels. Perhaps "
                "use check_X_y in fit."
            ),
        ):
            regressor.fit(X, y[:-1])

    # fit
    regressor.fit(X, y)
    regressor.fit(X.tolist(), y.tolist())
    y_pred = regressor.predict(X)
    assert y_pred.shape == y.shape


@ignore_warnings(category=FutureWarning)
def check_regressors_no_decision_function(name, regressor_orig):
    """Check that regressors don't have a decision_function or predict_proba.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data(regression_target=True)

    regressor = clone(regressor_orig)

    regressor.fit(X, y)
    for method in [
        "decision_function",
        "predict_proba",
    ]:
        assert not hasattr(regressor, method)


@ignore_warnings(category=FutureWarning)
def check_estimators_overwrite_params(name, estimator_orig):
    """Check estimators do not overwrite parameters during fit.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

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
            f"Estimator {name} should not change or mutate  the parameter {param_name} "
            f"from {original_value} to {new_value} during fit."
        )


@ignore_warnings(category=FutureWarning)
def check_estimator_data_not_an_array(name, estimator_orig):
    """Check non-array data input runs without error and produces similar output.

    Modified version of the scikit-learn 1.2.1 function with similar names for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    for obj_type in ["NotAnArray", "PandasDataframe"]:
        _check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type)


@ignore_warnings(category=FutureWarning)
def _check_estimators_data_not_an_array(name, estimator_orig, X, y, obj_type):
    estimator_1 = clone(estimator_orig)
    estimator_2 = clone(estimator_orig)
    set_random_state(estimator_1)
    set_random_state(estimator_2)

    if obj_type not in ["NotAnArray", "PandasDataframe"]:
        raise ValueError("Data type {0} not supported".format(obj_type))

    if obj_type == "NotAnArray":
        y_ = _NotAnArray(np.asarray(y))
        X_ = _NotAnArray(np.asarray(X))
    else:
        # Here pandas objects (Series and DataFrame) are tested explicitly
        # because some estimators may handle them (especially their indexing)
        # specially.
        try:
            import pandas as pd

            y_ = np.asarray(y)
            if y_.ndim == 1:
                y_ = pd.Series(y_)
            else:
                y_ = pd.DataFrame(y_)
            X_ = pd.DataFrame(np.asarray(X.reshape((X.shape[0], -1))))

        except ImportError:
            raise SkipTest(
                "pandas is not installed: not checking estimators for pandas objects."
            )

    # fit
    estimator_1.fit(X_, y_)
    pred1 = estimator_1.predict(X_)
    estimator_2.fit(X, y)
    pred2 = estimator_2.predict(X)

    if _safe_tags(estimator_orig, key="non_deterministic"):
        raise SkipTest(name + " is non deterministic")

    assert_allclose(pred1, pred2, atol=1e-2, err_msg=name)


@ignore_warnings(category=FutureWarning)
def check_classifiers_regression_target(name, classifier_orig):
    """Check if classifier throws an exception when fed regression targets.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data(regression_target=True)

    classifier = clone(classifier_orig)
    msg = "Unknown label type: "
    with raises(ValueError, match=msg):
        classifier.fit(X, y)


@ignore_warnings(category=FutureWarning)
def check_decision_proba_consistency(name, estimator_orig):
    """Check estimators with both decision_function and predict_proba methods.

    Check whether an estimator having both decision_function and
    predict_proba methods has outputs with perfect rank correlation.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

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


@ignore_warnings(category=FutureWarning)
def check_fit_idempotent(name, estimator_orig):
    """Check that estimator can be fit multiple times with the same results.

    Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    check that the estimated parameters during training (e.g. coefs_) are
    the same, but having a universal comparison function for those
    attributes is difficult and full of edge cases. So instead we check that
    predict(), predict_proba(), decision_function() and transform() return
    the same results.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X_train, y_train = test_utils.generate_3d_test_data()
    X_test, y_test = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)
    set_random_state(estimator, 1)

    # Fit for the first time
    estimator.fit(X_train, y_train)

    check_methods = ["predict", "transform", "decision_function", "predict_proba"]
    result = {
        method: getattr(estimator, method)(X_test)
        for method in check_methods
        if hasattr(estimator, method)
    }

    # Fit again
    estimator.fit(X_train, y_train)

    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)

            assert_allclose(
                result[method],
                new_result,
                err_msg="Idempotency check failed for method {}".format(method),
            )


@ignore_warnings(category=FutureWarning)
def check_fit_check_is_fitted(name, estimator_orig):
    """Check check_is_fitted works as expected.

    Make sure that estimator doesn't pass check_is_fitted before calling fit
    and that passes check_is_fitted once it's fit.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    if _safe_tags(estimator, key="requires_fit"):
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


@ignore_warnings(category=FutureWarning)
def check_n_features_in(name, estimator_orig):
    """Check n_features_in_ attribute.

    Make sure that n_features_in_ attribute doesn't exist until fit is
    called, and that its value is correct.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, y = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    assert not hasattr(estimator, "n_features_in_")
    estimator.fit(X, y)
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == (X.shape[1], X.shape[2], X.shape[2])


@ignore_warnings(category=FutureWarning)
def check_requires_y_none(name, estimator_orig):
    """Check that estimators requiring y fail gracefully.

    Make sure that an estimator with requires_y=True fails gracefully when
    given y=None.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    X, _ = test_utils.generate_3d_test_data()

    estimator = clone(estimator_orig)

    msg = [
        "requires y to be passed, but the target y is None",
        "Expected array-like (array or non-string sequence), got None",
        "y should be a 1d array",
    ]

    with raises(ValueError, match=msg):
        estimator.fit(X, None)


@ignore_warnings(category=FutureWarning)
def check_estimator_get_tags_default_keys(name, estimator_orig):
    """Check that if _get_tags is implemented, it contains all keys from _DEFAULT_KEYS.

    Modified version of the scikit-learn 1.2.1 function with the same name for time
    series data.
    """
    estimator = clone(estimator_orig)
    if not hasattr(estimator, "_get_tags"):
        return

    tags_keys = set(estimator._get_tags().keys())
    default_tags_keys = set(_DEFAULT_TAGS.keys())
    assert tags_keys.intersection(default_tags_keys) == default_tags_keys, (
        f"{name}._get_tags() is missing entries for the following default tags: "
        f"{default_tags_keys - tags_keys.intersection(default_tags_keys)}"
    )


# todo add pandas tests again?
