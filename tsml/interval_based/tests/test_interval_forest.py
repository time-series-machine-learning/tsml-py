# -*- coding: utf-8 -*-
"""Tests for the BaseIntervalForest class."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def test_interval_forest_on_dataframe_input():
    """Test of BaseIntervalForest on unit test data."""
    # load unit test data
    X, y = load_unit_test(split="train")

    est = BaseIntervalForest(
        DecisionTreeClassifier(), "classifier", n_estimators=2, n_intervals=2
    )
    est._fit(X, y)

    assert est._predict_proba(X).shape == (20, 2)


@pytest.mark.parametrize(
    "interval_selection_method",
    ["random", "supervised", "random-supervised"],
)
def test_interval_forest_selection_methods(interval_selection_method):
    """Test BaseIntervalForest with different interval selection methods."""
    X, y = _generate_data()

    est = BaseIntervalForest(
        DecisionTreeClassifier(),
        "classifier",
        n_estimators=2,
        n_intervals=2,
        interval_selection_method=interval_selection_method,
    )
    est._fit(X, y)

    assert est._predict_proba(X).shape == (10, 2)


base_est = [DecisionTreeClassifier(), DecisionTreeRegressor(), ContinuousIntervalTree()]
est_type = ["classifier", "regressor", "classifier"]


@pytest.mark.parametrize(
    "est_idx",
    list(range(3)),
)
def test_interval_forest_feature_skipping(est_idx):
    """Test BaseIntervalForest feature skipping with different base estimators."""
    X, y = _generate_data()

    est = BaseIntervalForest(
        base_est[est_idx],
        est_type[est_idx],
        n_estimators=2,
        n_intervals=2,
        random_state=np.random.randint(np.iinfo(np.int32).max),
    )
    est._fit(X, y)
    preds = est._predict(X)

    assert est._efficient_predictions is True

    est._test_flag = True
    est._fit(X, y)

    assert est._efficient_predictions is False
    assert (preds == est._predict(X)).all()


def test_interval_forest_invalid_feature_skipping():
    """Test BaseIntervalForest with an invalid transformer for feature skipping."""
    X, y = _generate_data()

    est = BaseIntervalForest(
        DecisionTreeClassifier(),
        "classifier",
        n_estimators=2,
        n_intervals=2,
        interval_features=SummaryTransformer(
            summary_function=("mean", "min", "max"),
            quantiles=0.5,
        ),
        random_state=np.random.randint(np.iinfo(np.int32).max),
    )
    est._fit(X, y)

    assert est._efficient_predictions is False


att_subsample_c22 = Catch22(
    features=[
        "DN_HistogramMode_5",
        "DN_HistogramMode_10",
        "SB_BinaryStats_diff_longstretch0",
    ]
)
features = [
    None,
    _clone_estimator(att_subsample_c22),
    [_clone_estimator(att_subsample_c22), _clone_estimator(att_subsample_c22)],
    [
        row_mean,
        Catch22(features=["DN_HistogramMode_5", "DN_HistogramMode_10"]),
        row_numba_min,
    ],
]
feature_lens = [3, 3, 6, 4]


@pytest.mark.parametrize(
    "feature_idx",
    list(range(4)),
)
def test_interval_forest_attribute_subsample(feature_idx):
    """Test BaseIntervalForest subsampling with different interval features."""
    X, y = _generate_data()

    est = BaseIntervalForest(
        DecisionTreeClassifier(),
        "classifier",
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=0.5,
        interval_features=features[feature_idx],
        save_transformed_data=True,
    )
    est._fit(X, y)
    est._predict_proba(X)

    data = est.transformed_data_
    assert data[0].shape[1] == int(feature_lens[feature_idx] * 0.5) * 2


def test_interval_forest_invalid_attribute_subsample():
    """Test BaseIntervalForest with an invalid transformer for subsampling."""
    X, y = _generate_data()

    est = BaseIntervalForest(
        DecisionTreeClassifier(),
        "classifier",
        n_estimators=2,
        n_intervals=2,
        att_subsample_size=2,
        interval_features=SummaryTransformer(
            summary_function=("mean", "min", "max"),
            quantiles=0.5,
        ),
    )

    with pytest.raises(ValueError):
        est._fit(X, y)


@pytest.mark.parametrize(
    "series_transformer",
    [[None, Differencer()], [Differencer(), ExponentTransformer()]],
)
def test_interval_forest_series_transformer(series_transformer):
    """Test BaseIntervalForest with different series transformers."""
    X, y = _generate_data()

    est = BaseIntervalForest(
        DecisionTreeClassifier(),
        "classifier",
        n_estimators=2,
        n_intervals=2,
        series_transformers=series_transformer,
        save_transformed_data=True,
    )
    est._fit(X, y)
    est._predict_proba(X)

    data = est.transformed_data_
    assert data[0].shape[1] == 12


n_intervals = ["sqrt", "sqrt-div", ["sqrt-div", 2], [[1, 2], "sqrt-div"]]
n_interval_lens = [24, 12, 12, 15]


@pytest.mark.parametrize(
    "n_intervals_idx",
    list(range(4)),
)
def test_interval_forest_n_intervals(n_intervals_idx):
    """Test BaseIntervalForest n_interval options."""
    X, y = _generate_data()

    est = BaseIntervalForest(
        DecisionTreeClassifier(),
        "classifier",
        n_estimators=2,
        n_intervals=n_intervals[n_intervals_idx],
        series_transformers=[None, ExponentTransformer()],
        save_transformed_data=True,
    )
    est._fit(X, y)
    est._predict_proba(X)

    data = est.transformed_data_
    assert data[0].shape[1] == n_interval_lens[n_intervals_idx]
