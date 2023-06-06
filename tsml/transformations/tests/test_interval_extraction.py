# -*- coding: utf-8 -*-
from tsml.transformations import Catch22Transformer, SupervisedIntervalTransformer
from tsml.utils.numba_functions.stats import row_mean
from tsml.utils.testing import generate_3d_test_data


def test_supervised_transformers():
    X, y = generate_3d_test_data(random_state=0)

    sit = SupervisedIntervalTransformer(
        features=[
            Catch22Transformer(
                features=["DN_HistogramMode_5", "SB_BinaryStats_mean_longstretch1"]
            ),
            row_mean,
        ],
        n_intervals=2,
        random_state=0,
    )
    X_t = sit.fit_transform(X, y)

    assert X_t.shape == (X.shape[0], 8)
