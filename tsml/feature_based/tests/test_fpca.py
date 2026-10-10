"""Tests for the FPCA pipeline estimators."""

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tsml.datasets import load_minimal_chinatown, load_minimal_gas_prices
from tsml.feature_based import FPCAClassifier, FPCARegressor

# Predictions made using scikit-fda 0.10.1, generated with the previous version of
# FPCATransformer which wrapped its FPCA.
GAS_PRICES_DEFAULT = [
    -0.5346837294278272,
    -0.5096129932694516,
    -0.3250477740947468,
    -0.2188173871053728,
    -0.3306473389207443,
    -0.5336157273822133,
    -0.441275520848197,
    -0.3873708149909935,
    -0.4682849268670188,
    -0.6024416746109142,
    6.693740988073212,
    -0.3631734582495089,
    -0.3063221485732117,
    -0.26277560480992973,
    -0.3784086533148212,
    -0.4346159378571754,
    -0.3760178663743615,
    -0.38042330630366666,
    -0.38870421879709294,
    -0.6379753596584453,
]
GAS_PRICES_BSPLINE = [
    -0.5783393587385739,
    -0.5509453590754624,
    -0.35033137324093133,
    -0.23575874719182743,
    -0.3540619201209665,
    -0.5283445028188842,
    -0.5077822468663241,
    -0.4216192975028944,
    -0.5134293227040485,
    -0.6626500429612009,
    5.562982463275164,
    -0.3481859819154067,
    -0.2930150530631145,
    -0.3155865248116583,
    -0.4192214978835258,
    -0.4707678091846772,
    -0.3846410457639866,
    -0.360056938021058,
    -0.40046098120243445,
    -0.6720378854300684,
]


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, GAS_PRICES_DEFAULT),
        ({"bspline": True, "order": 4, "n_basis": 10}, GAS_PRICES_BSPLINE),
    ],
)
def test_fpca_regressor_matches_scikit_fda(params, expected):
    """Test predictions match those made using scikit-fda 0.10.1.

    The parameters are the two configurations used in tsml-eval.
    """
    X_train, y_train = load_minimal_gas_prices("TRAIN")
    X_test, _ = load_minimal_gas_prices("TEST")

    reg = FPCARegressor(**params)
    reg.fit(X_train, y_train)

    assert_allclose(reg.predict(X_test), expected, rtol=1e-8, atol=1e-10)


def test_fpca_classifier_matches_scikit_fda():
    """Test predictions match those made using scikit-fda 0.10.1."""
    X_train, y_train = load_minimal_chinatown("TRAIN")
    X_test, _ = load_minimal_chinatown("TEST")

    clf = FPCAClassifier()
    clf.fit(X_train, y_train)

    expected = [1] * 10 + [2, 2, 1, 2, 2, 1, 2, 2, 2, 2]
    assert_array_equal(clf.predict(X_test), expected)
