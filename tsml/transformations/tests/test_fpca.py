"""Tests for the FPCA transformer."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tsml.datasets import load_minimal_gas_prices
from tsml.transformations import FPCATransformer
from tsml.utils.testing import generate_3d_test_data

# Scores from scikit-fda 0.10.1, generated with the previous version of FPCATransformer
# which wrapped its FPCA. Each holds the scores of the first train case followed by
# those of the first two test cases.
GAS_PRICES_DEFAULT = [
    [
        [
            -1.857951996504731,
            -0.5010194447404395,
            0.3592643287479542,
            0.12460246976376857,
            0.08842452295639075,
            0.10789342929518345,
            0.033731719840572834,
            -0.10547281054081965,
            -0.1610537123036559,
            0.08500257892305291,
        ]
    ],
    [
        [
            -0.9247123237657991,
            -0.30641493825103755,
            -0.4091825076375773,
            -0.4681037706917189,
            -0.23274492390887228,
            0.11431508420686022,
            -0.22713651580235628,
            0.12327423759743805,
            0.07714501503504212,
            0.28999505166670897,
        ]
    ],
    [
        [
            0.6154540524267198,
            -0.30792327461627034,
            0.03922721176024962,
            -0.32306836480253487,
            -0.04038963838707784,
            0.0006272579273613234,
            -0.05050822417268916,
            0.21291445370071638,
            0.024709437886863583,
            0.15244316641535904,
        ]
    ],
]
GAS_PRICES_BSPLINE = [
    [
        [
            -1.8766142617820967,
            -0.5306254414808107,
            0.35086821579038285,
            -0.17238379941129764,
            -0.03296475685835699,
            -0.1299751567523994,
            -0.013714728086087745,
            0.052315762412203946,
            -0.024729016251858874,
            -0.005418507040517995,
        ]
    ],
    [
        [
            -0.9297414681270051,
            -0.28709171195349575,
            -0.3820850015548448,
            0.5514491587136915,
            0.1580396659010885,
            -0.09917907307495338,
            -0.19033160040172298,
            0.2729410166495314,
            -0.06382630666637173,
            -0.03492945255798162,
        ]
    ],
    [
        [
            0.6103577848856844,
            -0.27231480316247086,
            0.04500387352976078,
            0.36217530980226254,
            -0.03977948445697691,
            -0.023953149380829326,
            0.018262616228731057,
            0.2422584065377346,
            -0.037258176805002416,
            -0.00425939283153677,
        ]
    ],
]
GAS_PRICES_NO_CENTERING = [
    [[11.281428378673844, -0.9687100241933331, 0.252925453260739]],
    [[12.214668051412776, -0.7741055177039317, -0.5155213831247922]],
    [[13.754834427605296, -0.7756138540691639, -0.06711166372696575]],
]
GAS_PRICES_BSPLINE_NO_CENTERING = [
    [[11.255469188601964, -0.8350987357839555, 0.15109062585607957]],
    [[12.203513385363397, -0.7457870603961665, -0.5552627896432832]],
    [[13.745992097841432, -0.6401902430806745, -0.20538843168618928]],
]
MULTIVARIATE = [
    [
        [-0.4395020856091999, -0.4852460776334841, -0.03394248313526303],
        [-0.944338627484553, -0.9166522522426404, 0.4846473962875993],
    ],
    [
        [4.044225132256141, 1.4262479410789697, 0.7857863286016361],
        [0.9614273300039429, -0.3276969866783707, 0.49961102462863854],
    ],
    [
        [-0.5335219979226008, -0.40449401465393575, 2.9931358754040405],
        [1.568725949272281, 1.7859900233086334, -2.641885356490605],
    ],
]
MULTIVARIATE_BSPLINE = [
    [
        [-0.8014839971658254, -0.3300311758913074, -0.19718824468424193],
        [-0.9113345227122691, -0.6736760811899752, 1.3624761997423982],
    ],
    [
        [2.8366072061909247, 0.41549903816977385, 1.0971439335428022],
        [0.5511484190229397, 0.00953153638467867, 1.7313146238724493],
    ],
    [
        [0.3346816329564619, 1.3135280818751187, 1.0118369194526389],
        [1.7343220731008198, -0.0471491388910467, -2.602000905028717],
    ],
]


@pytest.mark.parametrize(
    "multivariate, params, expected",
    [
        (False, {}, GAS_PRICES_DEFAULT),
        (False, {"bspline": True, "order": 4, "n_basis": 10}, GAS_PRICES_BSPLINE),
        (False, {"n_components": 3, "centering": False}, GAS_PRICES_NO_CENTERING),
        (
            False,
            {
                "n_components": 3,
                "centering": False,
                "bspline": True,
                "order": 3,
                "n_basis": 6,
            },
            GAS_PRICES_BSPLINE_NO_CENTERING,
        ),
        (True, {"n_components": 3}, MULTIVARIATE),
        (
            True,
            {"n_components": 3, "bspline": True, "order": 4, "n_basis": 5},
            MULTIVARIATE_BSPLINE,
        ),
    ],
)
def test_fpca_transformer_matches_scikit_fda(multivariate, params, expected):
    """Test the scores of train and unseen cases match those of scikit-fda 0.10.1.

    The first two sets of parameters are the configurations used in tsml-eval. The
    gas prices series have an even length and the multivariate series an odd length,
    which use different quadrature weights.
    """
    if multivariate:
        X, _ = generate_3d_test_data(
            n_samples=14, n_channels=2, series_length=11, random_state=0
        )
        X_train, X_test = X[:10], X[10:]
    else:
        X_train, _ = load_minimal_gas_prices("TRAIN")
        X_test, _ = load_minimal_gas_prices("TEST")

    fpca = FPCATransformer(**params)
    X_t = np.concatenate((fpca.fit_transform(X_train)[:1], fpca.transform(X_test)[:2]))

    # the sign of each principal component is arbitrary
    expected = np.array(expected)
    X_t = X_t * np.sign((X_t * expected).sum(axis=0))

    assert_allclose(X_t, expected, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize(
    "series_length, expected",
    [
        (2, [1 / 2, 1 / 2]),
        (3, [1 / 3, 4 / 3, 1 / 3]),
        (4, [1 / 3, 5 / 4, 1, 5 / 12]),
        (5, [1 / 3, 4 / 3, 2 / 3, 4 / 3, 1 / 3]),
        (6, [1 / 3, 4 / 3, 2 / 3, 5 / 4, 1, 5 / 12]),
    ],
)
def test_fpca_transformer_quadrature_weights(series_length, expected):
    """Test the weights are those of Simpson's rule in scipy 1.11 and later."""
    X, _ = generate_3d_test_data(series_length=series_length, random_state=0)

    fpca = FPCATransformer(n_components=2)
    fpca.fit(X)

    assert_allclose(fpca._weights, expected)


@pytest.mark.parametrize(
    "params",
    [
        {"n_components": 3},
        {"n_components": 3, "centering": False},
        {"n_components": 3, "bspline": True, "order": 4, "n_basis": 6},
        {
            "n_components": 3,
            "centering": False,
            "bspline": True,
            "order": 2,
            "n_basis": 5,
        },
    ],
)
def test_fpca_transformer_fitted_attributes(params):
    """Test the fitted attributes are consistent with each other and the scores."""
    X, _ = generate_3d_test_data(
        n_samples=20, n_channels=2, series_length=15, random_state=0
    )

    fpca = FPCATransformer(**params)
    X_t = fpca.fit_transform(X)

    n_features = params.get("n_basis", 15)
    assert X_t.shape == (20, 2, 3)
    assert fpca.components_.shape == (2, 3, n_features)
    assert fpca.mean_.shape == (2, n_features)
    assert fpca.explained_variance_.shape == (2, 3)
    assert fpca.explained_variance_ratio_.shape == (2, 3)
    assert fpca.singular_values_.shape == (2, 3)

    for j in range(2):
        # the components are orthonormal functions
        if fpca.bspline:
            products = fpca.components_[j] @ fpca._gram @ fpca.components_[j].T
        else:
            products = (fpca.components_[j] * fpca._weights) @ fpca.components_[j].T
        assert_allclose(products, np.eye(3), atol=1e-10)

        # centering moves the scores of a component without changing their variance
        assert_allclose(X_t[:, j].var(axis=0, ddof=1), fpca.explained_variance_[j])

    if not fpca.bspline:
        assert_allclose(fpca.mean_, X.mean(axis=0))

    # fitting then transforming gives the same scores as doing both together
    assert_allclose(FPCATransformer(**params).fit(X).transform(X), X_t)


def test_fpca_transformer_n_basis_and_order():
    """Test n_basis is raised to order and limits the number of components."""
    X, _ = generate_3d_test_data(n_samples=20, series_length=15, random_state=0)

    fpca = FPCATransformer(bspline=True, order=4, n_basis=3)
    X_t = fpca.fit_transform(X)

    assert X_t.shape == (20, 1, 3)
    assert fpca.components_.shape == (1, 3, 4)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"n_components": 11}, "greater than the number of cases"),
        (
            {"n_components": 11, "bspline": True, "order": 4, "n_basis": 12},
            "greater than the number of cases",
        ),
        ({"n_components": 9}, "greater than the series length"),
        ({"n_components": 0}, "n_components must be at least 1"),
        ({"bspline": True}, "n_basis and order must be set"),
        ({"bspline": True, "n_basis": 5}, "n_basis and order must be set"),
        ({"bspline": True, "order": 4}, "n_basis and order must be set"),
        ({"bspline": True, "order": 0, "n_basis": 5}, "must be at least 1"),
        ({"bspline": True, "order": 4, "n_basis": 0}, "must be at least 1"),
    ],
)
def test_fpca_transformer_invalid_parameters(params, match):
    """Test parameters which are invalid for the data raise an informative error."""
    X, _ = generate_3d_test_data(n_samples=10, series_length=8, random_state=0)

    with pytest.raises(ValueError, match=match):
        FPCATransformer(**params).fit(X)
