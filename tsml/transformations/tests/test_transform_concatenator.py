"""Tests for the TransformerConcatenator class."""

from tsml.transformations import (
    FunctionTransformer,
    SevenNumberSummaryTransformer,
    TransformerConcatenator,
)
from tsml.utils.numba_functions.general import first_order_differences_3d
from tsml.utils.testing import generate_3d_test_data


def test_concatenate_features():
    """Test TransformerConcatenator on features."""
    X, y = generate_3d_test_data()

    concat = TransformerConcatenator(
        transformers=[
            SevenNumberSummaryTransformer(),
            SevenNumberSummaryTransformer(),
        ]
    )

    assert concat.fit_transform(X).shape == (X.shape[0], 14)


def test_concatenate_series():
    """Test TransformerConcatenator on series."""
    X, y = generate_3d_test_data()

    concat = TransformerConcatenator(
        transformers=[
            FunctionTransformer(func=first_order_differences_3d),
            FunctionTransformer(func=first_order_differences_3d),
        ]
    )

    assert concat.fit_transform(X).shape == (X.shape[0], 1, 22)
