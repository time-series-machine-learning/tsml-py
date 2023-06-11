# -*- coding: utf-8 -*-
from numpy.testing import assert_array_almost_equal

from tsml.transformations import PeriodogramTransformer
from tsml.utils.testing import generate_3d_test_data


def test_periodogram_same_output():
    """Test that the output is the same using pyfftw and not."""
    X, y = generate_3d_test_data()

    p1 = PeriodogramTransformer()
    p2 = PeriodogramTransformer(use_pyfftw=False)

    assert_array_almost_equal(p1.fit_transform(X), p2.fit_transform(X))
