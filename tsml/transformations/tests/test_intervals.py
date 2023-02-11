# -*- coding: utf-8 -*-
"""RandomIntervals test code."""
import numpy as np
from numpy import testing
from sktime.datasets import load_basic_motions
from sktime.transformations.panel.catch22 import Catch22
from sktime.transformations.panel.random_intervals import RandomIntervals
from sktime.transformations.panel.supervised_intervals import SupervisedIntervals
from sktime.utils.numba.stats import row_mean, row_numba_max, row_numba_min


def test_random_intervals_on_basic_motions():
    """Test of RandomIntervals on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    # fit random intervals
    ri = RandomIntervals(random_state=0, n_intervals=3)
    data = ri.fit_transform(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    testing.assert_array_almost_equal(
        data, random_intervals_basic_motions_data, decimal=4
    )


def test_random_intervals_feature_skip():
    """Test of RandomIntervals with skipped features on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    # fit random intervals
    ri = RandomIntervals(
        random_state=0,
        n_intervals=2,
        features=[
            Catch22(features=["DN_HistogramMode_5", "DN_HistogramMode_10"]),
            Catch22(
                catch24=True,
                features=["DN_HistogramMode_5", "Mean", "StandardDeviation"],
            ),
            row_mean,
            row_numba_min,
            row_numba_max,
        ],
    )

    ri.fit(X_train.iloc[indices], y_train[indices])
    data = ri.transform(X_train.iloc[indices]).to_numpy()

    arr = [True] * 16
    for i in [1, 7, 9, 10, 14, 15]:
        arr[i] = False
    ri.set_features_to_transform(arr)

    skip_data = ri.transform(X_train.iloc[indices]).to_numpy()

    for i in range(len(arr)):
        if arr[i]:
            testing.assert_array_almost_equal(data[:, i], skip_data[:, i], decimal=4)
        else:
            testing.assert_array_almost_equal(
                [0, 0, 0, 0, 0], skip_data[:, i], decimal=4
            )


def test_supervised_intervals_on_basic_motions():
    """Test of SupervisedIntervals on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    # fit supervised intervals
    si = SupervisedIntervals(random_state=0, n_intervals=1, randomised_split_point=True)
    data = si.fit_transform(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    testing.assert_array_almost_equal(
        data, supervised_intervals_basic_motions_data, decimal=4
    )


random_intervals_basic_motions_data = np.array(
    [
        [
            0.0693,
            3.5492,
            -5.1601,
            5.837,
            -3.224,
            0.1725,
            3.6041,
            -0.0466,
            0.5553,
            -0.9695,
            0.6312,
            -0.5327,
            0.1811,
            0.4042,
            -0.3882,
            0.7039,
            -1.3554,
            0.703,
            -0.9854,
            -0.4856,
            0.2672,
        ],
        [
            0.59,
            7.384,
            -17.0329,
            14.971,
            -3.4541,
            0.8937,
            5.2368,
            -0.1824,
            3.6398,
            -8.1925,
            6.6771,
            -2.4663,
            -0.4142,
            0.8097,
            -4.4477,
            5.5903,
            -21.5251,
            3.3514,
            -6.0732,
            -3.9562,
            -0.4152,
        ],
        [
            0.0748,
            0.4744,
            -0.6229,
            1.1303,
            -0.3131,
            0.053,
            0.4611,
            -0.0203,
            0.0458,
            -0.0932,
            0.0692,
            -0.0473,
            -0.0173,
            -0.002,
            -0.0073,
            0.0932,
            -0.2562,
            0.1533,
            -0.0281,
            0.006,
            0.0323,
        ],
        [
            -4.6708,
            13.4856,
            -22.5108,
            23.366,
            -15.1675,
            -10.6761,
            8.5336,
            0.2304,
            2.6666,
            -5.9846,
            4.6982,
            -1.1359,
            -0.5007,
            2.2053,
            -3.7573,
            2.7504,
            -10.9416,
            0.5036,
            -4.9566,
            -3.6576,
            -2.1361,
        ],
        [
            0.1345,
            3.1222,
            -5.0414,
            5.0777,
            -2.5675,
            0.263,
            3.24,
            -0.0078,
            0.9858,
            -2.3411,
            1.7418,
            -0.3422,
            -0.0133,
            0.6792,
            -0.5084,
            0.6721,
            -1.5731,
            0.8509,
            -0.8769,
            -0.581,
            -0.1604,
        ],
    ]
)
supervised_intervals_basic_motions_data = np.array(
    [
        [
            0.626,
            0.7227,
            0.3178,
            0.9688,
            0.7869,
            1.236,
            0.6791,
            0.5434,
            0.4455,
            0.4455,
            0.684,
            0.5485,
            0.8139,
            0.8614,
            1.0003,
            1.0003,
            0.9257,
            1.1139,
            -0.017,
            0.2045,
            0.225,
            -0.0024,
            -0.0518,
            0.0229,
            0.0065,
            0.0585,
            -1.0121,
            -0.8239,
            -0.8115,
            -0.8115,
            -0.8115,
            -0.9448,
            0.1642,
            2.7955,
            2.7955,
            1.8018,
            1.8018,
            3.2893,
            3.2893,
            3.2893,
            3.2332,
            3.2332,
            1.6751,
            1.816,
            2.4171,
            1.6654,
            1.1895,
            5.0,
            2.0,
            2.0,
            6.0,
            2.0,
            1.0,
            24.0,
            15.0,
            11.0,
            3.0,
            2.0,
            -1.647,
            -0.0381,
            0.2157,
            0.4265,
            2.6268,
            0.056,
            0.4237,
            0.9371,
            1.1772,
            1.1772,
            2.7923,
            -0.8065,
            1.462,
            3.7191,
            3.9181,
            3.4678,
            2.3107,
            3.5808,
            3.6162,
            2.7454,
            0.0053,
            0.1471,
            3.9589,
            -0.6521,
            0.014,
            -4.7231,
            -4.5975,
            -4.5975,
            -5.2175,
            -5.2175,
            -5.2175,
            -5.2175,
            -5.2175,
            3.9983,
            1.6519,
            -3.4958,
            5.5333,
            5.5333,
            5.5333,
            6.9543,
            6.953,
            6.8239,
            0.2703,
            5.6656,
            4.6279,
            1.1417,
            1.0,
            2.0,
            1.0,
            4.0,
            3.0,
            2.0,
            6.0,
            4.0,
            2.0,
            0.208,
            0.3239,
            -0.5247,
            -0.2383,
            -0.0047,
            0.1646,
            0.1646,
            -0.5616,
            -0.269,
            -0.5616,
            0.5479,
            0.7763,
            0.6494,
            0.7987,
            0.4333,
            0.015,
            0.242,
            0.2337,
            0.3991,
            0.0177,
            -0.085,
            0.183,
            -1.3511,
            -1.3511,
            -2.1784,
            -2.1784,
            -2.1784,
            -1.4846,
            -1.4846,
            -0.1738,
            2.3167,
            2.3167,
            2.3167,
            0.9789,
            0.8038,
            0.4039,
            0.4039,
            -0.2675,
            1.1344,
            0.9447,
            14.0,
            9.0,
            3.0,
            2.0,
            2.0,
            1.0,
            4.0,
            1.0,
            38.0,
            3.0,
            1.0,
            4.0,
            3.0,
            0.0662,
            0.3384,
            0.4251,
            -0.0442,
            -0.0182,
            -0.1567,
            -0.0646,
            -0.4064,
            -0.4168,
            -0.1438,
            0.0746,
            -0.0453,
            0.4661,
            0.4816,
            0.2389,
            0.1289,
            0.5949,
            0.6332,
            0.4664,
            0.3131,
            0.0014,
            -0.0468,
            0.1092,
            -0.0435,
            -0.0138,
            -0.6499,
            0.1944,
            -1.0041,
            -1.0041,
            -1.0041,
            -0.277,
            1.3903,
            1.3903,
            1.0387,
            0.6312,
            0.6312,
            -0.2024,
            0.9189,
            0.7997,
            0.2597,
            0.4341,
            13.0,
            4.0,
            1.0,
            6.0,
            5.0,
            4.0,
            4.0,
            1.0,
            11.0,
            8.0,
            3.0,
            1.0,
            21.0,
            7.0,
            3.0,
            2.0,
            0.0375,
            0.0093,
            -0.016,
            0.0756,
            -0.1467,
            -0.0725,
            -0.4817,
            -0.2424,
            -0.0107,
            0.0027,
            0.0,
            0.0226,
            -0.2557,
            0.1571,
            0.2059,
            0.4901,
            0.4302,
            0.5083,
            0.5113,
            0.3875,
            -0.0351,
            -0.053,
            -0.0549,
            0.0176,
            -0.0005,
            0.2834,
            -0.6499,
            -0.6206,
            -1.0014,
            -1.0014,
            -1.0014,
            -1.0014,
            -1.0014,
            -0.9562,
            0.9428,
            -0.1838,
            0.2557,
            0.7298,
            0.7218,
            0.2184,
            0.6792,
            0.7284,
            0.6692,
            0.8316,
            0.1398,
            2.0,
            1.0,
            3.0,
            2.0,
            31.0,
            19.0,
            9.0,
            5.0,
            3.0,
            6.0,
            2.0,
            1.9316,
            -0.0549,
            -0.0184,
            1.7379,
            0.8842,
            -0.1159,
            -0.4847,
            -0.2504,
            -0.6046,
            0.6807,
            0.3631,
            -0.4733,
            -0.0031,
            -0.0203,
            0.4792,
            0.3729,
            -2.3384,
            -1.9736,
            -1.2331,
            1.2917,
            2.6314,
            1.7312,
            1.353,
            1.353,
            0.807,
            2.8232,
            2.8924,
            1.8138,
            0.9801,
            1.4995,
            0.4481,
            5.0,
            3.0,
            2.0,
            2.0,
            1.0,
            1.0,
            2.0,
            16.0,
            8.0,
            4.0,
            2.0,
            4.0,
            2.0,
        ],
        [
            6.5719,
            8.2935,
            12.0961,
            7.1972,
            2.6966,
            3.2518,
            1.7161,
            5.2718,
            9.3521,
            13.0218,
            4.4759,
            3.4477,
            2.342,
            0.7107,
            7.9835,
            5.5422,
            4.7486,
            1.4328,
            -0.2138,
            0.1157,
            -0.909,
            -0.014,
            0.1107,
            -0.5285,
            0.359,
            -4.8076,
            -1.9252,
            -1.9252,
            -1.9252,
            6.2461,
            12.5742,
            -2.4465,
            5.2718,
            29.2109,
            29.2109,
            29.2109,
            29.2109,
            25.0083,
            16.2637,
            16.2637,
            11.8029,
            9.9696,
            10.2495,
            3.817,
            5.0551,
            5.3575,
            1.9979,
            1.0,
            1.0,
            3.0,
            5.0,
            4.0,
            2.0,
            22.0,
            15.0,
            8.0,
            3.0,
            3.0,
            -1.8467,
            1.0829,
            -0.3004,
            -0.4342,
            0.1941,
            0.4297,
            0.2463,
            -1.1148,
            -3.4779,
            -3.4779,
            -6.3323,
            0.5164,
            1.4633,
            7.9899,
            8.8212,
            9.8567,
            11.9041,
            4.7311,
            4.3192,
            1.8308,
            0.1143,
            -0.5099,
            -2.1225,
            -2.7335,
            -1.0768,
            -17.0329,
            -3.0882,
            0.4022,
            -10.2532,
            -4.9047,
            -3.4086,
            0.4341,
            0.4341,
            3.1305,
            3.1305,
            2.0726,
            12.8606,
            10.2274,
            9.7027,
            7.2582,
            10.3641,
            9.3943,
            3.3204,
            4.435,
            5.4961,
            9.8812,
            2.0,
            3.0,
            2.0,
            5.0,
            4.0,
            2.0,
            8.0,
            6.0,
            3.0,
            -2.6856,
            -8.35,
            -11.6095,
            -2.7092,
            -0.4124,
            0.3668,
            0.3278,
            -2.0193,
            -2.9727,
            -0.7441,
            1.6884,
            1.425,
            5.4383,
            7.2685,
            6.5232,
            0.0837,
            1.0265,
            -0.7899,
            -0.5396,
            -0.0031,
            0.1225,
            -1.1717,
            -21.5251,
            -21.5251,
            -20.5453,
            -17.9567,
            -17.9567,
            -17.9567,
            -17.9567,
            -17.9567,
            7.1132,
            7.1132,
            7.1132,
            7.1132,
            7.1132,
            5.684,
            -0.339,
            -0.339,
            9.3371,
            6.5964,
            13.0,
            12.0,
            9.0,
            5.0,
            2.0,
            1.0,
            2.0,
            2.0,
            41.0,
            3.0,
            2.0,
            2.0,
            2.0,
            -0.9121,
            -1.1924,
            -5.3843,
            0.1436,
            0.1492,
            -0.3027,
            0.0182,
            -0.9489,
            -1.7285,
            0.0586,
            0.2251,
            0.5753,
            0.3382,
            3.8801,
            4.5466,
            1.7466,
            2.1565,
            1.7756,
            1.3827,
            1.0115,
            -0.0543,
            -1.1091,
            -0.6459,
            -0.4213,
            0.1103,
            -13.8948,
            0.0613,
            -5.4066,
            -5.4066,
            -5.4066,
            -3.6728,
            6.2856,
            6.2856,
            -2.0428,
            6.6771,
            2.1627,
            2.1627,
            2.4976,
            0.4415,
            2.3797,
            0.7164,
            12.0,
            8.0,
            3.0,
            8.0,
            5.0,
            6.0,
            5.0,
            1.0,
            14.0,
            10.0,
            3.0,
            1.0,
            22.0,
            10.0,
            6.0,
            3.0,
            -0.3723,
            -0.6256,
            -1.3881,
            -2.0167,
            0.2956,
            0.3668,
            0.9912,
            1.0773,
            -0.6951,
            -0.7018,
            -0.7857,
            -0.6951,
            -0.5633,
            -2.0668,
            0.3767,
            2.7968,
            1.2211,
            0.913,
            0.7892,
            0.7283,
            -0.0524,
            0.0036,
            -0.9137,
            -1.2803,
            0.1017,
            1.4667,
            -1.3264,
            -1.3264,
            -10.5603,
            -10.5603,
            -10.5603,
            -10.5603,
            -10.5603,
            -10.5603,
            -0.5273,
            -0.5273,
            1.4728,
            2.1893,
            0.6392,
            0.3995,
            1.7538,
            1.5188,
            1.7951,
            0.9741,
            0.532,
            2.0,
            1.0,
            1.0,
            1.0,
            26.0,
            19.0,
            11.0,
            7.0,
            2.0,
            7.0,
            2.0,
            -8.4016,
            1.2228,
            1.2993,
            3.1321,
            0.7671,
            0.4355,
            1.6486,
            0.2477,
            -3.0948,
            2.4823,
            0.2048,
            0.5881,
            0.0046,
            -0.1377,
            0.4865,
            1.1732,
            -24.5163,
            -0.3596,
            -2.4157,
            -2.4157,
            1.377,
            7.7824,
            7.7824,
            7.1698,
            2.1094,
            1.9829,
            3.4218,
            2.8072,
            4.9113,
            2.2279,
            1.1945,
            8.0,
            5.0,
            4.0,
            1.0,
            1.0,
            2.0,
            1.0,
            15.0,
            8.0,
            3.0,
            1.0,
            5.0,
            2.0,
        ],
        [
            -0.2386,
            -0.2511,
            -0.22,
            -0.2654,
            -0.3013,
            -0.3253,
            -0.3362,
            -0.2628,
            -0.1765,
            -0.1765,
            -0.2949,
            -0.3,
            -0.3764,
            0.1526,
            0.1477,
            0.1629,
            0.1581,
            0.1312,
            -0.0054,
            0.0077,
            -0.0632,
            -0.0007,
            -0.0027,
            -0.0299,
            -0.0677,
            -0.1437,
            -0.5613,
            -0.5613,
            -0.5613,
            -0.3341,
            -0.3332,
            -0.4948,
            -0.4948,
            0.0579,
            0.0579,
            -0.1636,
            -0.2313,
            0.126,
            0.126,
            0.126,
            -0.2437,
            -0.3721,
            0.1982,
            0.1582,
            0.2537,
            0.1082,
            0.045,
            6.0,
            4.0,
            2.0,
            10.0,
            5.0,
            3.0,
            26.0,
            14.0,
            10.0,
            2.0,
            3.0,
            0.1967,
            0.0913,
            0.1044,
            0.1351,
            0.3001,
            0.2196,
            0.107,
            0.1778,
            0.2827,
            0.3761,
            0.5104,
            -0.0195,
            -0.2538,
            0.4758,
            0.448,
            0.5026,
            0.3705,
            0.414,
            0.4216,
            0.4447,
            -0.0026,
            -0.0037,
            0.1688,
            0.1356,
            0.0697,
            -0.6229,
            -0.5379,
            0.31,
            -0.5077,
            -0.5077,
            -0.5077,
            -0.4215,
            0.4805,
            0.6402,
            0.6402,
            0.6402,
            0.6691,
            0.4805,
            -0.2615,
            0.855,
            0.774,
            0.5405,
            0.4522,
            0.4618,
            0.3557,
            0.3084,
            1.0,
            2.0,
            2.0,
            5.0,
            3.0,
            2.0,
            6.0,
            4.0,
            3.0,
            0.041,
            0.0642,
            0.0684,
            0.0302,
            0.0814,
            -0.1382,
            -0.286,
            -0.0369,
            -0.0466,
            -0.0423,
            0.1069,
            0.0679,
            0.0888,
            0.107,
            0.0943,
            0.0,
            0.006,
            0.0025,
            0.0062,
            0.0026,
            -0.004,
            0.0451,
            -0.2562,
            -0.2562,
            -0.209,
            -0.209,
            -0.1657,
            -0.1657,
            -0.1331,
            -0.1331,
            0.5172,
            0.5172,
            0.5172,
            0.5172,
            0.5172,
            0.1941,
            0.0193,
            -0.0369,
            0.1088,
            0.0871,
            19.0,
            15.0,
            5.0,
            3.0,
            2.0,
            1.0,
            5.0,
            1.0,
            37.0,
            4.0,
            1.0,
            4.0,
            2.0,
            -0.0093,
            0.0185,
            0.073,
            -0.0189,
            -0.0238,
            -0.0277,
            -0.0493,
            -0.0145,
            -0.012,
            -0.0985,
            -0.0133,
            -0.0346,
            -0.0346,
            0.0774,
            0.0856,
            0.0602,
            0.0417,
            0.0483,
            0.0478,
            0.0085,
            0.0023,
            -0.0097,
            0.0333,
            0.0052,
            0.0104,
            -0.1252,
            0.1571,
            -0.1518,
            -0.1518,
            -0.1518,
            -0.1518,
            0.1252,
            0.1252,
            0.1252,
            0.0,
            0.0,
            0.0,
            0.1292,
            0.0293,
            0.0146,
            0.04,
            9.0,
            8.0,
            1.0,
            6.0,
            5.0,
            3.0,
            4.0,
            2.0,
            11.0,
            8.0,
            3.0,
            2.0,
            20.0,
            8.0,
            5.0,
            2.0,
            0.0104,
            -0.0069,
            -0.0184,
            -0.025,
            0.011,
            0.0016,
            -0.0008,
            0.0067,
            0.008,
            0.0067,
            0.008,
            0.0013,
            -0.0692,
            -0.0186,
            0.0552,
            0.0447,
            0.0359,
            0.0467,
            0.0484,
            0.0111,
            -0.0028,
            -0.0044,
            -0.0004,
            0.0165,
            0.0005,
            -0.0575,
            0.0053,
            0.0586,
            -0.1358,
            -0.1358,
            -0.1358,
            -0.0559,
            -0.0559,
            -0.0559,
            0.1758,
            0.1758,
            0.0,
            0.0559,
            0.0266,
            0.0586,
            0.0493,
            0.0493,
            0.0579,
            0.04,
            0.0719,
            4.0,
            2.0,
            2.0,
            1.0,
            27.0,
            16.0,
            7.0,
            4.0,
            3.0,
            4.0,
            1.0,
            0.1232,
            0.0366,
            0.0604,
            0.2231,
            -0.0586,
            -0.0107,
            0.0479,
            0.0479,
            0.1225,
            0.1726,
            0.0585,
            0.1305,
            0.0001,
            -0.0067,
            -0.0792,
            -0.0786,
            -0.1651,
            -0.1571,
            -0.1225,
            -0.0506,
            0.2903,
            0.1838,
            0.1838,
            0.1838,
            -0.0719,
            0.2117,
            0.2231,
            0.1918,
            0.2983,
            0.261,
            0.1658,
            7.0,
            5.0,
            3.0,
            2.0,
            1.0,
            2.0,
            1.0,
            13.0,
            8.0,
            4.0,
            2.0,
            5.0,
            2.0,
        ],
        [
            4.9693,
            4.6136,
            5.9108,
            18.9091,
            9.3738,
            13.8804,
            11.5766,
            13.6674,
            -0.6101,
            -0.6101,
            11.1454,
            12.5917,
            14.0189,
            15.3765,
            13.6393,
            13.6346,
            13.0582,
            5.8662,
            2.1228,
            -2.2677,
            2.1855,
            0.0238,
            -0.9947,
            0.4832,
            5.0363,
            9.7028,
            -19.5103,
            -17.5795,
            -17.5795,
            -17.5795,
            -17.5795,
            -16.928,
            -7.3626,
            23.3188,
            23.3188,
            19.8883,
            19.8883,
            21.4385,
            21.4385,
            20.9104,
            20.9104,
            -0.1914,
            22.6803,
            18.9769,
            11.5968,
            16.2478,
            0.8931,
            4.0,
            2.0,
            1.0,
            3.0,
            2.0,
            1.0,
            26.0,
            17.0,
            12.0,
            4.0,
            4.0,
            -0.5006,
            -4.5361,
            -6.0035,
            -7.668,
            -17.1291,
            -10.4552,
            -9.9763,
            -10.1587,
            -10.8458,
            -9.7962,
            -7.2535,
            -11.8329,
            -15.6726,
            13.6299,
            13.9041,
            11.5526,
            3.6235,
            13.1029,
            13.3587,
            13.5704,
            0.0052,
            -1.2133,
            -2.0341,
            3.8787,
            7.7748,
            -20.4405,
            -19.9688,
            -19.9688,
            -18.2651,
            -18.2651,
            -18.2651,
            -18.2651,
            -18.2651,
            20.1147,
            20.1147,
            20.1147,
            20.2195,
            19.3417,
            3.8734,
            18.8669,
            16.5185,
            18.2906,
            4.5155,
            23.8068,
            6.2318,
            3.9221,
            1.0,
            3.0,
            1.0,
            4.0,
            2.0,
            2.0,
            4.0,
            3.0,
            2.0,
            -2.0868,
            -3.0887,
            -5.2977,
            -2.4415,
            -1.0172,
            -0.0646,
            -0.0646,
            -0.5295,
            -0.1384,
            -0.6629,
            3.0324,
            2.835,
            2.9084,
            2.7418,
            2.256,
            0.1121,
            0.3182,
            -0.0259,
            1.2784,
            -0.0215,
            0.3051,
            -0.8339,
            -10.9416,
            -10.9416,
            -9.9634,
            -9.9634,
            -9.9634,
            -9.9634,
            -3.6422,
            -1.6446,
            1.921,
            1.921,
            1.921,
            1.921,
            1.817,
            2.1536,
            2.1536,
            1.9978,
            1.324,
            0.3728,
            28.0,
            23.0,
            8.0,
            5.0,
            3.0,
            2.0,
            3.0,
            3.0,
            42.0,
            4.0,
            2.0,
            4.0,
            2.0,
            0.3865,
            0.7008,
            0.5284,
            0.0608,
            -0.0018,
            -0.0799,
            0.2468,
            0.6103,
            1.2032,
            0.04,
            -0.6845,
            0.8723,
            2.0215,
            2.2897,
            1.4545,
            1.5154,
            1.9729,
            1.9862,
            1.8386,
            1.8133,
            -0.0531,
            -0.1892,
            -2.8405,
            -0.0298,
            -0.1553,
            -4.4825,
            -3.6755,
            -4.5517,
            -4.5517,
            -4.5517,
            1.622,
            6.1737,
            6.1737,
            2.6074,
            3.9285,
            3.9285,
            3.9285,
            1.7851,
            0.4268,
            1.6566,
            2.7606,
            23.0,
            9.0,
            1.0,
            8.0,
            5.0,
            3.0,
            3.0,
            1.0,
            7.0,
            7.0,
            2.0,
            2.0,
            18.0,
            8.0,
            3.0,
            2.0,
            0.2068,
            0.2305,
            -0.3789,
            -0.9141,
            0.2761,
            0.126,
            0.0544,
            0.548,
            -0.4022,
            -0.5713,
            -0.6206,
            -0.5713,
            1.4502,
            -2.9404,
            0.29,
            2.0979,
            1.8121,
            1.5896,
            1.6379,
            1.8723,
            0.0986,
            -0.1313,
            -0.8862,
            -0.4288,
            0.0029,
            1.2606,
            -0.4022,
            -0.0426,
            -3.0522,
            -3.0522,
            -3.0522,
            -2.2266,
            -1.8058,
            -1.8058,
            0.5726,
            0.5726,
            -0.2397,
            0.9375,
            0.9002,
            0.3076,
            2.2838,
            2.1607,
            2.1853,
            2.3591,
            1.1979,
            1.0,
            2.0,
            2.0,
            1.0,
            18.0,
            10.0,
            5.0,
            2.0,
            1.0,
            4.0,
            2.0,
            2.3131,
            0.6707,
            -0.8847,
            -3.9618,
            -5.0524,
            -0.6152,
            1.9576,
            -0.791,
            5.356,
            5.9829,
            2.5068,
            4.5458,
            -0.0561,
            -0.1034,
            -0.995,
            3.0059,
            -11.4685,
            -4.2827,
            -9.1194,
            -9.1194,
            10.9758,
            7.8197,
            7.8197,
            7.8197,
            7.8197,
            8.1073,
            11.1748,
            8.9649,
            3.1348,
            7.2058,
            6.1844,
            8.0,
            7.0,
            4.0,
            3.0,
            2.0,
            1.0,
            1.0,
            16.0,
            9.0,
            4.0,
            2.0,
            3.0,
            3.0,
        ],
        [
            0.9442,
            0.7444,
            0.6902,
            0.6376,
            0.4957,
            -0.1092,
            0.6829,
            0.679,
            0.304,
            0.5325,
            0.5398,
            0.6665,
            0.3049,
            0.2999,
            1.3156,
            1.2707,
            1.0574,
            1.3459,
            0.0868,
            0.2289,
            0.4413,
            -0.0056,
            -0.0782,
            0.1461,
            -0.0871,
            -1.0255,
            -1.2445,
            -1.2445,
            -1.2445,
            -1.2445,
            -1.2445,
            -1.5622,
            -0.3122,
            3.4906,
            2.1394,
            2.1394,
            2.0806,
            3.5911,
            3.1539,
            3.1539,
            3.131,
            2.5775,
            2.1163,
            1.4076,
            2.2262,
            2.2663,
            1.4929,
            5.0,
            3.0,
            2.0,
            6.0,
            3.0,
            2.0,
            24.0,
            15.0,
            10.0,
            3.0,
            2.0,
            -0.8955,
            0.1643,
            0.124,
            1.0359,
            1.394,
            0.3496,
            0.3496,
            0.1945,
            0.5349,
            1.3851,
            3.0534,
            0.6518,
            0.9988,
            3.0595,
            2.77,
            2.9574,
            2.6059,
            2.8157,
            3.122,
            2.635,
            0.0062,
            0.4826,
            0.5927,
            -0.0994,
            -1.4582,
            -5.0414,
            -4.6967,
            -4.6967,
            -4.8493,
            -3.4368,
            -3.4368,
            -1.9472,
            -1.9472,
            4.4576,
            1.6656,
            1.6656,
            3.7276,
            3.7276,
            1.9038,
            5.5883,
            5.5883,
            5.804,
            1.0731,
            4.8604,
            3.059,
            0.5368,
            2.0,
            2.0,
            1.0,
            4.0,
            3.0,
            3.0,
            6.0,
            4.0,
            2.0,
            -0.0455,
            -0.4001,
            -0.7657,
            -0.3157,
            -0.8293,
            0.0524,
            -0.0645,
            -0.7074,
            -0.8944,
            -0.9304,
            0.558,
            0.3167,
            0.6102,
            0.5997,
            0.3096,
            0.0025,
            0.152,
            0.1762,
            0.0781,
            0.0032,
            0.0733,
            0.2437,
            -0.9023,
            -0.9023,
            -1.9773,
            -1.9773,
            -1.2871,
            -1.2871,
            -0.3966,
            -0.3966,
            0.931,
            0.931,
            0.931,
            0.931,
            0.1707,
            0.5132,
            -0.2252,
            -0.7074,
            0.9762,
            0.6712,
            15.0,
            14.0,
            3.0,
            2.0,
            1.0,
            1.0,
            4.0,
            1.0,
            37.0,
            2.0,
            1.0,
            4.0,
            2.0,
            0.0883,
            0.0573,
            0.7159,
            -0.0359,
            -0.0598,
            0.0351,
            -0.1598,
            -0.2233,
            -0.5447,
            0.0692,
            -0.0972,
            -0.0439,
            -1.1479,
            0.6642,
            0.4435,
            0.0866,
            0.8863,
            0.7026,
            0.5505,
            0.4737,
            -0.0043,
            -0.0732,
            0.2916,
            -0.1361,
            -0.0064,
            -1.1959,
            -0.0027,
            -1.0867,
            -0.9562,
            -0.7431,
            0.7671,
            1.5714,
            1.5714,
            1.5714,
            0.9322,
            0.7671,
            0.7671,
            0.3928,
            0.1798,
            0.3862,
            0.1412,
            8.0,
            5.0,
            1.0,
            7.0,
            4.0,
            4.0,
            4.0,
            1.0,
            11.0,
            8.0,
            3.0,
            1.0,
            19.0,
            7.0,
            4.0,
            2.0,
            0.0552,
            0.1089,
            -0.1076,
            0.13,
            -0.0459,
            -0.0631,
            -0.0384,
            -0.7844,
            0.0772,
            0.0613,
            0.0772,
            0.0826,
            -0.3596,
            0.032,
            0.2712,
            0.6288,
            0.7145,
            0.6584,
            0.6542,
            0.3882,
            -0.0287,
            -0.0838,
            -0.1079,
            -0.0003,
            -0.0085,
            0.3542,
            -0.4501,
            -0.3329,
            -1.2038,
            -1.0547,
            -1.0547,
            -1.0547,
            -0.3942,
            0.4075,
            0.2424,
            0.2424,
            0.7804,
            0.6299,
            0.6206,
            0.2876,
            0.9735,
            1.0926,
            1.0314,
            0.9462,
            0.7664,
            2.0,
            1.0,
            2.0,
            1.0,
            32.0,
            16.0,
            8.0,
            5.0,
            3.0,
            5.0,
            3.0,
            1.1819,
            -0.1489,
            0.002,
            1.8823,
            -0.7111,
            -0.0373,
            0.4847,
            -0.1625,
            0.6099,
            0.5389,
            0.9512,
            -0.8387,
            0.0068,
            0.0278,
            0.4929,
            0.7457,
            -1.7312,
            -1.7312,
            -0.8869,
            1.1026,
            0.9588,
            1.6886,
            1.6886,
            1.2252,
            1.2252,
            1.7312,
            1.893,
            1.875,
            0.6951,
            1.0294,
            0.9328,
            4.0,
            3.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            15.0,
            8.0,
            4.0,
            2.0,
            3.0,
            2.0,
        ],
    ]
)
