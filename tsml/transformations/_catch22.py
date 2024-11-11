"""Catch22 features.

A transformer for the Catch22 features.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["Catch22Transformer"]


import numpy as np
from joblib import Parallel
from sklearn.base import TransformerMixin
from sklearn.utils.parallel import delayed

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.numba_functions.general import z_normalise_series
from tsml.utils.validation import _check_optional_dependency, check_n_jobs

feature_names = [
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "SB_BinaryStats_diff_longstretch0",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
    "CO_trev_1_num",
    "CO_HistogramAMI_even_2_5",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1",
    "SB_MotifThree_quantile_hh",
    "FC_LocalSimple_mean1_tauresrat",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
]


class Catch22Transformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Canonical Time-series Characteristics (Catch22).

    Overview: Input n series with d dimensions of length m
    Transforms series into the 22 Catch22 [1]_ features extracted from the hctsa [2]_
    toolbox.

    Parameters
    ----------
    features : int/str or List of int/str, optional, default="all"
        The Catch22 features to extract by feature index, feature name as a str or as a
        list of names or indices for multiple features. If "all", all features are
        extracted.
        Valid features are as follows:
            ["DN_HistogramMode_5", "DN_HistogramMode_10",
            "SB_BinaryStats_diff_longstretch0", "DN_OutlierInclude_p_001_mdrmd",
            "DN_OutlierInclude_n_001_mdrmd", "CO_f1ecac", "CO_FirstMin_ac",
            "SP_Summaries_welch_rect_area_5_1", "SP_Summaries_welch_rect_centroid",
            "FC_LocalSimple_mean3_stderr", "CO_trev_1_num", "CO_HistogramAMI_even_2_5",
            "IN_AutoMutualInfoStats_40_gaussian_fmmi", "MD_hrv_classic_pnn40",
            "SB_BinaryStats_mean_longstretch1", "SB_MotifThree_quantile_hh",
            "FC_LocalSimple_mean1_tauresrat", "CO_Embed2_Dist_tau_d_expfit_meandiff",
            "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
            "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
            "SB_TransitionMatrix_3ac_sumdiagcov", "PD_PeriodicityWang_th0_01"]
    catch24 : bool, optional, default=False
        Extract the mean and standard deviation as well as the 22 Catch22 features if
        true. If a List of specific features to extract is provided, "Mean" and/or
        "StandardDeviation" must be added to the List to extract these features.
    outlier_norm : bool, optional, default=False
        Normalise each series during the two outlier Catch22 features, which can take a
        while to process for large values.
    replace_nans : bool, optional, default=False
        Replace NaN or inf values from the Catch22 transform with 0.
    use_pycatch22 : bool, optional, default=True
        Wraps the C based pycatch22 implementation for tsml.
        (https://github.com/DynamicsAndNeuralSystems/pycatch22). This requires the
        ``pycatch22`` package to be installed if True.
    n_jobs : int, optional, default=1
        The number of jobs to run in parallel for `transform`. Requires multiple input
        cases. ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.

    See Also
    --------
    Catch22Classifier

    Notes
    -----
    Original Catch22 package implementations:
    https://github.com/DynamicsAndNeuralSystems/Catch22

    For the Java version, see
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /tsml/transformers/Catch22.java

    References
    ----------
    .. [1] Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., &
    Jones, N. S. (2019). catch22: Canonical time-series characteristics. Data Mining
    and Knowledge Discovery, 33(6), 1821-1852.
    .. [2] Fulcher, B. D., Little, M. A., & Jones, N. S. (2013). Highly comparative
    time-series analysis: the empirical structure of time series and their methods.
    Journal of the Royal Society Interface, 10(83), 20130048.

    Examples
    --------
    >>> from tsml.transformations import Catch22Transformer
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, _ = generate_3d_test_data(n_samples=4, series_length=10, random_state=0)
    >>> tnf = Catch22Transformer(replace_nans=True)
    >>> tnf.fit(X)
    Catch22Transformer(...)
    >>> print(tnf.transform(X)[0])
    [6.27596874e-02 3.53871087e-01 4.00000000e+00 7.00000000e-01
     2.00000000e-01 5.66227710e-01 2.00000000e+00 3.08148791e-34
     1.96349541e+00 9.99913411e-01 1.39251594e+00 3.89048349e-01
     2.00000000e+00 1.00000000e+00 3.00000000e+00 2.04319187e+00
     1.00000000e+00 2.44474814e-01 0.00000000e+00 0.00000000e+00
     8.23045267e-03 0.00000000e+00]
    """

    def __init__(
        self,
        features="all",
        catch24=False,
        outlier_norm=False,
        replace_nans=False,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.features = features
        self.catch24 = catch24
        self.outlier_norm = outlier_norm
        self.replace_nans = replace_nans
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        _check_optional_dependency("pycatch22", "pycatch22", self)

        super().__init__()

    def fit(self, X, y=None):
        """Unused. Validates X."""
        self._validate_data(X=X)
        return self

    def transform(self, X, y=None):
        """Transform X into the catch22 features.

        Parameters
        ----------
        X : 3D np.array (any number of channels, equal length series)
                of shape (n_instances, n_channels, n_timepoints)
            or list of numpy arrays (any number of channels, unequal length series)
                of shape [n_instances], 2D np.array (n_channels, n_timepoints_i), where
                n_timepoints_i is length of series i

        Returns
        -------
        Xt : array-like, shape = [n_instances, num_features*n_channels]
            The catch22 features for each dimension.
        """
        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        n_instances = len(X)

        f_idx = _verify_features(self.features, self.catch24)

        threads_to_use = check_n_jobs(self.n_jobs)

        import pycatch22

        features = [
            pycatch22.DN_HistogramMode_5,
            pycatch22.DN_HistogramMode_10,
            pycatch22.SB_BinaryStats_diff_longstretch0,
            pycatch22.DN_OutlierInclude_p_001_mdrmd,
            pycatch22.DN_OutlierInclude_n_001_mdrmd,
            pycatch22.CO_f1ecac,
            pycatch22.CO_FirstMin_ac,
            pycatch22.SP_Summaries_welch_rect_area_5_1,
            pycatch22.SP_Summaries_welch_rect_centroid,
            pycatch22.FC_LocalSimple_mean3_stderr,
            pycatch22.CO_trev_1_num,
            pycatch22.CO_HistogramAMI_even_2_5,
            pycatch22.IN_AutoMutualInfoStats_40_gaussian_fmmi,
            pycatch22.MD_hrv_classic_pnn40,
            pycatch22.SB_BinaryStats_mean_longstretch1,
            pycatch22.SB_MotifThree_quantile_hh,
            pycatch22.FC_LocalSimple_mean1_tauresrat,
            pycatch22.CO_Embed2_Dist_tau_d_expfit_meandiff,
            pycatch22.SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1,
            pycatch22.SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1,
            pycatch22.SB_TransitionMatrix_3ac_sumdiagcov,
            pycatch22.PD_PeriodicityWang_th0_01,
        ]

        c22_list = Parallel(
            n_jobs=threads_to_use, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._transform_case_pycatch22)(
                X[i],
                f_idx,
                features,
            )
            for i in range(n_instances)
        )

        if self.replace_nans:
            c22_list = np.nan_to_num(c22_list, False, 0, 0, 0)

        return np.array(c22_list)

    def _transform_case_pycatch22(self, X, f_idx, features):
        c22 = np.zeros(len(f_idx) * len(X))

        if hasattr(self, "_transform_features") and len(
            self._transform_features
        ) == len(c22):
            transform_feature = self._transform_features
        else:
            transform_feature = [True] * len(c22)

        f_count = -1
        for i in range(len(X)):
            dim = i * len(f_idx)
            series = list(X[i])

            if self.outlier_norm and (3 in f_idx or 4 in f_idx):
                outlier_series = list(z_normalise_series(X[i]))

            for n, feature in enumerate(f_idx):
                f_count += 1
                if not transform_feature[f_count]:
                    continue

                if self.outlier_norm and feature in [3, 4]:
                    c22[dim + n] = features[feature](outlier_series)
                if feature == 22:
                    c22[dim + n] = np.mean(series)
                elif feature == 23:
                    c22[dim + n] = np.std(series)
                else:
                    c22[dim + n] = features[feature](series)

        return c22

    @property
    def get_features_arguments(self):
        """Return feature names for the estimators features argument."""
        return (
            self.features
            if self.features != "all"
            else (
                feature_names + ["Mean", "StandardDeviation"]
                if self.catch24
                else feature_names
            )
        )

    def _more_tags(self) -> dict:
        return {
            "X_types": ["np_list", "3darray"],
            "requires_fit": False,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {}


def _verify_features(features, catch24):
    if isinstance(features, str):
        if features == "all":
            f_idx = [i for i in range(22)]
            if catch24:
                f_idx += [22, 23]
        elif features in feature_names:
            f_idx = [feature_names.index(features)]
        elif catch24 and features == "Mean":
            f_idx = [22]
        elif catch24 and features == "StandardDeviation":
            f_idx = [23]
        else:
            raise ValueError("Invalid feature selection.")
    elif isinstance(features, int):
        if features >= 0 and features < 22:
            f_idx = [features]
        elif catch24 and features == 22:
            f_idx = [22]
        elif catch24 and features == 23:
            f_idx = [23]
        else:
            raise ValueError("Invalid feature selection.")
    elif isinstance(features, (list, tuple)):
        if len(features) > 0:
            f_idx = []
            for f in features:
                if isinstance(f, str):
                    if f in feature_names:
                        f_idx.append(feature_names.index(f))
                    elif catch24 and f == "Mean":
                        f_idx.append(22)
                    elif catch24 and f == "StandardDeviation":
                        f_idx.append(23)
                    else:
                        raise ValueError("Invalid feature selection.")
                elif isinstance(f, int):
                    if f >= 0 and f < 22:
                        f_idx.append(f)
                    elif catch24 and f == 22:
                        f_idx.append(22)
                    elif catch24 and f == 23:
                        f_idx.append(23)
                    else:
                        raise ValueError("Invalid feature selection.")
                else:
                    raise ValueError("Invalid feature selection.")
        else:
            raise ValueError("Invalid feature selection.")
    else:
        raise ValueError("Invalid feature selection.")

    return f_idx
