# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.utils.validation import _check_optional_dependency


class MrSQMClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Multiple Representations Sequence Miner (MrSQM) classifier.

    This is a wrapper for the MrSQMClassifier algorithm from the `mrsqm` package.
    MrSQM is not included in all extras as it requires gcc and fftw
    (http://www.fftw.org/index.html) to be installed for Windows and some Linux OS.

    Overview: MrSQM is an efficient time series classifier utilising symbolic
    representations of time series. MrSQM implements four different feature selection
    strategies (R,S,RS,SR) that can quickly select subsequences from multiple symbolic
    representations of time series data.

    Parameters
    ----------
    strat : 'R','S','SR', or 'RS;, default="RS"
        Feature selection strategy. R and S are single-stage filters while
        RS and SR are two-stage filters. By default set to 'RS'.
    features_per_rep : int, default=500
        The (maximum) number of features selected per representation.
    selection_per_rep : int, default=2000
        The (maximum) number of candidate features selected per representation.
        Only applied in two stages strategies (RS and SR).
    nsax : int, default=0
        The number of representations produced by SAX transformation.
    nsfa : int, default=5
        The number of representations produced by SFA transformation.
        Note: including any SFA transformations will prevent the estimator from being
        serialised (no pickling).
    custom_config : dict, default=None
        Customized parameters for the symbolic transformation.
    random_state : int or None, default=None
        Random seed for classifier.
    sfa_norm : bool, default=True
        Time series normalisation (standardisation).

    Notes
    -----
    The `mrsqm` package uses a different license (GPL-3.0) from the aeon BSD3 license
    covering this interface wrapper.
    See https://github.com/mlgig/mrsqm for the original implementation.

    References
    ----------
    .. [1] Nguyen, Thach Le, and Georgiana Ifrim. "Fast time series classification with
        random symbolic subsequences." Advanced Analytics and Learning on Temporal Data:
        7th ECML PKDD Workshop, AALTD 2022, Grenoble, France, September 19–23, 2022.
    .. [2] Nguyen, Thach Le, and Georgiana Ifrim. "MrSQM: Fast time series
        classification with symbolic representations." arXiv preprint arXiv:2109.01036
        (2021).

    Examples
    --------
    >>> from tsml.shapelet_based import MrSQMClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(random_state=0)
    >>> clf = MrSQMClassifier(random_state=0)  # doctest: +SKIP
    >>> clf.fit(X, y)  # doctest: +SKIP
    MrSQMClassifier(...)
    >>> clf.predict(X)  # doctest: +SKIP
    """

    def __init__(
        self,
        strat="RS",
        features_per_rep=500,
        selection_per_rep=2000,
        nsax=0,
        nsfa=5,
        sfa_norm=True,
        custom_config=None,
        random_state=None,
    ):
        self.strat = strat
        self.features_per_rep = features_per_rep
        self.selection_per_rep = selection_per_rep
        self.nsax = nsax
        self.nsfa = nsfa
        self.sfa_norm = sfa_norm
        self.custom_config = custom_config
        self.random_state = random_state

        _check_optional_dependency("mrsqm", "mrsqm", self)

        super(MrSQMClassifier, self).__init__()

    def fit(self, X, y):
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_dims_, self.series_length_ = (
            X.shape if X.ndim == 3 else (X.shape[0], 1, X.shape[1])
        )
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        from mrsqm import MrSQMClassifier

        self.clf_ = MrSQMClassifier(
            strat=self.strat,
            features_per_rep=self.features_per_rep,
            selection_per_rep=self.selection_per_rep,
            nsax=self.nsax,
            nsfa=self.nsfa,
            sfa_norm=self.sfa_norm,
            custom_config=self.custom_config,
            random_state=self.random_state,
        )
        self.clf_.fit(_convert_data(X), y)

        return self

    def predict(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        return self.clf_.predict(_convert_data(X))

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        return self.clf_.predict_proba(_convert_data(X))

    def _more_tags(self):
        return {
            "non_deterministic": True,
            "_xfail_checks": {"check_estimators_pickle": "External failure to pickle."},
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """

        return {
            "features_per_rep": 50,
            "selection_per_rep": 200,
            "nsax": 1,
            "nsfa": 1,
        }


def _convert_data(X):
    column_list = []
    for i in range(X.shape[1]):
        nested_column = (
            pd.DataFrame(X[:, i, :])
            .apply(lambda x: [pd.Series(x, dtype=X.dtype)], axis=1)
            .str[0]
            .rename(str(i))
        )
        column_list.append(nested_column)
    df = pd.concat(column_list, axis=1)
    return df
