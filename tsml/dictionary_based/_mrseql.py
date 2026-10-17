"""Multiple Representations Sequence Learner (MrSEQL) classifier.

A Python and numba implementation of MrSEQL. The symbolic transformations and the
sequence learner are written for tsml, inspired by the mrseql package
(https://github.com/mlgig/mrseql). They follow its settings and its order of
floating point operations so that the output can be compared with it.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["MrSEQLClassifier"]

import math

import numpy as np
from numba import njit
from scipy.stats import norm
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator

# elastic net penalty of the sequence learner, REGULARISATION * (L1_RATIO * l1 +
# (1 - L1_RATIO) * 0.5 * l2)
REGULARISATION = 1.0
L1_RATIO = 0.2
MAX_ITERATIONS = 5000
CONVERGENCE_THRESHOLD = 0.005
MIN_COEFFICIENT_CHANGE = 1e-8
# the line search stops once the interval it has narrowed to is this small
SEARCH_TOLERANCE = 1e-3

# columns of the n-gram tree, a row of zeros is a node which has not been extended
SYMBOL = 0
PARENT = 1
DEPTH = 2
FIRST_CHILD = 3
N_CHILDREN = 4
FIRST_OCCURRENCE = 5
N_OCCURRENCES = 6
FIRST_CASE = 7
N_CASES = 8
# 1 + the index of the n-gram in the model, 0 if it is not in the model
MODEL = 9
# number of n-grams in the model which start with this n-gram
N_MODEL_BELOW = 10
N_COLUMNS = 11


class MrSEQLClassifier(ClassifierMixin, BaseTimeSeriesEstimator):
    """Multiple Representations Sequence Learner (MrSEQL) classifier.

    MrSEQL [1]_ transforms each series into sequences of symbolic words using SAX
    and/or SFA over sliding windows of multiple sizes. For each of these
    representations and each class, the sequence learner SEQL [2]_ fits a one-vs-rest
    linear model in the space of all n-grams of symbols found within a word, greedily
    adding the n-gram with the largest gradient of an elastic net regularised logistic
    loss.

    The n-grams selected are either used as binary features (does the series contain
    the n-gram) for a logistic regression classifier, or the SEQL models are used
    directly as an ensemble.

    This is a Python and numba implementation inspired by the mrseql package [3]_.

    Parameters
    ----------
    seql_mode : {"fs", "clf"}, default="fs"
        If "fs", SEQL is used for feature selection and a logistic regression
        classifier is trained on the selected n-grams. If "clf", the predictions of
        the SEQL models are summed.
    symrep : {"sax", "sfa"} or list of these, default="sax"
        The symbolic representations to transform the series with.
    custom_config : list of dict or None, default=None
        The symbolic representations to use, in place of those generated from
        ``symrep`` and the series length. Each dict contains "method" ("sax" or
        "sfa") and "window" (the window size), and optionally "word" (the word
        length, by default 16 for SAX and 8 for SFA) and "alphabet" (the alphabet
        size, by default 4). ``symrep`` is ignored if this is set.

    Attributes
    ----------
    classes_ : np.ndarray
        The class labels.
    n_classes_ : int
        The number of classes.
    estimator_ : LogisticRegression
        The classifier trained on the selected n-grams, only set if ``seql_mode`` is
        "fs".
    n_features_ : int
        The number of n-gram features given to ``estimator_``, only set if
        ``seql_mode`` is "fs".

    Notes
    -----
    By default a representation is made for each window size in
    ``range(16, n_timepoints, int(sqrt(n_timepoints)))``, or a single window of the
    whole series for series of length 16 or less.

    Differences from mrseql 0.0.4 [3]_:

    - The logistic regression mrseql uses for two class problems is no longer
      available in scikit-learn. ``C=2.0`` is used for two class problems, which
      gives the same model up to the tolerance of the solver.
    - mrseql writes SFA symbols as characters and its sequence learner treats "*" as
      a gap, so n-grams of more than one symbol ending in that SFA symbol are never
      selected. All n-grams are considered here.
    - With three or more classes in "fs" mode, an n-gram selected for more than one
      class is used as a single feature, not once for each class.

    References
    ----------
    .. [1] Le Nguyen, T., Gsponer, S., Ilie, I., O'Reilly, M. and Ifrim, G., 2019.
       Interpretable time series classification using linear models and
       multi-resolution multi-domain symbolic representations. Data Mining and
       Knowledge Discovery, 33(4), pp.1183-1222.
    .. [2] Ifrim, G. and Wiuf, C., 2011. Bounded coordinate-descent for biological
       sequence classification in high dimensional predictor space. In Proceedings of
       the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data
       Mining, pp.708-716.
    .. [3] https://github.com/mlgig/mrseql

    Examples
    --------
    >>> from tsml.dictionary_based import MrSEQLClassifier
    >>> from tsml.utils.testing import generate_3d_test_data
    >>> X, y = generate_3d_test_data(random_state=0)
    >>> clf = MrSEQLClassifier()
    >>> clf = clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    """

    def __init__(self, seql_mode="fs", symrep="sax", custom_config=None):
        self.seql_mode = seql_mode
        self.symrep = symrep
        self.custom_config = custom_config

        super().__init__()

    def fit(self, X: np.ndarray | list[np.ndarray], y: np.ndarray) -> object:
        """Fit the estimator to training data.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
            The training data.
        y : 1D np.ndarray of shape (n_instances)
            The class labels for fitting, indices correspond to instance indices in X

        Returns
        -------
        self :
            Reference to self.
        """
        X, y = self._validate_data(
            X=X, y=y, ensure_min_samples=2, ensure_equal_length=True
        )
        X = self._convert_X(X)

        check_classification_targets(y)

        self.n_instances_, self.n_channels_, self.n_timepoints_ = X.shape
        self.classes_, y_cls = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        if self.n_classes_ == 1:
            return self

        if self.seql_mode not in ("fs", "clf"):
            raise ValueError(f"seql_mode must be 'fs' or 'clf', got {self.seql_mode}.")
        self._config = self._get_config()

        X = np.ascontiguousarray(X, dtype=np.float64)
        self._breakpoints = []
        representations = self._transform(X, fit=True)

        # a single model is enough to separate two classes
        n_models = 1 if self.n_classes_ == 2 else self.n_classes_
        models = []
        for cls in range(n_models):
            target = np.where(y_cls == cls, 1.0, -1.0)
            models.append(
                [
                    _seql_fit(
                        words,
                        np.repeat(np.arange(X.shape[0]), np.diff(offsets)),
                        target,
                        n_symbols,
                    )
                    for words, offsets, n_symbols in representations
                ]
            )

        if self.seql_mode == "clf":
            self._models = models
            return self

        self._features = []
        for i in range(len(representations)):
            if n_models == 1:
                features, lengths, _ = models[0][i]
            else:
                # n-grams which indicate one of the classes, once each in the order
                # they are first found
                selected = [m[i][2] > 0 for m in models]
                features = np.concatenate(
                    [m[i][0][s] for m, s in zip(models, selected)]
                )
                lengths = np.concatenate([m[i][1][s] for m, s in zip(models, selected)])
                _, first = np.unique(
                    np.column_stack((lengths, features)), axis=0, return_index=True
                )
                first = np.sort(first)
                features = features[first]
                lengths = lengths[first]
            self._features.append((features, lengths))

        X_t = self._feature_matrix(representations)
        self.n_features_ = sum(len(lengths) for _, lengths in self._features)
        self.estimator_ = LogisticRegression(
            solver="newton-cg",
            class_weight="balanced",
            C=2.0 if self.n_classes_ == 2 else 1.0,
        )
        self.estimator_.fit(X_t, y)

        return self

    def predict(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Predicts labels for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances)
            Predicted class labels.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat(list(self.class_dictionary_.keys()), X.shape[0], axis=0)

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape (n_instances, n_channels, n_timepoints)
            The testing data.

        Returns
        -------
        y : array-like of shape (n_instances, n_classes_)
            Predicted probabilities using the ordering in classes_.
        """
        check_is_fitted(self)

        # treat case of single class seen in fit
        if self.n_classes_ == 1:
            return np.repeat([[1]], X.shape[0], axis=0)

        X = self._validate_data(X=X, reset=False, ensure_equal_length=True)
        X = np.ascontiguousarray(self._convert_X(X), dtype=np.float64)

        representations = self._transform(X)

        if self.seql_mode == "fs":
            return self.estimator_.predict_proba(self._feature_matrix(representations))

        scores = np.zeros((X.shape[0], self.n_classes_))
        for cls, model in enumerate(self._models):
            for (words, offsets, _), (features, lengths, coefficients) in zip(
                representations, model
            ):
                _add_scores(
                    _contains(words, offsets, features, lengths),
                    coefficients,
                    scores[:, cls],
                )
        if self.n_classes_ == 2:
            scores[:, 1] = -scores[:, 0]

        proba = 1.0 / (1.0 + np.exp(-scores))
        return proba / proba.sum(axis=1, keepdims=True)

    def _get_config(self):
        """Get the (method, window, word length, alphabet size) of each transform."""
        n_timepoints = self.n_timepoints_

        if self.custom_config is not None:
            config = []
            for item in self.custom_config:
                unknown = set(item) - {"method", "window", "word", "alphabet"}
                if unknown:
                    raise ValueError(
                        "custom_config items can only contain 'method', 'window', "
                        f"'word' and 'alphabet', got {sorted(unknown)}."
                    )
                method = item["method"]
                if method not in ("sax", "sfa"):
                    raise ValueError(
                        f"custom_config method must be 'sax' or 'sfa', got {method}."
                    )
                config.append(
                    (
                        method,
                        item["window"],
                        item.get("word", 16 if method == "sax" else 8),
                        item.get("alphabet", 4),
                    )
                )
        else:
            symrep = [self.symrep] if isinstance(self.symrep, str) else self.symrep
            if any(method not in ("sax", "sfa") for method in symrep) or not symrep:
                raise ValueError(
                    f"symrep must be 'sax', 'sfa' or a list of these, got {symrep}."
                )

            if n_timepoints > 16:
                windows = range(16, n_timepoints, int(math.sqrt(n_timepoints)))
            else:
                windows = [n_timepoints]

            config = []
            if "sax" in symrep:
                config += [("sax", window, 16, 4) for window in windows]
            if "sfa" in symrep:
                config += [("sfa", window, 8, 4) for window in windows]

        for method, window, word_length, alphabet_size in config:
            if not 1 <= window <= n_timepoints:
                raise ValueError(
                    f"window must be in [1, {n_timepoints}], the series length, got "
                    f"{window}."
                )
            if word_length < 1 or alphabet_size < 2:
                raise ValueError(
                    "word must be >= 1 and alphabet >= 2, got "
                    f"{word_length} and {alphabet_size}."
                )
            n_symbols = alphabet_size * (word_length if method == "sfa" else 1)
            if n_symbols > 256:
                raise ValueError(
                    f"The {method} transform is limited to 256 symbols, got "
                    f"{n_symbols} from word={word_length} and alphabet={alphabet_size}."
                )

        return config

    def _transform(self, X, fit=False):
        """Transform each channel with each of the symbolic transforms.

        Returns the words, the index of the first word of each case and the number
        of symbols for each representation.
        """
        representations = []
        for method, window, word_length, alphabet_size in self._config:
            # equal probability intervals of a standard normal distribution
            sax_breakpoints = norm.ppf(np.arange(1, alphabet_size) / alphabet_size)
            for channel in range(X.shape[1]):
                X_c = np.ascontiguousarray(X[:, channel, :])
                if method == "sax":
                    words, offsets = _sax_words(
                        X_c, window, word_length, sax_breakpoints
                    )
                    representations.append((words, offsets, alphabet_size))
                    if fit:
                        self._breakpoints.append(None)
                else:
                    X_norm, disjoint = _sfa_normalise(X_c, window)
                    if fit:
                        self._breakpoints.append(
                            _sfa_breakpoints(disjoint, word_length, alphabet_size)
                        )

                    # Fourier coefficients of the first window of each series
                    n_coefficients = word_length // 2 + 1
                    first = np.fft.rfft(X_norm[:, :window], axis=1)[:, :n_coefficients]
                    real = np.zeros((X.shape[0], n_coefficients))
                    imag = np.zeros((X.shape[0], n_coefficients))
                    real[:, : first.shape[1]] = first.real
                    imag[:, : first.shape[1]] = first.imag

                    words = _sfa_words(
                        X_norm,
                        real,
                        imag,
                        window,
                        word_length,
                        alphabet_size,
                        self._breakpoints[len(representations)],
                    )
                    offsets = np.arange(X.shape[0] + 1) * (X.shape[2] - window + 1)
                    representations.append(
                        (words, offsets, word_length * alphabet_size)
                    )
        return representations

    def _feature_matrix(self, representations):
        """Whether each case contains each of the selected n-grams."""
        X_t = np.hstack(
            [
                _contains(words, offsets, features, lengths)
                for (words, offsets, _), (features, lengths) in zip(
                    representations, self._features
                )
            ]
        ).astype(np.float64)
        # no n-grams were selected, only the class distribution can be learnt
        if X_t.shape[1] == 0:
            X_t = np.zeros((X_t.shape[0], 1))
        return X_t

    @classmethod
    def get_test_params(cls, parameter_set: str | None = None) -> dict | list[dict]:
        """Return unit test parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : None or str, default=None
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        return [
            {"symrep": ["sax", "sfa"]},
            {"seql_mode": "clf"},
        ]


@njit(cache=True)
def _sax_words(X, window, word_length, breakpoints):
    """Transform each window of each series into a SAX word.

    A word which is the same as the word of the previous window is removed.

    Returns
    -------
    words : 2D np.ndarray of shape (n_words, word_length)
        The words of all series, as symbols from 0 to the alphabet size - 1.
    offsets : 1D np.ndarray of shape (n_cases + 1)
        The index of the first word of each series, and the number of words.
    """
    n_cases, n_timepoints = X.shape
    n_windows = n_timepoints - window + 1
    words = np.zeros((n_cases * n_windows, word_length), dtype=np.uint8)
    offsets = np.zeros(n_cases + 1, dtype=np.int64)
    normalised = np.zeros(window)
    # number of time points in each segment of the word, which can be a fraction
    segment_length = window / word_length

    n_words = 0
    for i in range(n_cases):
        first_word = n_words
        for start in range(n_windows):
            total = 0.0
            squares = 0.0
            for j in range(start, start + window):
                total += X[i, j]
                squares += X[i, j] * X[i, j]
            mean = total / window
            variance = squares / window - mean * mean
            std = math.sqrt(variance) if variance > 0 else 1.0
            for j in range(window):
                normalised[j] = (X[i, start + j] - mean) / std

            for j in range(word_length):
                segment_start = segment_length * j
                segment_end = segment_length * (j + 1) - 1
                first = int(math.ceil(segment_start))
                last = int(math.floor(segment_end))

                # time points partly in the segment are weighted by how much is
                segment = 0.0
                if first - segment_start > 0:
                    segment += normalised[first - 1] * (first - segment_start)
                if segment_end - last > 0 and last < window - 1:
                    segment += normalised[last + 1] * (segment_end - last)
                for k in range(first, last + 1):
                    segment += normalised[k]
                segment = segment / segment_length

                symbol = 0
                for k in range(breakpoints.shape[0]):
                    if segment >= breakpoints[k]:
                        symbol += 1
                words[n_words, j] = symbol

            repeat = n_words > first_word
            if repeat:
                for j in range(word_length):
                    if words[n_words, j] != words[n_words - 1, j]:
                        repeat = False
                        break
            if not repeat:
                n_words += 1
        offsets[i + 1] = n_words

    return words[:n_words], offsets


@njit(cache=True)
def _mean_std(x):
    """Mean and standard deviation of a series, the latter 0 if it is constant."""
    n = x.shape[0]
    total = 0.0
    for i in range(n):
        total += x[i]
    mean = total / n
    squares = 0.0
    for i in range(n):
        squares += x[i] * x[i]
    variance = (1.0 / n) * squares - mean * mean
    return mean, math.sqrt(variance) if variance > 0 else 0.0


@njit(cache=True)
def _sfa_normalise(X, window):
    """Normalise series for SFA.

    Returns
    -------
    X_norm : 2D np.ndarray of shape (n_cases, n_timepoints)
        The z-normalised series.
    disjoint : 2D np.ndarray of shape (n_cases * (n_timepoints // window), window)
        The non-overlapping windows of the normalised series, each scaled to a
        standard deviation of 1.
    """
    n_cases, n_timepoints = X.shape
    n_disjoint = n_timepoints // window
    X_norm = np.zeros((n_cases, n_timepoints))
    disjoint = np.zeros((n_cases * n_disjoint, window))

    for i in range(n_cases):
        mean, std = _mean_std(X[i])
        scale = 1.0 / (std if std > 0 else 1.0)
        for j in range(n_timepoints):
            X_norm[i, j] = (X[i, j] - mean) * scale

        for w in range(n_disjoint):
            row = i * n_disjoint + w
            _, std = _mean_std(X_norm[i, w * window : (w + 1) * window])
            scale = 1.0 / (std if std > 0 else 1.0)
            for j in range(window):
                disjoint[row, j] = X_norm[i, w * window + j] * scale

    return X_norm, disjoint


def _sfa_breakpoints(disjoint, word_length, alphabet_size):
    """Find the breakpoints between the symbols of each position of an SFA word.

    The values of each position are split into bins of an equal number of windows.

    Parameters
    ----------
    disjoint : 2D np.ndarray of shape (n_windows, window)
        The normalised windows to find breakpoints from.

    Returns
    -------
    breakpoints : 2D np.ndarray of shape (word_length, alphabet_size - 1)
        The ascending breakpoints for each position of the word, infinite where
        there are too few distinct values.
    """
    n_windows, window = disjoint.shape
    n_coefficients = word_length // 2 + 1
    fourier = np.fft.rfft(disjoint, axis=1)[:, :n_coefficients]
    real = np.zeros((n_windows, n_coefficients))
    imag = np.zeros((n_windows, n_coefficients))
    real[:, : fourier.shape[1]] = fourier.real
    imag[:, : fourier.shape[1]] = fourier.imag

    scale = 1.0 / math.sqrt(window)
    values = np.zeros((n_windows, word_length))
    for i in range(word_length):
        coefficient = (i + 1) // 2
        if i == 0 or i % 2 == 1:
            values[:, i] = real[:, coefficient] * scale
        else:
            values[:, i] = -imag[:, coefficient] * scale

    # the breakpoints are found for values rounded to two decimal places
    values = values * 100.0
    values = np.trunc(values + np.copysign(0.5, values)) * 0.01

    breakpoints = np.full((word_length, alphabet_size - 1), np.inf)
    bin_size = n_windows / alphabet_size
    for i in range(word_length):
        distinct, counts = np.unique(values[:, i], return_counts=True)
        n_found = 0
        for value, count in zip(distinct, np.cumsum(counts)):
            if n_found < alphabet_size - 1 and count > math.ceil(
                bin_size * (n_found + 1)
            ):
                breakpoints[i, n_found] = value
                n_found += 1

    return breakpoints


@njit(cache=True)
def _sfa_words(X, real, imag, window, word_length, alphabet_size, breakpoints):
    """Transform each window of each series into an SFA word.

    The Fourier coefficients of a window are found by updating those of the
    previous window. The values of a word are the real part of the first coefficient
    followed by the real and imaginary parts of the following coefficients.

    Parameters
    ----------
    X : 2D np.ndarray of shape (n_cases, n_timepoints)
        The z-normalised series.
    real : 2D np.ndarray of shape (n_cases, word_length // 2 + 1)
        The real part of the Fourier coefficients of the first window of each
        series.
    imag : 2D np.ndarray of shape (n_cases, word_length // 2 + 1)
        The imaginary part of the Fourier coefficients of the first window of each
        series.

    Returns
    -------
    words : 2D np.ndarray of shape (n_cases * n_windows, word_length)
        The words of all series. Each position of the word has its own alphabet,
        the symbols are position * alphabet_size + the symbol of the value.
    """
    n_cases, n_timepoints = X.shape
    n_windows = n_timepoints - window + 1
    n_coefficients = real.shape[1]
    words = np.zeros((n_cases * n_windows, word_length), dtype=np.uint8)

    # rotation applied to each coefficient when the window moves by one time point
    cos = np.zeros(n_coefficients)
    sin = np.zeros(n_coefficients)
    for k in range(n_coefficients):
        angle = 2 * math.pi * k / window
        cos[k] = math.cos(angle)
        sin[k] = math.sin(angle)

    inverse_window = 1.0 / window
    scale = 1.0 / math.sqrt(window)
    re = np.zeros(n_coefficients)
    im = np.zeros(n_coefficients)
    for i in range(n_cases):
        re[:] = real[i]
        im[:] = imag[i]
        total = 0.0
        squares = 0.0
        for j in range(window):
            total += X[i, j]
            squares += X[i, j] * X[i, j]

        for t in range(n_windows):
            if t > 0:
                added = X[i, t + window - 1]
                removed = X[i, t - 1]
                total += added - removed
                squares += added * added - removed * removed
                for k in range(n_coefficients):
                    moved = re[k] + added - removed
                    rotated = moved * cos[k] - im[k] * sin[k]
                    im[k] = moved * sin[k] + cos[k] * im[k]
                    re[k] = rotated

            # scale by the standard deviation of the window
            mean = total * inverse_window
            variance = squares * inverse_window - mean * mean
            window_scale = scale * (1.0 / math.sqrt(variance) if variance > 0 else 1.0)

            row = i * n_windows + t
            for j in range(word_length):
                k = (j + 1) // 2
                if j == 0 or j % 2 == 1:
                    value = re[k] * window_scale
                else:
                    value = im[k] * -window_scale

                symbol = 0
                while symbol < alphabet_size - 1 and value >= breakpoints[j, symbol]:
                    symbol += 1
                words[row, j] = j * alphabet_size + symbol

    return words


@njit(cache=True)
def _resize(array, n_rows):
    """Copy a 1D or 2D array into a larger array of zeros."""
    resized = np.zeros((n_rows,) + array.shape[1:], dtype=array.dtype)
    resized[: array.shape[0]] = array
    return resized


@njit(cache=True)
def _extend(node, words, word_case, n_symbols, nodes, occurrences, cases, sizes):
    """Add a node to the n-gram tree for each symbol which follows an n-gram.

    Nodes are added in order of their symbol, with the positions of the last symbol
    of the n-gram in the words and the cases which contain it.

    Parameters
    ----------
    sizes : 1D np.ndarray of shape (3)
        The number of rows of nodes, occurrences and cases in use, updated in place.

    Returns
    -------
    nodes, occurrences, cases : np.ndarray
        The arrays, which are replaced if more space was required.
    """
    n_words, word_length = words.shape
    start = nodes[node, FIRST_OCCURRENCE]
    end = start + nodes[node, N_OCCURRENCES]

    # the root of the tree is followed by all symbols
    counts = np.zeros(n_symbols, dtype=np.int64)
    if node == 0:
        for w in range(n_words):
            for p in range(word_length):
                counts[words[w, p]] += 1
    else:
        for i in range(start, end):
            p = occurrences[i, 1] + 1
            if p < word_length:
                counts[words[occurrences[i, 0], p]] += 1

    n_nodes = sizes[0]
    n_occurrences = sizes[1]
    n_case_entries = sizes[2]
    n_new = np.count_nonzero(counts)
    n_new_occurrences = counts.sum()
    if n_nodes + n_new > nodes.shape[0]:
        nodes = _resize(nodes, max(2 * nodes.shape[0], n_nodes + n_new))
    if n_occurrences + n_new_occurrences > occurrences.shape[0]:
        occurrences = _resize(
            occurrences,
            max(2 * occurrences.shape[0], n_occurrences + n_new_occurrences),
        )
    if n_case_entries + n_new_occurrences > cases.shape[0]:
        cases = _resize(
            cases, max(2 * cases.shape[0], n_case_entries + n_new_occurrences)
        )

    nodes[node, FIRST_CHILD] = n_nodes
    nodes[node, N_CHILDREN] = n_new
    child_of_symbol = np.zeros(n_symbols, dtype=np.int64)
    for symbol in range(n_symbols):
        if counts[symbol] > 0:
            nodes[n_nodes, SYMBOL] = symbol
            nodes[n_nodes, PARENT] = node
            nodes[n_nodes, DEPTH] = nodes[node, DEPTH] + 1
            nodes[n_nodes, FIRST_OCCURRENCE] = n_occurrences
            child_of_symbol[symbol] = n_nodes
            n_occurrences += counts[symbol]
            n_nodes += 1

    if node == 0:
        for w in range(n_words):
            for p in range(word_length):
                child = child_of_symbol[words[w, p]]
                i = nodes[child, FIRST_OCCURRENCE] + nodes[child, N_OCCURRENCES]
                occurrences[i, 0] = w
                occurrences[i, 1] = p
                nodes[child, N_OCCURRENCES] += 1
    else:
        for o in range(start, end):
            w = occurrences[o, 0]
            p = occurrences[o, 1] + 1
            if p < word_length:
                child = child_of_symbol[words[w, p]]
                i = nodes[child, FIRST_OCCURRENCE] + nodes[child, N_OCCURRENCES]
                occurrences[i, 0] = w
                occurrences[i, 1] = p
                nodes[child, N_OCCURRENCES] += 1

    # words are ordered by case, so the occurrences of each node are too
    for child in range(sizes[0], n_nodes):
        nodes[child, FIRST_CASE] = n_case_entries
        previous = -1
        first = nodes[child, FIRST_OCCURRENCE]
        for i in range(first, first + nodes[child, N_OCCURRENCES]):
            case = word_case[occurrences[i, 0]]
            if case != previous:
                cases[n_case_entries] = case
                n_case_entries += 1
                previous = case
        nodes[child, N_CASES] = n_case_entries - nodes[child, FIRST_CASE]

    sizes[0] = n_nodes
    sizes[1] = n_occurrences
    sizes[2] = n_case_entries
    return nodes, occurrences, cases


@njit(cache=True)
def _penalty(change, coefficient, in_model, total_abs, total_squares):
    """Elastic net penalty after changing the coefficient of an n-gram.

    Parameters
    ----------
    change : float
        The change to the coefficient of the n-gram.
    coefficient : float
        The current coefficient of the n-gram.
    in_model : bool
        Whether the n-gram is in the model.
    total_abs : float
        The sum of the absolute coefficients of the model.
    total_squares : float
        The sum of the squared coefficients of the model.
    """
    if total_squares == 0:
        absolute = abs(change)
        squares = change * change
    elif not in_model:
        absolute = total_abs + abs(change)
        squares = total_squares + change * change
    else:
        new = coefficient + change
        absolute = total_abs - abs(coefficient) + abs(new)
        squares = total_squares - coefficient * coefficient + new * new
    return REGULARISATION * (L1_RATIO * absolute + (1 - L1_RATIO) * 0.5 * squares)


@njit(cache=True)
def _total_loss(values, support, losses, y):
    """Logistic loss of all cases when the scores of some of them are changed.

    Parameters
    ----------
    values : 1D np.ndarray of shape (n_support)
        The scores of the cases in support.
    support : 1D np.ndarray of shape (n_support)
        The ascending indices of the cases whose score is changed.
    losses : 1D np.ndarray of shape (n_cases)
        The loss of each case with its current score.
    """
    total = 0.0
    s = 0
    for i in range(losses.shape[0]):
        if s < support.shape[0] and support[s] == i:
            total += math.log(1 + math.exp(-y[i] * values[s]))
            s += 1
        else:
            total += losses[i]
    return total


@njit(cache=True)
def _line_search(
    gradient,
    support,
    scores,
    losses,
    y,
    objective,
    coefficient,
    in_model,
    total_abs,
    total_squares,
):
    """Find the change to the coefficient of an n-gram which minimises the loss.

    The step along the gradient is doubled until the loss stops decreasing, then
    the interval the minimum is in is repeatedly reduced.

    Parameters
    ----------
    gradient : float
        The gradient of the loss for the coefficient of the n-gram.
    support : 1D np.ndarray of shape (n_support)
        The ascending indices of the cases which contain the n-gram.
    scores : 1D np.ndarray of shape (n_cases)
        The current score of the model for each case.
    losses : 1D np.ndarray of shape (n_cases)
        The current loss of each case.
    objective : float
        The current loss and penalty of the model.

    Returns
    -------
    values : 1D np.ndarray of shape (n_support)
        The new scores of the cases in support.
    """
    n_support = support.shape[0]
    low = np.zeros(n_support)
    mid = np.zeros(n_support)
    high = np.zeros(n_support)
    for i in range(n_support):
        low[i] = mid[i] = high[i] = scores[support[i]]
    initial = scores[support[0]]

    magnitude = math.ceil(math.log10(abs(gradient)))
    rate = min(1e-3, math.pow(10.0, -magnitude))
    change = 0.0
    previous = objective
    while True:
        step = rate * gradient
        change -= step
        for i in range(n_support):
            low[i] = mid[i]
            mid[i] = high[i]
            high[i] = mid[i] - step
        current = _total_loss(high, support, losses, y) + _penalty(
            change, coefficient, in_model, total_abs, total_squares
        )
        rate *= 2.0
        if not current < previous:
            break
        previous = current

    while True:
        width = 0.0
        for i in range(n_support):
            width += abs(high[i] - low[i])
        if not width > SEARCH_TOLERANCE:
            break

        lower = (low + mid) / 2
        upper = (mid + high) / 2
        loss_mid = _total_loss(mid, support, losses, y) + _penalty(
            mid[0] - initial, coefficient, in_model, total_abs, total_squares
        )
        loss_lower = _total_loss(lower, support, losses, y) + _penalty(
            lower[0] - initial, coefficient, in_model, total_abs, total_squares
        )
        loss_upper = _total_loss(upper, support, losses, y) + _penalty(
            upper[0] - initial, coefficient, in_model, total_abs, total_squares
        )

        if loss_mid <= loss_lower and loss_mid <= loss_upper:
            low = lower
            high = upper
        elif loss_lower <= loss_mid and loss_lower <= loss_upper:
            high = mid
            mid = lower
        else:
            low = mid
            mid = upper

    return mid


@njit(cache=True)
def _seql_fit(words, word_case, y, n_symbols):
    """Fit a linear model on the n-grams of symbols found in the words of each case.

    Each iteration finds the n-gram with the largest gradient of the logistic loss
    with an elastic net penalty, and changes its coefficient to minimise the loss.
    The n-grams are searched breadth first in a tree where the children of an
    n-gram extend it by one symbol. As cases containing an n-gram also contain the
    n-gram it extends, the gradient of all n-grams below a node is bounded and the
    search does not continue from nodes which cannot improve on the best found.

    Parameters
    ----------
    words : 2D np.ndarray of shape (n_words, word_length)
        The words of all cases, ordered by case.
    word_case : 1D np.ndarray of shape (n_words)
        The index of the case each word is from.
    y : 1D np.ndarray of shape (n_cases)
        The target, 1 or -1.
    n_symbols : int
        The number of different symbols.

    Returns
    -------
    features : 2D np.ndarray of shape (n_features, word_length)
        The symbols of each n-gram of the model, in lexicographic order.
    lengths : 1D np.ndarray of shape (n_features)
        The length of each n-gram.
    coefficients : 1D np.ndarray of shape (n_features)
        The coefficient of each n-gram, scaled so their absolute values would sum to
        2 if no coefficient was changed more than once.
    """
    n_cases = y.shape[0]
    word_length = words.shape[1]

    # node 0 is the root of the tree, the empty n-gram
    nodes = np.zeros((1024, N_COLUMNS), dtype=np.int64)
    occurrences = np.zeros((max(1024, 2 * words.size), 2), dtype=np.int64)
    cases = np.zeros(max(1024, 2 * words.size), dtype=np.int64)
    sizes = np.array([1, 0, 0])

    model_node = np.zeros(64, dtype=np.int64)
    # coefficient of the n-gram and the gradient of its penalty
    model_coefficient = np.zeros(64)
    model_penalty_gradient = np.zeros(64)
    # sum of the accepted changes to the coefficient
    model_weight = np.zeros(64)
    model_accepted = np.zeros(64, dtype=np.bool_)
    n_model = 0
    total_abs = 0.0
    total_squares = 0.0
    total_change = 0.0

    scores = np.zeros(n_cases)
    # derivative of the loss of each case by y * score, negated
    fractions = np.full(n_cases, 0.5)
    losses = np.zeros(n_cases)
    objective = 0.0
    for i in range(n_cases):
        losses[i] = math.log(1 + math.exp(-y[i] * scores[i]))
        objective += losses[i]

    level = np.zeros(1024, dtype=np.int64)
    next_level = np.zeros(1024, dtype=np.int64)
    for _ in range(MAX_ITERATIONS):
        # find the n-gram with the largest gradient
        best = 0
        best_gradient = 0.0
        largest = 0.0
        level[0] = 0
        n_level = 1
        while n_level > 0:
            n_next = 0
            for a in range(n_level):
                node = level[a]
                if nodes[node, FIRST_CHILD] == 0:
                    nodes, occurrences, cases = _extend(
                        node,
                        words,
                        word_case,
                        n_symbols,
                        nodes,
                        occurrences,
                        cases,
                        sizes,
                    )

                first = nodes[node, FIRST_CHILD]
                for child in range(first, first + nodes[node, N_CHILDREN]):
                    gradient = 0.0
                    positive = 0.0
                    negative = 0.0
                    first_case = nodes[child, FIRST_CASE]
                    for i in range(first_case, first_case + nodes[child, N_CASES]):
                        case = cases[i]
                        v = y[case] * fractions[case]
                        gradient -= v
                        if y[case] > 0:
                            positive -= v
                        else:
                            negative -= v

                    # the gradient of an n-gram extending this one is between
                    # the gradient from just the positive or negative cases
                    if nodes[child, N_MODEL_BELOW] > 0:
                        if nodes[child, MODEL] > 0:
                            gradient += model_penalty_gradient[nodes[child, MODEL] - 1]

                        keep = False
                        for m in range(n_model):
                            other = model_node[m]
                            while nodes[other, DEPTH] > nodes[child, DEPTH]:
                                other = nodes[other, PARENT]
                            if (
                                other == child
                                and max(
                                    abs(positive + model_penalty_gradient[m]),
                                    abs(negative + model_penalty_gradient[m]),
                                )
                                > largest
                            ):
                                keep = True
                                break
                    else:
                        keep = max(abs(positive), abs(negative)) > largest

                    if keep and abs(gradient) > largest:
                        largest = abs(gradient)
                        best = child
                        best_gradient = gradient

                    if keep:
                        if n_next == next_level.shape[0]:
                            next_level = _resize(next_level, 2 * n_next)
                        next_level[n_next] = child
                        n_next += 1

            level, next_level = next_level, level
            n_level = n_next

        if best == 0:
            break

        first_case = nodes[best, FIRST_CASE]
        support = cases[first_case : first_case + nodes[best, N_CASES]]
        m = nodes[best, MODEL] - 1
        values = _line_search(
            best_gradient,
            support,
            scores,
            losses,
            y,
            objective,
            model_coefficient[m] if m >= 0 else 0.0,
            m >= 0,
            total_abs,
            total_squares,
        )
        change = values[0] - scores[support[0]]

        if m < 0:
            if n_model == model_node.shape[0]:
                model_node = _resize(model_node, 2 * n_model)
                model_coefficient = _resize(model_coefficient, 2 * n_model)
                model_penalty_gradient = _resize(model_penalty_gradient, 2 * n_model)
                model_weight = _resize(model_weight, 2 * n_model)
                model_accepted = _resize(model_accepted, 2 * n_model)
            m = n_model
            n_model += 1
            model_node[m] = best
            model_coefficient[m] = change
            nodes[best, MODEL] = m + 1
            node = best
            while node > 0:
                nodes[node, N_MODEL_BELOW] += 1
                node = nodes[node, PARENT]
        else:
            total_squares -= model_coefficient[m] * model_coefficient[m]
            total_abs -= abs(model_coefficient[m])
            model_coefficient[m] += change
        total_squares += model_coefficient[m] * model_coefficient[m]
        total_abs += abs(model_coefficient[m])
        model_penalty_gradient[m] = REGULARISATION * (
            L1_RATIO * (1 if model_coefficient[m] > 0 else -1)
            + (1 - L1_RATIO) * model_coefficient[m]
        )

        previous = objective
        objective = 0.0
        total_moved = 0.0
        total_score = 0.0
        s = 0
        for i in range(n_cases):
            value = scores[i]
            if s < support.shape[0] and support[s] == i:
                value = values[s]
                s += 1
                fractions[i] = 1 / (1 + math.exp(y[i] * value))
                losses[i] = math.log(1 + math.exp(-y[i] * value))
                total_moved += abs(value - scores[i])
            objective += losses[i]
            total_score += abs(value)
        objective = objective + REGULARISATION * (
            L1_RATIO * total_abs + (1 - L1_RATIO) * 0.5 * total_squares
        )

        if previous - objective == 0 or not abs(change) > MIN_COEFFICIENT_CHANGE:
            break
        total_change += abs(change)
        model_weight[m] += change
        model_accepted[m] = True
        if total_moved / (1 + total_score) <= CONVERGENCE_THRESHOLD:
            break

        for i in range(support.shape[0]):
            scores[support[i]] = values[i]

    # visit the tree depth first to return the n-grams in lexicographic order
    n_features = np.count_nonzero(model_accepted[:n_model])
    features = np.zeros((n_features, word_length), dtype=np.uint8)
    lengths = np.zeros(n_features, dtype=np.int64)
    coefficients = np.zeros(n_features)
    stack = np.zeros(sizes[0], dtype=np.int64)
    n_stack = 1
    n_found = 0
    while n_stack > 0:
        n_stack -= 1
        node = stack[n_stack]

        m = nodes[node, MODEL] - 1
        if m >= 0 and model_accepted[m]:
            lengths[n_found] = nodes[node, DEPTH]
            coefficients[n_found] = model_weight[m] * 2 / total_change
            other = node
            while other > 0:
                features[n_found, nodes[other, DEPTH] - 1] = nodes[other, SYMBOL]
                other = nodes[other, PARENT]
            n_found += 1

        if nodes[node, N_MODEL_BELOW] > 0 or node == 0:
            first = nodes[node, FIRST_CHILD]
            for child in range(first + nodes[node, N_CHILDREN] - 1, first - 1, -1):
                stack[n_stack] = child
                n_stack += 1

    return features, lengths, coefficients


@njit(cache=True)
def _contains(words, offsets, features, lengths):
    """Find whether any word of each case contains each n-gram.

    Parameters
    ----------
    words : 2D np.ndarray of shape (n_words, word_length)
        The words of all cases.
    offsets : 1D np.ndarray of shape (n_cases + 1)
        The index of the first word of each case, and the number of words.
    features : 2D np.ndarray of shape (n_features, word_length)
        The symbols of each n-gram.
    lengths : 1D np.ndarray of shape (n_features)
        The length of each n-gram.

    Returns
    -------
    contains : 2D np.ndarray of shape (n_cases, n_features)
    """
    n_cases = offsets.shape[0] - 1
    n_features = features.shape[0]
    word_length = words.shape[1]
    contains = np.zeros((n_cases, n_features), dtype=np.bool_)

    for i in range(n_cases):
        for f in range(n_features):
            length = lengths[f]
            found = False
            for w in range(offsets[i], offsets[i + 1]):
                for start in range(word_length - length + 1):
                    j = 0
                    while j < length and words[w, start + j] == features[f, j]:
                        j += 1
                    if j == length:
                        found = True
                        break
                if found:
                    break
            contains[i, f] = found

    return contains


@njit(cache=True)
def _add_scores(contains, coefficients, scores):
    """Add the coefficients of the n-grams each case contains to its score."""
    for i in range(contains.shape[0]):
        for f in range(contains.shape[1]):
            if contains[i, f]:
                scores[i] += coefficients[f]
