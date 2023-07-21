"""Shapelet transformers.

A transformer from the time domain into the shapelet domain.
"""

__author__ = ["MatthewMiddlehurst", "baraline"]
__all__ = ["RandomShapeletTransformer", "RandomDilatedShapeletTransformer"]

import heapq
import math
import time
import warnings

import numpy as np
from joblib import Parallel
from numba import njit, prange, set_num_threads
from numba.typed.typedlist import List
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import check_is_fitted

from tsml.base import BaseTimeSeriesEstimator
from tsml.distances import manhattan_distance
from tsml.utils.numba_functions.general import (
    choice_log,
    combinations_1d,
    get_subsequence,
    get_subsequence_with_mean_std,
    set_numba_random_seed,
    sliding_mean_std_one_series,
    z_normalise_series,
)
from tsml.utils.numba_functions.stats import prime_up_to
from tsml.utils.validation import check_n_jobs


class RandomShapeletTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Random Shapelet Transform.

    Implementation of the binary shapelet transform along the lines of [1]_[2]_, with
    randomly extracted shapelets.

    Overview: Input "n" series with "d" dimensions of length "m". Continuously extract
    candidate shapelets and filter them in batches.
        For each candidate shapelet
            - Extract a shapelet from an instance with random length, position and
              dimension
            - Using its distance to train cases, calculate the shapelets information
              gain
            - Abandon evaluating the shapelet if it is impossible to obtain a higher
              information gain than the current worst
        For each shapelet batch
            - Add each candidate to its classes shapelet heap, removing the lowest
              information gain shapelet if the max number of shapelets has been met
            - Remove self-similar shapelets from the heap
    Using the final set of filtered shapelets, transform the data into a vector of
    of distances from a series to each shapelet.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to <= max_shapelets, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to n_classes / max_shapelets. If None uses the min between
        10 * n_instances and 1000
    min_shapelet_length : int, default=3
        Lower bound on candidate shapelet lengths.
    max_shapelet_length : int or None, default= None
        Upper bound on candidate shapelet lengths. If None no max length is used.
    remove_self_similar : boolean, default=True
        Remove overlapping "self-similar" shapelets when merging candidate shapelets.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_shapelet_samples.
        Default of 0 means n_shapelet_samples is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when time_limit_in_minutes is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes_ : int
        The number of classes.
    n_instances_ : int
        The number of train cases.
    n_dims_ : int
        The number of dimensions per case.
    series_length_ : int
        The length of each series.
    classes_ : list
        The classes labels.
    shapelets_ : list
        The stored shapelets and relating information after a dataset has been
        processed.
        Each item in the list is a tuple containing the following 7 items:
        (shapelet information gain, shapelet length, start position the shapelet was
        extracted from, shapelet dimension, index of the instance the shapelet was
        extracted from in fit, class value of the shapelet, The z-normalised shapelet
        array)

    See Also
    --------
    ShapeletTransformClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/transformers/ShapeletTransform.java>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge Centered
       Systems, 32, 2017.

    Examples
    --------
    >>> from tsml.transformations._shapelet_transform import RandomShapeletTransformer
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X_train, y_train = load_minimal_chinatown(split="train")
    >>> t = RandomShapeletTransformer(
    ...     n_shapelet_samples=500,
    ...     max_shapelets=10,
    ...     batch_size=100,
    ... )
    >>> t.fit(X_train, y_train)
    RandomShapeletTransformer(...)
    >>> X_t = t.transform(X_train)
    """

    def __init__(
        self,
        n_shapelet_samples=10000,
        max_shapelets=None,
        min_shapelet_length=3,
        max_shapelet_length=None,
        remove_self_similar=True,
        time_limit_in_minutes=0.0,
        contract_max_n_shapelet_samples=np.inf,
        n_jobs=1,
        parallel_backend=None,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.remove_self_similar = remove_self_similar

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.batch_size = batch_size
        self.random_state = random_state

        super(RandomShapeletTransformer, self).__init__()

    def fit(self, X, y=None):
        """Fit the shapelet transform to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : RandomShapeletTransformer
            This estimator.
        """
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        self.n_instances_, self.n_dims_, self.series_length_ = X.shape
        self.classes_, self._class_counts = np.unique(y, return_counts=True)
        self.n_classes_ = self.classes_.shape[0]
        self.class_dictionary_ = {}
        for index, class_val in enumerate(self.classes_):
            self.class_dictionary_[class_val] = index

        self._n_jobs = check_n_jobs(self.n_jobs)

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        self._max_shapelets = self.max_shapelets
        if self.max_shapelets is None:
            self._max_shapelets = min(10 * self.n_instances_, 1000)
        if self._max_shapelets < self.n_classes_:
            self._max_shapelets = self.n_classes_

        self._max_shapelet_length = self.max_shapelet_length
        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.series_length_

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        fit_time = 0

        max_shapelets_per_class = int(self._max_shapelets / self.n_classes_)
        if max_shapelets_per_class < 1:
            max_shapelets_per_class = 1

        shapelets = List(
            [List([(-1.0, -1, -1, -1, -1, -1)]) for _ in range(self.n_classes_)]
        )
        n_shapelets_extracted = 0

        if time_limit > 0:
            while (
                fit_time < time_limit
                and n_shapelets_extracted < self.contract_max_n_shapelet_samples
            ):
                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                    )
                    for i in range(self.batch_size)
                )

                for i, heap in enumerate(shapelets):
                    self._merge_shapelets(
                        heap,
                        List(candidate_shapelets),
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = self._remove_self_similar_shapelets(heap)
                        shapelets[i] = List([n for (n, b) in zip(heap, to_keep) if b])

                n_shapelets_extracted += self.batch_size
                fit_time = time.time() - start_time
        else:
            while n_shapelets_extracted < self.n_shapelet_samples:
                n_shapelets_to_extract = (
                    self.batch_size
                    if n_shapelets_extracted + self.batch_size
                    <= self.n_shapelet_samples
                    else self.n_shapelet_samples - n_shapelets_extracted
                )

                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                    )
                    for i in range(n_shapelets_to_extract)
                )

                for i, heap in enumerate(shapelets):
                    self._merge_shapelets(
                        heap,
                        List(candidate_shapelets),
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = self._remove_self_similar_shapelets(heap)
                        shapelets[i] = List([n for (n, b) in zip(heap, to_keep) if b])

                n_shapelets_extracted += n_shapelets_to_extract

        self.shapelets_ = [
            (
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                self.classes_[s[5]],
                z_normalise_series(X[s[4], s[3], s[2] : s[2] + s[1]]),
            )
            for class_shapelets in shapelets
            for s in class_shapelets
            if s[0] > 0
        ]
        self.shapelets_.sort(reverse=True, key=lambda s: (s[0], s[1], s[2], s[3], s[4]))

        to_keep = self._remove_identical_shapelets(List(self.shapelets_))
        self.shapelets_ = [n for (n, b) in zip(self.shapelets_, to_keep) if b]

        self._sorted_indicies = []
        for s in self.shapelets_:
            sabs = np.abs(s[6])
            self._sorted_indicies.append(
                np.array(
                    sorted(range(s[1]), reverse=True, key=lambda j, sabs=sabs: sabs[j])
                )
            )

        return self

    def transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            The transformed dataframe in tabular format.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        output = np.zeros((len(X), len(self.shapelets_)))

        for i, series in enumerate(X):
            dists = Parallel(
                n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
            )(
                delayed(_online_shapelet_distance)(
                    series[shapelet[3]],
                    shapelet[6],
                    self._sorted_indicies[n],
                    shapelet[2],
                    shapelet[1],
                )
                for n, shapelet in enumerate(self.shapelets_)
            )

            output[i] = dists

        return output

    def _extract_random_shapelet(self, X, y, i, shapelets, max_shapelets_per_class):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (i + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        inst_idx = i % self.n_instances_
        cls_idx = int(y[inst_idx])
        worst_quality = (
            shapelets[cls_idx][0][0]
            if len(shapelets[cls_idx]) == max_shapelets_per_class
            else -1
        )

        length = (
            rng.randint(0, self._max_shapelet_length - self.min_shapelet_length)
            + self.min_shapelet_length
        )
        position = rng.randint(0, self.series_length_ - length)
        dim = rng.randint(0, self.n_dims_)

        shapelet = z_normalise_series(X[inst_idx, dim, position : position + length])
        sabs = np.abs(shapelet)
        sorted_indicies = np.array(
            sorted(range(length), reverse=True, key=lambda j: sabs[j])
        )

        quality = self._find_shapelet_quality(
            X,
            y,
            shapelet,
            sorted_indicies,
            position,
            length,
            dim,
            inst_idx,
            self._class_counts[cls_idx],
            self.n_instances_ - self._class_counts[cls_idx],
            worst_quality,
        )

        return quality, length, position, dim, inst_idx, cls_idx

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _find_shapelet_quality(
        X,
        y,
        shapelet,
        sorted_indicies,
        position,
        length,
        dim,
        inst_idx,
        this_cls_count,
        other_cls_count,
        worst_quality,
    ):
        # todo optimise this more, we spend 99% of time here
        orderline = []
        this_cls_traversed = 0
        other_cls_traversed = 0

        for i, series in enumerate(X):
            if i != inst_idx:
                distance = _online_shapelet_distance(
                    series[dim], shapelet, sorted_indicies, position, length
                )
            else:
                distance = 0

            if y[i] == y[inst_idx]:
                cls = 1
                this_cls_traversed += 1
            else:
                cls = -1
                other_cls_traversed += 1

            orderline.append((distance, cls))
            orderline.sort()

            if worst_quality > 0:
                quality = _calc_early_binary_ig(
                    orderline,
                    this_cls_traversed,
                    other_cls_traversed,
                    this_cls_count - this_cls_traversed,
                    other_cls_count - other_cls_traversed,
                    worst_quality,
                )

                if quality <= worst_quality:
                    return -1

        quality = _calc_binary_ig(orderline, this_cls_count, other_cls_count)

        return round(quality, 12)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _merge_shapelets(
        shapelet_heap, candidate_shapelets, max_shapelets_per_class, cls_idx
    ):
        for shapelet in candidate_shapelets:
            if shapelet[5] == cls_idx and shapelet[0] > 0:
                if (
                    len(shapelet_heap) == max_shapelets_per_class
                    and shapelet[0] < shapelet_heap[0][0]
                ):
                    continue

                heapq.heappush(shapelet_heap, shapelet)

                if len(shapelet_heap) > max_shapelets_per_class:
                    heapq.heappop(shapelet_heap)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_self_similar_shapelets(shapelet_heap):
        to_keep = [True] * len(shapelet_heap)

        for i in range(len(shapelet_heap)):
            if to_keep[i] is False:
                continue

            for n in range(i + 1, len(shapelet_heap)):
                if to_keep[n] and _is_self_similar(shapelet_heap[i], shapelet_heap[n]):
                    if (shapelet_heap[i][0], shapelet_heap[i][1]) >= (
                        shapelet_heap[n][0],
                        shapelet_heap[n][1],
                    ):
                        to_keep[n] = False
                    else:
                        to_keep[i] = False
                        break

        return to_keep

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_identical_shapelets(shapelets):
        to_keep = [True] * len(shapelets)

        for i in range(len(shapelets)):
            for n in range(i + 1, len(shapelets)):
                if (
                    to_keep[n]
                    and shapelets[i][1] == shapelets[n][1]
                    and np.array_equal(shapelets[i][6], shapelets[n][6])
                ):
                    to_keep[n] = False

        return to_keep

    def _more_tags(self) -> dict:
        return {"requires_y": True}

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
        return {"max_shapelets": 5, "n_shapelet_samples": 50, "batch_size": 20}


@njit(fastmath=True, cache=True)
def _online_shapelet_distance(series, shapelet, sorted_indicies, position, length):
    subseq = series[position : position + length]

    sum = 0.0
    sum2 = 0.0
    for i in subseq:
        sum += i
        sum2 += i * i

    mean = sum / length
    std = (sum2 - mean * mean * length) / length
    if std > 0:
        subseq = (subseq - mean) / std
    else:
        subseq = np.zeros(length)

    best_dist = 0
    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [sum, sum]
    sums2 = [sum2, sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            start = series[pos - n]
            end = series[pos - n + length]

            sums[n] += mod * end - mod * start
            sums2[n] += mod * end * end - mod * start * start

            mean = sums[n] / length
            std = math.sqrt((sums2[n] - mean * mean * length) / length)

            dist = 0
            use_std = std != 0
            for j in range(length):
                val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist if best_dist == 0 else 1 / length * best_dist


@njit(fastmath=True, cache=True)
def _calc_early_binary_ig(
    orderline,
    c1_traversed,
    c2_traversed,
    c1_to_add,
    c2_to_add,
    worst_quality,
):
    initial_ent = _binary_entropy(
        c1_traversed + c1_to_add,
        c2_traversed + c2_to_add,
    )

    total_all = c1_traversed + c2_traversed + c1_to_add + c2_to_add

    bsf_ig = 0
    # actual observations in orderline
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        # optimistically add this class to left side first and other to right
        left_prop = (split + 1 + c1_to_add) / total_all
        ent_left = _binary_entropy(c1_count + c1_to_add, c2_count)

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

        ent_right = _binary_entropy(
            c1_traversed - c1_count,
            c2_traversed - c2_count + c2_to_add,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        # now optimistically add this class to right, other to left
        left_prop = (split + 1 + c2_to_add) / total_all
        ent_left = _binary_entropy(c1_count, c2_count + c2_to_add)

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

        ent_right = _binary_entropy(
            c1_traversed - c1_count + c1_to_add,
            c2_traversed - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        if bsf_ig > worst_quality:
            return bsf_ig

    return bsf_ig


@njit(fastmath=True, cache=True)
def _calc_binary_ig(orderline, c1, c2):
    initial_ent = _binary_entropy(c1, c2)

    total_all = c1 + c2

    bsf_ig = 0
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        left_prop = (split + 1) / total_all
        ent_left = _binary_entropy(c1_count, c2_count)

        right_prop = 1 - left_prop
        ent_right = _binary_entropy(
            c1 - c1_count,
            c2 - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig


@njit(fastmath=True, cache=True)
def _binary_entropy(c1, c2):
    ent = 0
    if c1 != 0:
        ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
    if c2 != 0:
        ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
    return ent


@njit(fastmath=True, cache=True)
def _is_self_similar(s1, s2):
    # not self similar if from different series or dimension
    if s1[4] == s2[4] and s1[3] == s2[3]:
        if s2[2] <= s1[2] <= s2[2] + s2[1]:
            return True
        if s1[2] <= s2[2] <= s1[2] + s1[1]:
            return True

    return False


class RandomDilatedShapeletTransformer(TransformerMixin, BaseTimeSeriesEstimator):
    """Random Dilated Shapelet Transform (RDST) as described in [1]_[2]_.

    Overview: The input is n series with d channels of length m. First step is to
    extract candidate shapelets from the inputs. This is done randomly, and for
    each candidate shapelet:
        - Length is randomly selected from shapelet_lengths parameter
        - Dilation is sampled as a function the shapelet length and time series length
        - Normalization is chosen randomly given the probability given as parameter
        - Value is sampled randomly from an input time series given the length and
        dilation parameter.
        - Threshold is randomly chosen between two percentiles of the distribution
        of the distance vector between the shapelet and another time series. This time
        serie is drawn from the same class if classes are given during fit. Otherwise,
        a random sample will be used. If there is only one sample per class, the same
        sample will be used.
    Then, once the set of shapelets have been initialized, we extract the shapelet
    features from each pair of shapelets and input series. Three features are extracted:
        - min d(S,X): the minimum value of the distance vector between a shapelet S and
        a time series X.
        - argmin d(S,X): the location of the minumum.
        - SO(d(S,X), threshold): The number of point in the distance vector that are
        bellow the threshold parameter of the shapelet.

    This is a duplicate of the original implementation in aeon, adapted for bugfixing
    and experimentation. All credit to the original author @baraline for the
    implementation.

    Parameters
    ----------
    max_shapelets : int, default=10000
        The maximum number of shapelets to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array, default=None
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set. If None, the shapelets length will be equal to
        min(max(2,series_length//2),11).
    proba_normalization : float, default=0.8
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    distance_function: function
        A distance function defined as a numba function with signature as
        (x: np.ndarray, y: np.ndarray) -> float. The default distance function is the
        manhattan distance.
    threshold_percentiles : array, default=None
        The two percentiles used to select the threshold used to compute the Shapelet
        Occurrence feature. If None, the 5th and the 10th percentiles (i.e. [5,10])
        will be used.
    alpha_similarity : float, default=0.5
        The strength of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5, and the maximum is
        1. Value above it have no effect for now.
    use_prime_dilations : bool, default=False
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    n_jobs : int, default=1
        The number of threads used for both `fit` and `transform`.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    shapelets : list
        The stored shapelets. Each item in the list is a tuple containing:
            - shapelet values
            - length parameter
            - dilation parameter
            - threshold parameter
            - normalization parameter
            - mean parameter
            - standard deviation parameter

    Notes
    -----
    This implementation use all the features for multivariate shapelets, without
    affecting a random feature subsets to each shapelet as done in the original
    implementation. See `convst
    https://github.com/baraline/convst/blob/main/convst/transformers/rdst.py`_.

    References
    ----------
    .. [1] Antoine Guillaume et al. "Random Dilated Shapelet Transform: A New Approach
       for Time Series Shapelets", Pattern Recognition and Artificial Intelligence.
       ICPRAI 2022.
    .. [2] Antoine Guillaume, "Time series classification with shapelets: Application
       to predictive maintenance on event logs", PhD Thesis, University of Orléans,
       2023.
    """

    _tags = {
        "scitype:transform-output": "Primitives",
        "fit_is_empty": False,
        "univariate-only": False,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "requires_y": False,
        "capability:inverse_transform": False,
        "handles-missing-data": False,
    }

    def __init__(
        self,
        max_shapelets=10000,
        shapelet_lengths=None,
        proba_normalization=0.8,
        threshold_percentiles=None,
        alpha_similarity=0.5,
        use_prime_dilations=False,
        random_state=None,
        n_jobs=1,
    ):
        self.max_shapelets = max_shapelets
        self.shapelet_lengths = shapelet_lengths
        self.proba_normalization = proba_normalization
        self.threshold_percentiles = threshold_percentiles
        self.alpha_similarity = alpha_similarity
        self.use_prime_dilations = use_prime_dilations
        self.random_state = random_state
        self.n_jobs = n_jobs

        super(RandomDilatedShapeletTransformer, self).__init__()

    def fit(self, X, y=None):
        """Fit the random dilated shapelet transform to a specified X and y.

        Parameters
        ----------
        X: np.ndarray shape (n_instances, n_channels, series_length)
            The training input samples.
        y: array-like or list, default=None
            The class values for X. If not specified, a random sample (i.e. not of the
            same class) will be used when computing the threshold for the Shapelet
            Occurence feature.

        Returns
        -------
        self : RandomDilatedShapeletTransform
            This estimator.
        """
        X, y = self._validate_data(X=X, y=y, ensure_min_samples=2)
        X = self._convert_X(X)

        self._random_state = (
            np.int32(self.random_state) if isinstance(self.random_state, int) else None
        )

        self.n_instances_, self.n_channels_, self.series_length_ = X.shape

        self._check_input_params()

        self._n_jobs = check_n_jobs(self.n_jobs)
        set_num_threads(self._n_jobs)

        if y is None:
            y = np.zeros(self.n_instances_)
        else:
            y = LabelEncoder().fit_transform(y)

        if any(self.shapelet_lengths_ > self.series_length_):
            raise ValueError(
                "Shapelets lengths can't be superior to input length,",
                "but got shapelets_lengths = {} ".format(self.shapelet_lengths_),
                "with input length = {}".format(self.series_length_),
            )

        self.shapelets_ = _random_dilated_shapelet_extraction(
            X,
            y,
            self.max_shapelets,
            self.shapelet_lengths_,
            self.proba_normalization,
            self.threshold_percentiles_,
            self.alpha_similarity,
            self.use_prime_dilations,
            self._random_state,
        )

        return self

    def transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : np.ndarray shape (n_instances, n_channels, series_length)
            The input data to transform.

        Returns
        -------
        X_new : 2D np.array of shape = (n_instances, 3*n_shapelets)
            The transformed data.
        """
        check_is_fitted(self)

        X = self._validate_data(X=X, reset=False)
        X = self._convert_X(X)

        X_new = _dilated_shapelet_transform(X, self.shapelets_)
        return X_new

    def _check_input_params(self):
        if isinstance(self.max_shapelets, bool):
            raise TypeError(
                "'max_shapelets' must be an integer, got {}.".format(self.max_shapelets)
            )

        if not isinstance(self.max_shapelets, (int, np.integer)):
            raise TypeError(
                "'max_shapelets' must be an integer, got {}.".format(self.max_shapelets)
            )
        self.shapelet_lengths_ = self.shapelet_lengths
        if self.shapelet_lengths_ is None:
            self.shapelet_lengths_ = np.array(
                [min(max(2, self.series_length_ // 2), 11)]
            )
        else:
            if not isinstance(self.shapelet_lengths_, (list, tuple, np.ndarray)):
                raise TypeError(
                    "'shapelet_lengths' must be a list, a tuple or "
                    "an array (got {}).".format(self.shapelet_lengths_)
                )

            self.shapelet_lengths_ = np.array(self.shapelet_lengths_, dtype=np.int32)
            if not np.all(self.shapelet_lengths_ >= 2):
                warnings.warn(
                    "Some values in 'shapelet_lengths' are inferior to 2."
                    "These values will be ignored.",
                    stacklevel=2,
                )
                self.shapelet_lengths_ = self.shapelet_lengths[
                    self.shapelet_lengths_ >= 2
                ]

            if not np.all(self.shapelet_lengths_ <= self.series_length_):
                warnings.warn(
                    "All the values in 'shapelet_lengths' must be lower or equal to"
                    + "the series length. Shapelet lengths above it will be ignored.",
                    stacklevel=2,
                )
                self.shapelet_lengths_ = self.shapelet_lengths_[
                    self.shapelet_lengths_ <= self.series_length_
                ]

            if len(self.shapelet_lengths_) == 0:
                raise ValueError(
                    "Shapelet lengths array is empty, did you give shapelets lengths"
                    " superior to the size of the series ?"
                )

        self.threshold_percentiles_ = self.threshold_percentiles
        if self.threshold_percentiles_ is None:
            self.threshold_percentiles_ = np.array([5, 10])
        else:
            if not isinstance(self.threshold_percentiles_, (list, tuple, np.ndarray)):
                raise TypeError(
                    "Expected a list, numpy array or tuple for threshold_percentiles"
                )
            if len(self.threshold_percentiles_) != 2:
                raise ValueError(
                    "The threshold_percentiles param should be an array of size 2"
                )
            self.threshold_percentiles_ = np.asarray(self.threshold_percentiles_)

    def _more_tags(self) -> dict:
        return {
            "requires_y": True,
            "non_deterministic": True,
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"max_shapelets": 10}


@njit(fastmath=True, cache=True)
def _init_random_shapelet_params(
    max_shapelets,
    shapelet_lengths,
    proba_normalization,
    use_prime_dilations,
    n_channels,
    series_length,
):
    """Randomly initialize the parameters of the shapelets.

    Parameters
    ----------
    max_shapelets : int
        The maximum number of shapelet to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set.
    proba_normalization : float
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    use_prime_dilations : bool
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    n_channels : int
        Number of channels of the input time series.
    series_length : int
        Size of the input time series.

    Returns
    -------
    values : array, shape (max_shapelets, n_channels, max(shapelet_lengths))
        An initialized (empty) value array for each shapelet
    lengths : array, shape (max_shapelets)
        The randomly initialized length of each shapelet
    dilations : array, shape (max_shapelets)
        The randomly initialized dilation of each shapelet
    threshold : array, shape (max_shapelets)
        An initialized (empty) value array for each shapelet
    normalize : array, shape (max_shapelets)
        The randomly initialized normalization indicator of each shapelet
    means : array, shape (max_shapelets, n_channels)
        Means of the shapelets
    stds : array, shape (max_shapelets, n_channels)
        Standard deviation of the shapelets

    """
    # Lengths of the shapelets
    # test dtypes correctness
    lengths = np.random.choice(shapelet_lengths, size=max_shapelets).astype(np.int32)
    # Upper bound values for dilations
    dilations = np.zeros(max_shapelets, dtype=np.int32)
    upper_bounds = np.log2(np.floor_divide(series_length - 1, lengths - 1))

    if use_prime_dilations:
        _primes = prime_up_to(np.int32(2 ** upper_bounds.max()))
        # 1 is not prime, but it is still a valid dilation for the "prime" scheme
        primes = np.zeros((_primes.shape[0] + 1), dtype=np.int32)
        primes[0] = 1
        primes[1:] = _primes
        for i in prange(max_shapelets):
            shp_primes = primes[primes <= np.int32(2 ** upper_bounds[i])]
            dilations[i] = shp_primes[choice_log(shp_primes.shape[0], 1)[0]]
    else:
        for i in prange(max_shapelets):
            dilations[i] = np.int32(2 ** np.random.uniform(0, upper_bounds[i]))

    # Init threshold array
    threshold = np.zeros(max_shapelets, dtype=np.float64)

    # Init values array
    values = np.zeros(
        (max_shapelets, n_channels, max(shapelet_lengths)), dtype=np.float64
    )

    # Is shapelet using z-normalization ?
    normalize = np.random.random(size=max_shapelets)
    normalize = normalize < proba_normalization

    means = np.zeros((max_shapelets, n_channels), dtype=np.float64)
    stds = np.zeros((max_shapelets, n_channels), dtype=np.float64)

    return values, lengths, dilations, threshold, normalize, means, stds


@njit(fastmath=True, cache=True, parallel=True)
def _random_dilated_shapelet_extraction(
    X,
    y,
    max_shapelets,
    shapelet_lengths,
    proba_normalization,
    threshold_percentiles,
    alpha_similarity,
    use_prime_dilations,
    seed,
):
    """Randomly generate a set of shapelets given the input parameters.

    Parameters
    ----------
    X : array, shape (n_instances, n_channels, series_length)
        Time series dataset
    y : array, shape (n_instances)
        Class of each input time series
    max_shapelets : int
        The maximum number of shapelet to keep for the final transformation.
        A lower number of shapelets can be kept if alpha similarity have discarded the
        whole dataset.
    shapelet_lengths : array
        The set of possible length for shapelets. Each shapelet length is uniformly
        drawn from this set.
    proba_normalization : float
        This probability (between 0 and 1) indicate the chance of each shapelet to be
        initialized such as it will use a z-normalized distance, inducing either scale
        sensitivity or invariance. A value of 1 would mean that all shapelets will use
        a z-normalized distance.
    threshold_percentiles : array
        The two percentiles used to select the threshold used to compute the Shapelet
        Occurrence feature.
    alpha_similarity : float
        The strength of the alpha similarity pruning. The higher the value, the lower
        the allowed number of common indexes with previously sampled shapelets
        when sampling a new candidate with the same dilation parameter.
        It can cause the number of sampled shapelets to be lower than max_shapelets if
        the whole search space has been covered. The default is 0.5.
    use_prime_dilations : bool
        If True, restrict the value of the shapelet dilation parameter to be prime
        values. This can greatly speed up the algorithm for long time series and/or
        short shapelet length, possibly at the cost of some accuracy.
    seed : int
        Seed for random number generation.

    Returns
    -------
    Shapelets : tuple
    The returned tuple contains 7 arrays describing the shapelets parameters:
        - values : array, shape (max_shapelets, n_channels, max(shapelet_lengths))
            Values of the shapelets.
        - lengths : array, shape (max_shapelets)
            Length parameter of the shapelets
        - dilations : array, shape (max_shapelets)
            Dilation parameter of the shapelets
        - threshold : array, shape (max_shapelets)
            Threshold parameter of the shapelets
        - normalize : array, shape (max_shapelets)
            Normalization indicator of the shapelets
        - means : array, shape (max_shapelets, n_channels)
            Means of the shapelets
        - stds : array, shape (max_shapelets, n_channels)
            Standard deviation of the shapelets
    """
    n_instances, n_channels, series_length = X.shape

    # Fix the random seed
    set_numba_random_seed(seed)

    # Initialize shapelets
    (
        values,
        lengths,
        dilations,
        threshold,
        normalize,
        means,
        stds,
    ) = _init_random_shapelet_params(
        max_shapelets,
        shapelet_lengths,
        proba_normalization,
        use_prime_dilations,
        n_channels,
        series_length,
    )

    # Get unique dilations to loop over
    unique_dil = np.unique(dilations)
    n_dilations = unique_dil.shape[0]

    # For each dilation, we can do in parallel
    for i_dilation in prange(n_dilations):
        # (2, _, _): Mask is different for normalized and non-normalized shapelets
        alpha_mask = np.ones((2, n_instances, series_length), dtype=np.bool_)
        id_shps = np.where(dilations == unique_dil[i_dilation])[0]
        min_len = min(lengths[id_shps])
        # For each shapelet id with this dilation
        for i_shp in id_shps:
            # Get shapelet params
            dilation = dilations[i_shp]
            length = lengths[i_shp]
            norm = np.int32(normalize[i_shp])
            dist_vect_shape = series_length - (length - 1) * dilation

            # Possible sampling points given self similarity mask
            current_mask = alpha_mask[norm, :, :dist_vect_shape]
            idx_mask = np.where(current_mask)

            n_admissible_points = idx_mask[0].shape[0]
            if n_admissible_points > 0:
                # Choose a sample and a timestamp
                idx_choice = np.random.choice(n_admissible_points)
                idx_sample = idx_mask[0][idx_choice]
                idx_timestamp = idx_mask[1][idx_choice]

                # Update the mask in two directions from the sampling point
                alpha_size = length - int(max(1, (1 - alpha_similarity) * min_len))
                for j in range(alpha_size):
                    alpha_mask[
                        norm, idx_sample, (idx_timestamp - (j * dilation))
                    ] = False
                    alpha_mask[
                        norm, idx_sample, (idx_timestamp + (j * dilation))
                    ] = False

                # Extract the values of shapelet
                if norm:
                    _val, _means, _stds = get_subsequence_with_mean_std(
                        X[idx_sample], idx_timestamp, length, dilation
                    )
                    for i_channel in prange(_val.shape[0]):
                        if _stds[i_channel] > 0:
                            _val[i_channel] = (
                                _val[i_channel] - _means[i_channel]
                            ) / _stds[i_channel]
                        else:
                            _val[i_channel] = _val[i_channel] - _means[i_channel]
                else:
                    _val = get_subsequence(
                        X[idx_sample], idx_timestamp, length, dilation
                    )

                # Select another sample of the same class as the sample used to
                loc_others = np.where(y == y[idx_sample])[0]
                if loc_others.shape[0] > 1:
                    loc_others = loc_others[loc_others != idx_sample]
                    id_test = np.random.choice(loc_others)
                else:
                    id_test = idx_sample

                # Compute distance vector, first get the subsequences
                X_subs = _get_all_subsequences(X[id_test], length, dilation)
                if norm:
                    # Normalize them if needed
                    X_means, X_stds = sliding_mean_std_one_series(
                        X[id_test], length, dilation
                    )
                    X_subs = _normalize_subsequences(X_subs, X_means, X_stds)

                x_dist = _compute_shapelet_dist_vector(X_subs, _val, length)

                lower_bound = np.percentile(x_dist, threshold_percentiles[0])
                upper_bound = np.percentile(x_dist, threshold_percentiles[1])

                threshold[i_shp] = np.random.uniform(lower_bound, upper_bound)
                values[i_shp, :, :length] = _val
                if norm:
                    means[i_shp] = _means
                    stds[i_shp] = _stds

    mask_values = np.ones(max_shapelets, dtype=np.bool_)
    for i in prange(max_shapelets):
        if threshold[i] == 0 and np.all(values[i] == 0):
            mask_values[i] = False

    return (
        values[mask_values],
        lengths[mask_values],
        dilations[mask_values],
        threshold[mask_values],
        normalize[mask_values],
        means[mask_values],
        stds[mask_values],
    )


@njit(fastmath=True, cache=True, parallel=True)
def _dilated_shapelet_transform(X, shapelets):
    """Perform the shapelet transform with a set of shapelets and a set of time series.

    Parameters
    ----------
    X : array, shape (n_instances, n_channels, series_length)
        Time series dataset
    shapelets : tuple
        The returned tuple contains 7 arrays describing the shapelets parameters:
            - values : array, shape (n_shapelets, n_channels, max(shapelet_lengths))
                Values of the shapelets.
            - lengths : array, shape (n_shapelets)
                Length parameter of the shapelets
            - dilations : array, shape (n_shapelets)
                Dilation parameter of the shapelets
            - threshold : array, shape (n_shapelets)
                Threshold parameter of the shapelets
            - normalize : array, shape (n_shapelets)
                Normalization indicator of the shapelets
            - means : array, shape (n_shapelets, n_channels)
                Means of the shapelets
            - stds : array, shape (n_shapelets, n_channels)
                Standard deviation of the shapelets

    Returns
    -------
    X_new : array, shape=(n_instances, 3*n_shapelets)
        The transformed input time series with each shapelet extracting 3
        feature from the distance vector computed on each time series.

    """
    (values, lengths, dilations, threshold, normalize, means, stds) = shapelets
    n_shapelets = len(lengths)
    n_instances, n_channels, series_length = X.shape

    n_ft = 3

    # (u_l * u_d , 2)
    params_shp = combinations_1d(lengths, dilations)

    X_new = np.zeros((n_instances, n_ft * n_shapelets))
    for i_params in prange(params_shp.shape[0]):
        length = params_shp[i_params, 0]
        dilation = params_shp[i_params, 1]
        id_shps = np.where((lengths == length) & (dilations == dilation))[0]

        for i_x in prange(n_instances):
            X_subs = _get_all_subsequences(X[i_x], length, dilation)
            idx_no_norm = id_shps[np.where(~normalize[id_shps])[0]]
            for i_shp in idx_no_norm:
                X_new[
                    i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)
                ] = _compute_shapelet_features(
                    X_subs,
                    values[i_shp],
                    length,
                    threshold[i_shp],
                )

            idx_norm = id_shps[np.where(normalize[id_shps])[0]]
            if len(idx_norm) > 0:
                X_means, X_stds = sliding_mean_std_one_series(X[i_x], length, dilation)
                X_subs = _normalize_subsequences(X_subs, X_means, X_stds)
                for i_shp in idx_norm:
                    X_new[
                        i_x, (n_ft * i_shp) : (n_ft * i_shp + n_ft)
                    ] = _compute_shapelet_features(
                        X_subs,
                        values[i_shp],
                        length,
                        threshold[i_shp],
                    )
    return X_new


@njit(fastmath=True, cache=True)
def _normalize_subsequences(X_subs, X_means, X_stds):
    """
    Generate subsequences from a time series given the length and dilation parameters.

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    X_means : array, shape (n_channels, n_timestamps-(length-1)*dilation)
        Length of the subsequences to generate.
    X_stds : array, shape (n_channels, n_timestamps-(length-1)*dilation)
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_timestamps-(length-1)*dilation, n_channels, length)
        Subsequences of the input time series.
    """
    n_subsequences, n_channels, length = X_subs.shape
    X_new = np.zeros((n_subsequences, n_channels, length))
    for i_sub in prange(n_subsequences):
        for i_channel in prange(n_channels):
            if X_stds[i_channel, i_sub] > 0:
                X_new[i_sub, i_channel] = (
                    X_subs[i_sub, i_channel] - X_means[i_channel, i_sub]
                ) / X_stds[i_channel, i_sub]
            # else it gives 0, the default value
    return X_new


@njit(fastmath=True, cache=True)
def _get_all_subsequences(X, length, dilation):
    """
    Generate subsequences from a time series given the length and dilation parameters.

    Parameters
    ----------
    X : array, shape = (n_channels, n_timestamps)
        An input time series as (n_channels, n_timestamps).
    length : int
        Length of the subsequences to generate.
    dilation : int
        Dilation parameter to apply when generating the strides.

    Returns
    -------
    array, shape = (n_timestamps-(length-1)*dilation, n_channels, length)
        Subsequences of the input time series.
    """
    n_channels, n_timestamps = X.shape
    n_subsequences = n_timestamps - (length - 1) * dilation
    X_subs = np.zeros((n_subsequences, n_channels, length))
    for i_sub in prange(n_subsequences):
        for i_channel in prange(n_channels):
            for i_length in prange(length):
                X_subs[i_sub, i_channel, i_length] = X[
                    i_channel, i_sub + (i_length * dilation)
                ]
    return X_subs


@njit(fastmath=True, cache=True)
def _compute_shapelet_features(X_subs, values, length, threshold):
    """Extract the features from a shapelet distance vector.

    Given a shapelet and a time series, extract three features from the resulting
    distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Length of the shapelet
    values : array, shape (n_channels, length)
        The resulting subsequence
    threshold : float
        The threshold parameter of the shapelet
    distance_function: function
        A distance function defined as a numba function with signature as
        (x: np.ndarray, y: np.ndarray) -> float. The default distance function is the
        manhattan distance.

    Returns
    -------
    min, argmin, shapelet occurence
        The three computed features as float dtypes
    """
    _min = np.inf
    _argmin = np.inf
    _SO = 0

    n_subsequences = X_subs.shape[0]

    for i_sub in prange(n_subsequences):
        _dist = manhattan_distance(X_subs[i_sub], values[:, :length])

        if _dist < _min:
            _min = _dist
            _argmin = i_sub
        if _dist < threshold:
            _SO += 1

    return np.float64(_min), np.float64(_argmin), np.float64(_SO)


@njit(fastmath=True, cache=True)
def _compute_shapelet_dist_vector(X_subs, values, length):
    """Extract the features from a shapelet distance vector.

    Given a shapelet and a time series, extract three features from the resulting
    distance vector:
        - min
        - argmin
        - Shapelet Occurence : number of point in the distance vector inferior to the
        threshold parameter

    Parameters
    ----------
    X_subs : array, shape (n_timestamps-(length-1)*dilation, n_channels, length)
        The subsequences of an input time series given the length and dilation parameter
    values : array, shape (n_channels, length)
        The value array of the shapelet
    length : int
        Dilation of the shapelet

    Returns
    -------
    min, argmin, shapelet occurence
        The three computed features as float dtypes
    """
    n_subsequences = X_subs.shape[0]
    dist_vector = np.zeros(n_subsequences)
    for i_sub in prange(n_subsequences):
        dist_vector[i_sub] = manhattan_distance(X_subs[i_sub], values[:, :length])
    return dist_vector
