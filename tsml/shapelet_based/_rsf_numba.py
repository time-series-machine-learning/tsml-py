"""Numba functions for building and applying random shapelet trees.

A port of the shapelet tree implementation from wildboar 1.2.0 (BSD 3 clause,
Isak Samsten), restricted to the Euclidean distance. The random number generator,
the sample ordering (introsort) and the split search follow the original so that a
tree built from the same sample weights and seed is identical.
"""

__author__ = ["MatthewMiddlehurst"]

import numpy as np
from numba import njit

RAND_R_MAX = 2147483647
ATTRIBUTE_THRESHOLD = 1e-7

GINI = 0
ENTROPY = 1
SQUARED_ERROR = 2


@njit(cache=True)
def _rand_r(seed):
    # seed is a length 1 int64 array holding an unsigned 32-bit state
    seed[0] = (seed[0] * 1103515245 + 12345) & 0xFFFFFFFF
    return seed[0] % (RAND_R_MAX + 1)


@njit(cache=True)
def _rand_int(min_val, max_val, seed):
    """Return a random integer in the range [min_val, max_val)."""
    if min_val == max_val:
        return min_val
    return min_val + _rand_r(seed) % (max_val - min_val)


@njit(cache=True)
def _euclidean_distance(x, s):
    """Minimum Euclidean distance between shapelet s and all subsequences of x."""
    s_length = s.shape[0]
    min_dist = np.inf
    for i in range(x.shape[0] - s_length + 1):
        dist = 0.0
        for j in range(s_length):
            if dist >= min_dist:
                break
            v = x[i + j] - s[j]
            dist += v * v

        if dist < min_dist:
            min_dist = dist

    return np.sqrt(min_dist)


@njit(cache=True)
def _swap(values, samples, i, j):
    values[i], values[j] = values[j], values[i]
    samples[i], samples[j] = samples[j], samples[i]


@njit(cache=True)
def _median3(values, offset, n):
    a = values[offset]
    b = values[offset + n // 2]
    c = values[offset + n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


@njit(cache=True)
def _sift_down(values, samples, offset, start, end):
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and values[offset + maxind] < values[offset + child]:
            maxind = child
        if child + 1 < end and values[offset + maxind] < values[offset + child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            _swap(values, samples, offset + root, offset + maxind)
            root = maxind


@njit(cache=True)
def _heapsort(values, samples, offset, n):
    start = (n - 2) // 2
    end = n
    while True:
        _sift_down(values, samples, offset, start, end)
        if start == 0:
            break
        start -= 1

    end = n - 1
    while end > 0:
        _swap(values, samples, offset, offset + end)
        _sift_down(values, samples, offset, 0, end)
        end = end - 1


@njit(cache=True)
def _argsort(values, samples, offset, n):
    """Sort values[offset:offset + n] in place, applying the same swaps to samples.

    An iterative version of the introsort used by wildboar. Ties are ordered the
    same way, which matters because shapelets are sampled by position in samples.
    """
    if n == 0:
        return

    stack = np.empty((max(n, 1), 3), dtype=np.int64)
    stack[0, 0] = offset
    stack[0, 1] = n
    stack[0, 2] = 2 * int(np.log2(n))
    stack_size = 1

    while stack_size > 0:
        stack_size -= 1
        off = stack[stack_size, 0]
        m = stack[stack_size, 1]
        maxd = stack[stack_size, 2]

        while m > 1:
            if maxd <= 0:
                _heapsort(values, samples, off, m)
                break
            maxd -= 1

            pivot = _median3(values, off, m)

            i = 0
            left = 0
            right = m
            while i < right:
                value = values[off + i]
                if value < pivot:
                    _swap(values, samples, off + i, off + left)
                    i += 1
                    left += 1
                elif value > pivot:
                    right -= 1
                    _swap(values, samples, off + i, off + right)
                else:
                    i += 1

            # the ranges are disjoint, so the order they are sorted in does not
            # change the result
            if left > 1:
                stack[stack_size, 0] = off
                stack[stack_size, 1] = left
                stack[stack_size, 2] = maxd
                stack_size += 1

            off += right
            m -= right


@njit(cache=True)
def _criterion_init(
    criterion,
    start,
    end,
    samples,
    sample_weight,
    y_cls,
    y_reg,
    sum_total,
    stats,
):
    # stats: [weighted_n_total, weighted_n_left, weighted_n_right,
    #         sum_total_reg, sum_left_reg, sum_right_reg, sum_sq_total_reg]
    stats[:] = 0.0
    if criterion == SQUARED_ERROR:
        for i in range(start, end):
            j = samples[i]
            w = sample_weight[j]
            stats[3] += w * y_reg[j]
            stats[6] += w * y_reg[j] * y_reg[j]
            stats[0] += w
    else:
        sum_total[:] = 0.0
        for i in range(start, end):
            j = samples[i]
            w = sample_weight[j]
            sum_total[y_cls[j]] += w
            stats[0] += w


@njit(cache=True)
def _criterion_reset(criterion, sum_total, sum_left, sum_right, stats):
    stats[1] = 0.0
    stats[2] = stats[0]
    if criterion == SQUARED_ERROR:
        stats[4] = 0.0
        stats[5] = stats[3]
    else:
        sum_left[:] = 0.0
        sum_right[:] = sum_total


@njit(cache=True)
def _criterion_update(
    criterion,
    pos,
    new_pos,
    samples,
    sample_weight,
    y_cls,
    y_reg,
    sum_total,
    sum_left,
    sum_right,
    stats,
):
    if criterion == SQUARED_ERROR:
        for i in range(pos, new_pos):
            j = samples[i]
            w = sample_weight[j]
            stats[4] += w * y_reg[j]
            stats[1] += w
        stats[2] = stats[0] - stats[1]
        stats[5] = stats[3] - stats[4]
    else:
        for i in range(pos, new_pos):
            j = samples[i]
            w = sample_weight[j]
            sum_left[y_cls[j]] += w
            stats[1] += w
        stats[2] = stats[0] - stats[1]
        for i in range(sum_total.shape[0]):
            sum_right[i] = sum_total[i] - sum_left[i]


@njit(cache=True)
def _node_impurity(criterion, sum_total, stats):
    if criterion == SQUARED_ERROR:
        impurity = stats[6] / stats[0]
        impurity -= (stats[3] / stats[0]) ** 2
        return impurity
    elif criterion == GINI:
        sq_count = 0.0
        for i in range(sum_total.shape[0]):
            c = sum_total[i]
            sq_count += c * c
        return 1.0 - sq_count / (stats[0] * stats[0])
    else:
        entropy = 0.0
        for i in range(sum_total.shape[0]):
            c = sum_total[i]
            if c > 0:
                c /= stats[0]
                entropy -= c * np.log2(c)
        return entropy


@njit(cache=True)
def _child_impurity(
    criterion, start, pos, samples, sample_weight, y_reg, sum_left, sum_right, stats
):
    if criterion == SQUARED_ERROR:
        left_sq_sum = 0.0
        for i in range(start, pos):
            j = samples[i]
            left_sq_sum += sample_weight[j] * y_reg[j] * y_reg[j]
        right_sq_sum = stats[6] - left_sq_sum

        left = left_sq_sum / stats[1]
        left -= (stats[4] / stats[1]) ** 2
        right = right_sq_sum / stats[2]
        right -= (stats[5] / stats[2]) ** 2
        return left, right
    elif criterion == GINI:
        sq_left = 0.0
        sq_right = 0.0
        for i in range(sum_left.shape[0]):
            v = sum_left[i]
            sq_left += v * v
            v = sum_right[i]
            sq_right += v * v
        left = 1 - sq_left / (stats[1] * stats[1])
        right = 1 - sq_right / (stats[2] * stats[2])
        return left, right
    else:
        left = 0.0
        right = 0.0
        for i in range(sum_left.shape[0]):
            v = sum_left[i]
            if v > 0:
                v /= stats[1]
                left -= v * np.log2(v)
            v = sum_right[i]
            if v > 0:
                v /= stats[2]
                right -= v * np.log2(v)
        return left, right


@njit(cache=True)
def _proxy_impurity(
    criterion, start, pos, samples, sample_weight, y_reg, sum_left, sum_right, stats
):
    if criterion == SQUARED_ERROR:
        return stats[4] * stats[4] / stats[1] + stats[5] * stats[5] / stats[2]
    left, right = _child_impurity(
        criterion, start, pos, samples, sample_weight, y_reg, sum_left, sum_right, stats
    )
    return -stats[2] * right - stats[1] * left


@njit(cache=True, nogil=True)
def _build_tree(  # noqa: PLR0912, PLR0915
    X,
    y_cls,
    y_reg,
    sample_weight,
    n_outputs,
    criterion,
    n_shapelets,
    alpha,
    min_shapelet_length,
    max_shapelet_length,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    min_impurity_decrease,
    seed,
):
    """Build a single shapelet tree.

    Parameters
    ----------
    X : np.ndarray of shape (n_cases, n_channels, n_timepoints)
        The training data.
    y_cls : np.ndarray of shape (n_cases,)
        Encoded class labels. Unused for regression.
    y_reg : np.ndarray of shape (n_cases,)
        Regression targets. Unused for classification.
    sample_weight : np.ndarray of shape (n_cases,)
        Weight of each case. Cases with zero weight are not used.
    n_outputs : int
        The number of classes, or 1 for regression.
    criterion : int
        One of GINI, ENTROPY or SQUARED_ERROR.
    n_shapelets : int
        The number of shapelets to sample at each node.
    alpha : float
        If not 0, the number of shapelets sampled grows (alpha > 0) or shrinks
        (alpha < 0) with the depth of the node.
    min_shapelet_length, max_shapelet_length : int
        Shapelet lengths are sampled from [min_shapelet_length, max_shapelet_length).
    max_depth : int
        The maximum depth of the tree.
    min_samples_split, min_samples_leaf : int
        Nodes with fewer than min_samples_split or 2 * min_samples_leaf cases
        (with non-zero weight) become leaves.
    min_impurity_decrease : float
        Splits must decrease the weighted impurity by more than this.
    seed : int
        Seed for the random number generator, in [0, 2147483647).

    Returns
    -------
    node_count : int
    left, right : np.ndarray of int, -1 for leaves
    threshold : np.ndarray of float
    shapelet_info : np.ndarray of shape (n_nodes, 4)
        For each branch node, the case, channel, start and length of the shapelet.
    value : np.ndarray of shape (n_nodes, n_outputs)
        The leaf predictions.
    """
    n_cases = X.shape[0]
    random_seed = np.zeros(1, dtype=np.int64)
    random_seed[0] = seed

    samples = np.empty(n_cases, dtype=np.int64)
    n_samples = 0
    n_weighted_samples = 0.0
    for i in range(n_cases):
        if sample_weight[i] != 0.0:
            samples[n_samples] = i
            n_samples += 1
            n_weighted_samples += sample_weight[i]

    samples_buffer = np.empty(n_samples, dtype=np.int64)
    attribute_buffer = np.empty(n_samples, dtype=np.float64)

    capacity = max(1, 2 * n_samples - 1)
    left = np.full(capacity, -1, dtype=np.int64)
    right = np.full(capacity, -1, dtype=np.int64)
    threshold = np.full(capacity, np.nan, dtype=np.float64)
    shapelet_info = np.full((capacity, 4), -1, dtype=np.int64)
    value = np.zeros((capacity, n_outputs), dtype=np.float64)

    n_labels = n_outputs if criterion != SQUARED_ERROR else 1
    sum_total = np.zeros(n_labels, dtype=np.float64)
    sum_left = np.zeros(n_labels, dtype=np.float64)
    sum_right = np.zeros(n_labels, dtype=np.float64)
    stats = np.zeros(7, dtype=np.float64)

    # depth first, left child first, matching the recursion in wildboar.
    # stack rows: start, end, depth, parent, is_left; impurity stored separately
    stack = np.empty((capacity, 5), dtype=np.int64)
    stack_impurity = np.empty(capacity, dtype=np.float64)
    stack[0, 0] = 0
    stack[0, 1] = n_samples
    stack[0, 2] = 0
    stack[0, 3] = -1
    stack[0, 4] = 0
    stack_impurity[0] = np.nan
    stack_size = 1
    node_count = 0

    while stack_size > 0:
        stack_size -= 1
        start = stack[stack_size, 0]
        end = stack[stack_size, 1]
        depth = stack[stack_size, 2]
        parent = stack[stack_size, 3]
        is_left = stack[stack_size, 4] == 1
        impurity = stack_impurity[stack_size]

        node_id = node_count
        node_count += 1
        if parent >= 0:
            if is_left:
                left[parent] = node_id
            else:
                right[parent] = node_id

        _criterion_init(
            criterion,
            start,
            end,
            samples,
            sample_weight,
            y_cls,
            y_reg,
            sum_total,
            stats,
        )
        n_node_samples = end - start
        is_leaf = (
            depth >= max_depth
            or n_node_samples < min_samples_split
            or n_node_samples < 2 * min_samples_leaf
        )

        if not is_leaf:
            if parent < 0:
                impurity = _node_impurity(criterion, sum_total, stats)

            if alpha != 0.0:
                weight = 1.0 - np.exp(-np.abs(alpha) * depth)
                if alpha < 0:
                    weight = 1 - weight
                n_attributes = max(1, int(np.ceil(n_shapelets * weight)))
            else:
                n_attributes = n_shapelets

            best_impurity = -np.inf
            best_split_point = 0
            best_threshold = np.nan
            best_index = -1
            best_dim = -1
            best_start = -1
            best_length = -1

            for _ in range(n_attributes):
                shapelet_length = _rand_int(
                    min_shapelet_length, max_shapelet_length, random_seed
                )
                shapelet_start = _rand_int(0, X.shape[2] - shapelet_length, random_seed)
                shapelet_index = samples[
                    start + _rand_int(0, n_node_samples, random_seed)
                ]
                if X.shape[1] > 1:
                    shapelet_dim = _rand_int(0, X.shape[1], random_seed)
                else:
                    shapelet_dim = 0

                shapelet = X[
                    shapelet_index,
                    shapelet_dim,
                    shapelet_start : shapelet_start + shapelet_length,
                ]
                for i in range(start, end):
                    attribute_buffer[i] = _euclidean_distance(
                        X[samples[i], shapelet_dim], shapelet
                    )
                _argsort(attribute_buffer, samples, start, n_node_samples)

                # all attribute values are constant
                if (
                    attribute_buffer[end - 1]
                    <= attribute_buffer[start] + ATTRIBUTE_THRESHOLD
                ):
                    continue

                _criterion_reset(criterion, sum_total, sum_left, sum_right, stats)

                current_impurity = -np.inf
                current_threshold = np.nan
                current_split_point = 0
                pos = start
                i = start
                while i < end:
                    # ignore split points with almost equal attribute values
                    while i + 1 < end and (
                        attribute_buffer[i + 1]
                        <= attribute_buffer[i] + ATTRIBUTE_THRESHOLD
                    ):
                        i += 1

                    i += 1
                    if i < end:
                        _criterion_update(
                            criterion,
                            pos,
                            i,
                            samples,
                            sample_weight,
                            y_cls,
                            y_reg,
                            sum_total,
                            sum_left,
                            sum_right,
                            stats,
                        )
                        pos = i
                        proxy = _proxy_impurity(
                            criterion,
                            start,
                            pos,
                            samples,
                            sample_weight,
                            y_reg,
                            sum_left,
                            sum_right,
                            stats,
                        )
                        if proxy > current_impurity:
                            current_impurity = proxy
                            current_threshold = (
                                attribute_buffer[i - 1] / 2.0
                                + attribute_buffer[i] / 2.0
                            )
                            current_split_point = pos

                            if (
                                current_threshold == attribute_buffer[i]
                                or current_threshold == np.inf
                                or current_threshold == -np.inf
                            ):
                                current_threshold = attribute_buffer[i - 1]

                if current_impurity > best_impurity:
                    samples_buffer[:n_node_samples] = samples[start:end]
                    best_impurity = current_impurity
                    best_split_point = current_split_point
                    best_threshold = current_threshold
                    best_index = shapelet_index
                    best_dim = shapelet_dim
                    best_start = shapelet_start
                    best_length = shapelet_length

            is_leaf = best_index == -1
            if not is_leaf:
                # restore the order of the best split
                samples[start:end] = samples_buffer[:n_node_samples]

                _criterion_reset(criterion, sum_total, sum_left, sum_right, stats)
                _criterion_update(
                    criterion,
                    start,
                    best_split_point,
                    samples,
                    sample_weight,
                    y_cls,
                    y_reg,
                    sum_total,
                    sum_left,
                    sum_right,
                    stats,
                )
                impurity_left, impurity_right = _child_impurity(
                    criterion,
                    start,
                    best_split_point,
                    samples,
                    sample_weight,
                    y_reg,
                    sum_left,
                    sum_right,
                    stats,
                )
                impurity_improvement = (stats[0] / n_weighted_samples) * (
                    impurity
                    - (stats[2] / stats[0] * impurity_right)
                    - (stats[1] / stats[0] * impurity_left)
                )

                is_leaf = (
                    best_split_point <= start
                    or best_split_point >= end
                    or impurity_improvement <= min_impurity_decrease
                )

        if is_leaf:
            # stats and sum_total are unchanged since _criterion_init
            if criterion == SQUARED_ERROR:
                value[node_id, 0] = stats[3] / stats[0]
            else:
                for k in range(n_outputs):
                    value[node_id, k] = sum_total[k] / stats[0]
        else:
            threshold[node_id] = best_threshold
            shapelet_info[node_id, 0] = best_index
            shapelet_info[node_id, 1] = best_dim
            shapelet_info[node_id, 2] = best_start
            shapelet_info[node_id, 3] = best_length

            # push right first so the left child is built first
            stack[stack_size, 0] = best_split_point
            stack[stack_size, 1] = end
            stack[stack_size, 2] = depth + 1
            stack[stack_size, 3] = node_id
            stack[stack_size, 4] = 0
            stack_impurity[stack_size] = impurity_right
            stack_size += 1

            stack[stack_size, 0] = start
            stack[stack_size, 1] = best_split_point
            stack[stack_size, 2] = depth + 1
            stack[stack_size, 3] = node_id
            stack[stack_size, 4] = 1
            stack_impurity[stack_size] = impurity_left
            stack_size += 1

    return (
        node_count,
        left[:node_count].copy(),
        right[:node_count].copy(),
        threshold[:node_count].copy(),
        shapelet_info[:node_count].copy(),
        value[:node_count].copy(),
    )


@njit(cache=True, nogil=True)
def _apply_tree(X, left, threshold, right, shapelet_dim, shapelet_offset, shapelets):
    """Return the leaf each case in X ends up in.

    The shapelet of branch node i is
    `shapelets[shapelet_offset[i]:shapelet_offset[i + 1]]`.
    """
    out = np.zeros(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        node = 0
        while left[node] != -1:
            dist = _euclidean_distance(
                X[i, shapelet_dim[node]],
                shapelets[shapelet_offset[node] : shapelet_offset[node + 1]],
            )
            if dist <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        out[i] = node
    return out
