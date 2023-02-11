# -*- coding: utf-8 -*-
from abc import ABCMeta
from typing import Union

import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "BaseTimeSeriesEstimator",
    "clone_estimator",
]


class BaseTimeSeriesEstimator(BaseEstimator, metaclass=ABCMeta):
    pass


def clone_estimator(
    base_estimator: BaseEstimator, random_state: Union[None, int, RandomState] = None
) -> BaseEstimator:
    estimator = clone(base_estimator)

    if random_state is not None:
        # contents of _set_random_states from scikit-learn 1.2.1
        random_state = check_random_state(random_state)
        to_set = {}
        for key in sorted(estimator.get_params(deep=True)):
            if key == "random_state" or key.endswith("__random_state"):
                to_set[key] = random_state.randint(np.iinfo(np.int32).max)

        if to_set:
            estimator.set_params(**to_set)

    return estimator
