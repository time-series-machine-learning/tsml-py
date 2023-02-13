# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]
__all__ = ["check_n_jobs"]

import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def check_n_jobs(n_jobs: int) -> int:
    """Check `n_jobs` parameter according to the scikit-learn convention.

    https://scikit-learn.org/stable/glossary.html#term-n_jobs

    Parameters
    ----------
    n_jobs : int or None
        The number of jobs for parallelization.
        If None or 0, 1 is used.
        If negative, (n_cpus + 1 + n_jobs) is used. In such a case, -1 would use all
        available CPUs and -2 would use all but one.

    Returns
    -------
    n_jobs : int
        The number of threads to be used.
    """
    if n_jobs is None or n_jobs == 0:
        return 1
    elif not isinstance(n_jobs, (int, np.integer)):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return max(1, os.cpu_count() + 1 + n_jobs)
    else:
        return n_jobs


def is_transformer(estimator):
    """Check if an estimator is a transformer. todo

    Parameters
    ----------
    estimator : object
        The estimator to check.

    Returns
    -------
    is_transformer : bool
        True if the estimator is a transformer, False otherwise.
    """
    return isinstance(estimator, TransformerMixin) and isinstance(
        estimator, BaseEstimator
    )
