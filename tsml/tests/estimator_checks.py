# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst"]

from sklearn.utils.estimator_checks import _yield_all_checks


def _yield_all_time_series_checks(estimator):
    _yield_all_checks(estimator)
