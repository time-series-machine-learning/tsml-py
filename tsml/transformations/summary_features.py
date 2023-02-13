# -*- coding: utf-8 -*-
__author__ = ["MatthewMiddlehurst"]
__all__ = ["SummaryFeatures"]

from sklearn.base import TransformerMixin

from tsml.base import BaseTimeSeriesEstimator


class SummaryFeatures(TransformerMixin, BaseTimeSeriesEstimator):
    def __init__(
        self,
        summary_function=None,
        quantiles=None,
    ):
        self.summary_function = summary_function
        self.quantiles = quantiles

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        pass
