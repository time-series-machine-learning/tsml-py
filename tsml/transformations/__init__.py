# -*- coding: utf-8 -*-
"""tsml transformations."""

__all__ = [
    "AutocorrelationFunctionTransformer",
    "ARCoefficientTransformer",
    "Catch22Transformer",
    "FPCATransformer",
    "FunctionTransformer",
    "RandomIntervalTransformer",
    "SupervisedIntervalTransformer",
    "PeriodogramTransformer",
    # "SFATransformer",
    "RandomShapeletTransformer",
    "RandomDilatedShapeletTransformer",
    "SevenNumberSummaryTransformer",
    "TransformerConcatenator",
]

from tsml.transformations._acf import AutocorrelationFunctionTransformer
from tsml.transformations._ar_coefficient import ARCoefficientTransformer
from tsml.transformations._catch22 import Catch22Transformer
from tsml.transformations._fpca import FPCATransformer
from tsml.transformations._function_transformer import FunctionTransformer
from tsml.transformations._interval_extraction import (
    RandomIntervalTransformer,
    SupervisedIntervalTransformer,
)
from tsml.transformations._periodogram import PeriodogramTransformer
from tsml.transformations._shapelet_transform import (
    RandomDilatedShapeletTransformer,
    RandomShapeletTransformer,
)
from tsml.transformations._summary_features import SevenNumberSummaryTransformer
from tsml.transformations._transform_concatenator import TransformerConcatenator
