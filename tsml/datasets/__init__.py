# -*- coding: utf-8 -*-
"""Datasets and data loaders."""

__all__ = [
    "load_from_ts_file",
    "load_minimal_chinatown",
    "load_unequal_minimal_chinatown",
    "load_equal_minimal_japanese_vowels",
    "load_minimal_japanese_vowels",
    "load_minimal_gas_prices",
    "load_unequal_minimal_gas_prices",
]

from tsml.datasets._data_io import (
    load_equal_minimal_japanese_vowels,
    load_from_ts_file,
    load_minimal_chinatown,
    load_minimal_gas_prices,
    load_minimal_japanese_vowels,
    load_unequal_minimal_chinatown,
    load_unequal_minimal_gas_prices,
)
