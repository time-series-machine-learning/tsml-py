# -*- coding: utf-8 -*-
"""Data loading functions."""

import os
from typing import Tuple, Union

import numpy as np

__author__ = ["MatthewMiddlehurst"]
__all__ = [
    "load_from_ts_file",
    "load_minimal_chinatown",
    "load_unequal_minimal_chinatown",
    "load_equal_minimal_japanese_vowels",
    "load_minimal_japanese_vowels",
    "load_minimal_gas_prices",
    "load_unequal_minimal_gas_prices",
]


def load_from_ts_file(
    file_path: str,
    replace_missing_vals_with: Union[str, int, float] = "NaN",
    X_dtype: Union[None, type] = None,
    y_dtype: Union[None, type] = None,
) -> Union[np.ndarray, list, Tuple[np.ndarray, np.ndarray], Tuple[list, np.ndarray]]:
    """Load data from a .ts file into a 3D numpy array or list of 2D numpy arrays.

    If the data to be loaded is equal length, a 3D numpy array will be returned. If the
    data is unequal length, a list of 2D numpy arrays will be returned. If labels are
    present, they will be returned as well as the data.

    The only mandatory tags in the loaded file are one of @targetlabels or @classlabels.
    Other details can be inferred, though some error checking will be done if they are
    present.

    Parameters
    ----------
    file_path: str
        The full pathname of the .ts file to read.
    replace_missing_vals_with: str, int or float, default="NaN"
        The value that missing values reprented by '?' in the text file should be
        replaced with.
    X_dtype: type or None, default=None
        The data type of the loaded data X.
    y_dtype: type or None, default=None
        The data type of the loaded labels y if present.

    Returns
    -------
    X: np.ndarray or list
        The data from the file. If the data is equal length, a 3D numpy array will be
        returned, else a list of 2D numpy arrays.
    y: np.ndarray
        If labels are present in the file, a numpy array containing the label values
        will also be returned.

    Examples
    --------
    >>> from tsml.datasets import load_from_ts_file
    >>> path = (
    ... "MinimalChinatown/MinimalChinatown_TRAIN.ts"
    ...  if os.path.exists("MinimalChinatown/MinimalChinatown_TRAIN.ts") else
    ... "tsml/datasets/MinimalChinatown/MinimalChinatown_TRAIN.ts"
    ... )
    >>> X, y = load_from_ts_file(path)
    """
    # Initialize flags and variables used when parsing the file
    timestamps = False
    missing = False
    univariate = True
    dimensions = -1
    equallength = True
    serieslength = -1
    targetlabel = False
    classlabel = False

    timestamps_tag = False
    missing_tag = False
    univariate_tag = False
    dimensions_tag = False
    equallength_tag = False
    serieslength_tag = False
    targetlabels_tag = False
    classlabels_tag = False

    data_started = False

    # Read the file, stripping white space from start/end of line, converting to lower
    # case and removing blank lines
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip().lower() for line in file]
        lines = [line for line in lines if line]

    # Parse the file
    for i, line in enumerate(lines):
        # Check if this line contains metadata
        if line.startswith("@") and not line.startswith("@data"):
            if data_started:
                raise IOError(
                    "Invalid .ts file. Metadata must come before the @data tag."
                )

            if line.startswith("@problemname"):
                tokens = line.split(" ")
                if len(tokens) == 1:
                    raise IOError(
                        "Invalid .ts file. @problemname tag requires str value "
                        "(the problems name)."
                    )
            elif line.startswith("@timestamps"):
                tokens = line.split(" ")
                if len(tokens) != 2:
                    raise IOError(
                        "Invalid .ts file. @timestamps tag requires a bool value."
                    )
                elif tokens[1] == "true":
                    timestamps = True
                elif tokens[1] == "false":
                    timestamps = False
                else:
                    raise IOError(
                        "Invalid .ts file. @timestamps tag requires a bool value."
                    )

                timestamps_tag = True
            elif line.startswith("@missing"):
                tokens = line.split(" ")
                if len(tokens) != 2:
                    raise IOError(
                        "Invalid .ts file. @missing tag requires a bool value."
                    )
                elif tokens[1] == "true":
                    missing = True
                elif tokens[1] == "false":
                    missing = False
                else:
                    raise IOError(
                        "Invalid .ts file. @missing tag requires a bool value."
                    )

                missing_tag = True
            elif line.startswith("@univariate"):
                tokens = line.split(" ")
                if len(tokens) != 2:
                    raise IOError(
                        "Invalid .ts file. @univariate tag requires a bool value."
                    )
                elif tokens[1] == "true":
                    univariate = True
                elif tokens[1] == "false":
                    univariate = False
                else:
                    raise IOError(
                        "Invalid .ts file. @univariate tag requires a bool value."
                    )

                univariate_tag = True
            elif line.startswith("@dimension"):
                tokens = line.split(" ")
                if len(tokens) != 2:
                    raise IOError(
                        "Invalid .ts file. @dimension tag requires a int value "
                        "(the number of channels for the problem)."
                    )

                try:
                    dimensions = int(tokens[1])
                except ValueError:
                    raise IOError(
                        "Invalid .ts file. @dimension tag requires a int value "
                        "(the number of channels for the problem)."
                    )

                dimensions_tag = True
            elif line.startswith("@equallength"):
                tokens = line.split(" ")
                if len(tokens) != 2:
                    raise IOError(
                        "Invalid .ts file. @equallength tag requires a bool value."
                    )
                elif tokens[1] == "true":
                    equallength = True
                elif tokens[1] == "false":
                    equallength = False
                else:
                    raise IOError(
                        "Invalid .ts file. @equallength tag requires a bool value."
                    )

                equallength_tag = True
            elif line.startswith("@serieslength"):
                tokens = line.split(" ")
                if len(tokens) != 2:
                    raise IOError(
                        "Invalid .ts file. @serieslength tag requires a int value "
                        "(the series length for the problem)."
                    )

                try:
                    serieslength = int(tokens[1])
                except ValueError:
                    raise IOError(
                        "Invalid .ts file. @serieslength tag requires a int value "
                        "(the series length for the problem)."
                    )

                serieslength_tag = True
            elif line.startswith("@targetlabel"):
                if classlabels_tag:
                    raise IOError(
                        "Invalid .ts file. @targetlabel tag cannot be used with "
                        "@classlabel tag."
                    )

                tokens = line.split(" ")
                token_len = len(tokens)
                if token_len == 1:
                    raise IOError(
                        "Invalid .ts file. @targetlabel tag requires a bool value."
                    )
                if tokens[1] == "true":
                    targetlabel = True
                elif tokens[1] == "false":
                    targetlabel = False
                else:
                    raise IOError(
                        "Invalid .ts file. @targetlabel tag requires a bool value."
                    )

                if token_len > 2:
                    raise IOError(
                        f"Invalid .ts file. @targetlabel tag should not be accompanied "
                        f"with info apart from True/False, but found {tokens}."
                    )

                targetlabels_tag = True
            elif line.startswith("@classlabel"):
                if targetlabels_tag:
                    raise IOError(
                        "Invalid .ts file. @classlabel tag cannot be used with "
                        "@targetlabel tag."
                    )

                tokens = line.split(" ")
                token_len = len(tokens)
                if token_len == 1:
                    raise IOError(
                        "Invalid .ts file. @classlabel tag requires a bool value."
                    )
                if tokens[1] == "true":
                    classlabel = True
                elif tokens[1] == "false":
                    classlabel = False
                else:
                    raise IOError(
                        "Invalid .ts file. @classlabel tag requires a bool value."
                    )

                if not classlabel and token_len > 2:
                    raise IOError(
                        f"Invalid .ts file. @classlabel tag should not be accompanied "
                        f"with additional info when False, but found {tokens}."
                    )

                elif classlabel and token_len == 2:
                    raise IOError(
                        "Invalid .ts file. @classlabel tag is true but no Class values "
                        "are supplied."
                    )

                class_label_list = [token.strip() for token in tokens[2:]]

                classlabels_tag = True
            else:
                IOError(f"Invalid .ts file. Found unknown tag {line}.")

        # Check if this line contains the start of data
        elif line.startswith("@data"):
            if line != "@data":
                raise IOError(
                    "Invalid .ts file. @data tag should not have an associated value."
                )
            else:
                data_started = True
                data_start_line = i + 1

            if len(lines) == data_start_line + 1:
                raise IOError(
                    "Invalid .ts file. A @data tag is present but no subsequent data "
                    "is present."
                )
            else:
                first_line = lines[data_start_line].split(":")

            # One of the label tags is required
            if targetlabels_tag:
                has_labels = targetlabel
            elif classlabels_tag:
                has_labels = classlabel
            else:
                raise IOError(
                    "Unable to read .ts file. A @classlabel or @targetlabel tag is "
                    "required."
                )

            # Assume equal length if no tag.
            if not equallength_tag:
                equallength = True

            n_instances = len(lines) - data_start_line
            data_dims = len(first_line) - 1 if has_labels else len(first_line)
            first_line = first_line[0].split(",")
            data_length = len(first_line)

            # Do some verification on remaining tags

            has_timestamps = (
                True
                if first_line[0].startswith("(") and first_line[0].endswith(")")
                else False
            )
            if (
                not timestamps_tag or (timestamps_tag and not timestamps)
            ) and has_timestamps:
                raise IOError(
                    "Value mismatch in .ts file. @timestamps tag is missing or False "
                    "but data has timestamps. Timestamps are currently not supported."
                )
            elif has_timestamps:
                raise IOError(
                    "Unable to read .ts file. Timestamps are currently not supported."
                )

            replace_missing = True if missing_tag and not missing else False

            if (
                not univariate_tag or (univariate_tag and univariate)
            ) and data_dims > 1:
                raise IOError(
                    "Value mismatch in .ts file. @univariate tag is missing or True "
                    "but data has more than one channel."
                )

            if dimensions_tag and dimensions != data_dims:
                raise IOError(
                    f"Value mismatch in .ts file. @dimensions tag value {dimensions} "
                    f"and read number of channels {data_dims} do not match."
                )

            if serieslength_tag and serieslength != data_length:
                raise IOError(
                    f"Value mismatch in .ts file. @serieslength tag value "
                    f"{serieslength} and read series length {data_length} do not match."
                )

            if equallength:
                X = np.zeros((n_instances, data_dims, data_length), dtype=X_dtype)
            else:
                X = [None] * n_instances

            if has_labels:
                y = np.zeros(n_instances, dtype=y_dtype)

        # If the @data tag has been found then metadata has been parsed and data can
        # be loaded
        elif data_started:
            data_idx = i - data_start_line

            # Replace any missing values with the value specified
            if replace_missing:
                line = line.replace("?", replace_missing_vals_with)

            line = line.split(":")

            # Does not support different number of channels
            read_dims = len(line) - 1 if has_labels else len(line)
            if read_dims != data_dims:
                raise IOError(
                    "Unable to read .ts file. Inconsistent number of channels."
                    f"Expected {data_dims} but read {read_dims} on line {data_idx}."
                )

            dimensions = line[:data_dims]
            split = dimensions[0].strip().split(",")
            length = len(split)
            if equallength and length != data_length:
                raise IOError(
                    "Unable to read .ts file. Inconsistent number of channels."
                    f"Expected {data_dims} but read {read_dims} on line {data_idx}."
                )

            # Process the data for each channel
            series = np.zeros((data_dims, length), dtype=X_dtype)
            series[0, :] = split
            for n in range(1, data_dims):
                series[n, :] = dimensions[n].strip().split(",")

            X[data_idx] = series

            if has_labels:
                label = line[-1].strip()

                if classlabel and label not in class_label_list:
                    raise IOError(
                        "Unable to read .ts file. Read unknown class value. Expected "
                        f"{class_label_list} but read {label} on line {data_idx}."
                    )

                y[data_idx] = label

        # Other lines must be comments are start with a % or #
        elif not line.startswith("%") and not line.startswith("#"):
            raise IOError(f"Invalid .ts file. Unable to parse line: {line}.")

    if not data_started:
        raise IOError(
            "Invalid .ts file. File contains metadata but no @data tag to signal the "
            "start of data."
        )

    if has_labels:
        return X, y
    else:
        return X


def load_minimal_chinatown(
    split: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load MinimalChinatown time series classification problem.

    This is an equal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification. It loads a two class classification problem with 20 cases
    for both the train and test split and a series length of 24.

    For the full dataset see
    http://timeseriesclassification.com/description.php?Dataset=Chinatown

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray
        The time series data for the problem of shape (20,1,24).
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X, y = load_minimal_chinatown()
    """
    return _load_provided_dataset("MinimalChinatown", split)


def load_unequal_minimal_chinatown(
    split: Union[None, str] = None
) -> Tuple[list, np.ndarray]:
    """Load UnequalMinimalChinatown time series classification problem.

    This is an unequal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification. Parts of the original series have been randomly removed. It
    loads a two class classification problem with 20 cases for both the train and test
    split.

    For the full dataset see
    http://timeseriesclassification.com/description.php?Dataset=Chinatown

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: list of np.ndarray
        The time series data for the problem in a list of size 20 containing 2D
        ndarrays.
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml.datasets import load_unequal_minimal_chinatown
    >>> X, y = load_unequal_minimal_chinatown()
    """
    return _load_provided_dataset("UnequalMinimalChinatown", split)


def load_equal_minimal_japanese_vowels(
    split: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the EqualMinimalJapaneseVowels time series classification problem.

    This is an equal length multivariate time series classification problem. It is a
    stripped down version of the JapaneseVowels problem that is used in correctness
    tests for classification. It has been altered so all series are equal length. It
    loads a nine class classification problem with 20 cases for both the train and test
    split, 12 channels and a series length of 25.

    For the full dataset see
    http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray
        The time series data for the problem of shape (20,12,25).
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml.datasets import load_equal_minimal_japanese_vowels
    >>> X, y = load_equal_minimal_japanese_vowels()
    """
    return _load_provided_dataset("EqualMinimalJapaneseVowels", split)


def load_minimal_japanese_vowels(
    split: Union[None, str] = None
) -> Tuple[list, np.ndarray]:
    """Load the MinimalJapaneseVowels time series classification problem.

    This is an unequal length multivariate time series classification problem. It is a
    stripped down version of the JapaneseVowels problem that is used in correctness
    tests for classification. It loads a nine class classification problem with 20 cases
    for both the train and test split and 12 channels.

    For the full dataset see
    http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: list of np.ndarray
        The time series data for the problem in a list of size 20 containing 2D
        ndarrays.
    y: np.ndarray
        The class labels for each case in X.

    Examples
    --------
    >>> from tsml.datasets import load_minimal_japanese_vowels
    >>> X, y = load_minimal_japanese_vowels()
    """
    return _load_provided_dataset("MinimalJapaneseVowels", split)


def load_minimal_gas_prices(
    split: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the MinimalGasPrices time series extrinsic regression problem.

    This is an equal length univariate time series regression problem. It is a
    stripped down version of the GasPricesSentiment problem that is used in correctness
    tests for regression. It loads a regression problem with 20 cases for both the train
    and test split and a series length of 20.

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray
        The time series data for the problem of shape (20,1,20).
    y: np.ndarray
        The labels for each case in X.

    Examples
    --------
    >>> from tsml.datasets import load_minimal_gas_prices
    >>> X, y = load_minimal_gas_prices()
    """
    return _load_provided_dataset("MinimalGasPrices", split)


def load_unequal_minimal_gas_prices(
    split: Union[None, str] = None
) -> Tuple[list, np.ndarray]:
    """Load the UnequalMinimalGasPrices time series extrinsic regression problem.

    This is an unequal length univariate time series regression problem. It is a
    stripped down version of the GasPricesSentiment problem that is used in correctness
    tests for regression. Parts of the original series have been randomly removed. It
    loads a regression problem with 20 cases for both the train  and test split.

    Parameters
    ----------
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: list of np.ndarray
        The time series data for the problem in a list of size 20 containing 2D
        ndarrays.
    y: np.ndarray
        The labels for each case in X.

    Examples
    --------
    >>> from tsml.datasets import load_unequal_minimal_gas_prices
    >>> X, y = load_unequal_minimal_gas_prices()
    """
    return _load_provided_dataset("UnequalMinimalGasPrices", split)


def _load_provided_dataset(
    name: str,
    split: Union[None, str] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[list, np.ndarray]]:
    """Load baked in time series datasets.

    Loads data from the provided tsml dataset files only.

    Parameters
    ----------
    name : str
        File name to load from.
    split: "TRAIN", "TEST" or None, default=None
        Whether to load the train or test instances of the problem. If None, loads
        both train and test instances (in a single container).

    Returns
    -------
    X: np.ndarray or list of np.ndarray
        The time series data for the problem in a 3D array if the data is equal length
        or a list containing 2D arrays if it is unequal.
    y: np.ndarray
        The labels for each case in X.
    """
    if isinstance(split, str):
        split = split.upper()

    if split in ("TRAIN", "TEST"):
        fname = name + "_" + split + ".ts"
        path = os.path.join(os.path.dirname(__file__), name, fname)
        X, y = load_from_ts_file(path)
    # if split is None, load both train and test set
    elif split is None:
        fname = name + "_TRAIN.ts"
        path = os.path.join(os.path.dirname(__file__), name, fname)
        X_train, y_train = load_from_ts_file(path)

        fname = name + "_TEST.ts"
        path = os.path.join(os.path.dirname(__file__), name, fname)
        X_test, y_test = load_from_ts_file(path)

        X = (
            X_train + X_test
            if isinstance(X_train, list)
            else np.concatenate([X_train, X_test])
        )
        y = np.concatenate([y_train, y_test])
    else:
        raise ValueError("Invalid `split` value =", split)

    return X, y
