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
    """Load data from a .ts file into a Pandas DataFrame.

    req equallength and (targetlabels or classlabels)

    Parameters
    ----------
    file_path: str
        The full pathname of the .ts file to read.
    replace_missing_vals_with: None, int or float, default=None
       The value that missing values in the text file should be replaced
       with prior to parsing.

    Returns
    -------
    DataFrame (default) or ndarray (i
        If return_separate_X_and_y then a tuple containing a DataFrame and a
        numpy array containing the relevant time-series and corresponding
        class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing
        all time-series and (if relevant) a column "class_vals" the
        associated class values.
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
                        "Invalid .ts file. @problemname tag requires str value (the problems name)."
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
                        "Invalid .ts file. @dimension tag requires a int value (the number of dimensions for the problem)."
                    )

                try:
                    dimensions = int(tokens[1])
                except ValueError:
                    raise IOError(
                        "Invalid .ts file. @dimension tag requires a int value (the number of dimensions for the problem)."
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
                        "Invalid .ts file. @serieslength tag requires a int value (the number of dimensions for the problem)."
                    )

                try:
                    serieslength = int(tokens[1])
                except ValueError:
                    raise IOError(
                        "Invalid .ts file. @serieslength tag requires a int value (the number of dimensions for the problem)."
                    )

                serieslength_tag = True
            elif line.startswith("@targetlabel"):
                if classlabels_tag:
                    raise IOError(
                        "Invalid .ts file. @targetlabel tag cannot be used with @classlabel tag."
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
                        f"Invalid .ts file. @targetlabel tag should not be accompanied with info apart from True/False, but found {tokens}."
                    )

                targetlabels_tag = True
            elif line.startswith("@classlabel"):
                if targetlabels_tag:
                    raise IOError(
                        "Invalid .ts file. @classlabel tag cannot be used with @targetlabel tag."
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
                        f"Invalid .ts file. @classlabel tag should not be accompanied with additional info when False, but found {tokens}."
                    )

                elif classlabel and token_len == 2:
                    raise IOError(
                        "Invalid .ts file. @classlabel tag is true but no Class values are supplied."
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
                    "Invalid .ts file. A @data tag is present but no subsequent data is present."
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
                    "Unable to read .ts file. A @classlabel or @targetlabel tag is required."
                )

            # Equal length tag is required.
            if not equallength_tag:
                raise IOError(
                    "Unable to read .ts file. The @equallength tag is required."
                )

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
                    "Value mismatch in .ts file. @timestamps tag is missing or False but data has timestamps. Timestamps are currently not supported."
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
                    "Value mismatch in .ts file. @univariate tag is missing or True but data has more than one dimension."
                )

            if dimensions_tag and dimensions != data_dims:
                raise IOError(
                    f"Value mismatch in .ts file. @dimensions tag value {dimensions} and read number of dimensions {data_dims} do not match."
                )

            if serieslength_tag and serieslength != data_length:
                raise IOError(
                    f"Value mismatch in .ts file. @serieslength tag value {serieslength} and read series length {data_length} do not match."
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

            # Does not support different number of dimensions
            read_dims = len(line) - 1 if has_labels else len(line)
            if read_dims != data_dims:
                raise IOError(
                    "Unable to read .ts file. Inconsistent number of dimensions."
                    f"Expected {data_dims} but read {read_dims} on line {data_idx}."
                )

            dimensions = line[:data_dims]
            if not equallength:
                data_length = len(dimensions[0].strip().split(","))

            # Process the data for each dimension
            series = np.zeros((data_dims, data_length), dtype=X_dtype)
            for i in range(data_dims):
                series[i, :] = dimensions[i].strip().split(",")

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
            "Invalid .ts file. File contains metadata but no @data tag to signal the start of data."
        )

    if has_labels:
        return X, y
    else:
        return X


def load_minimal_chinatown(
    split: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MinimalChinatown data.

    This is an equal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification. It loads a two class classification problem with number of
    cases, n, where n = 42 (if split is None) or 20/22 (if split is "train"/"test")
    of series length m = 24

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.
    return_type: valid Panel mtype str or None, optional (default=None="nested_univ")
        Memory data format specification to return X in, None = "nested_univ" type.
        str can be any supported sktime Panel mtype,
            for list of mtypes, see datatypes.MTYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            "nested_univ: nested pd.DataFrame, pd.Series in cells
            "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Examples
    --------
    >>> from tsml.datasets import load_minimal_chinatown
    >>> X, y = load_minimal_chinatown()

    Details
    -------
    This is the Chinatown problem with a smaller test set, useful for rapid tests.
    Dimensionality:     univariate
    Series length:      24
    Train cases:        20
    Test cases:         20 (full dataset has 345)
    Number of classes:  2

     See
    http://timeseriesclassification.com/description.php?Dataset=Chinatown
    for the full dataset
    """
    return _load_provided_dataset("MinimalChinatown", split)


def load_unequal_minimal_chinatown(
    split: Union[None, str] = None
) -> Tuple[list, np.ndarray]:
    """
    Load MinimalChinatown data.

    This is an equal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification. It loads a two class classification problem with number of
    cases, n, where n = 42 (if split is None) or 20/22 (if split is "train"/"test")
    of series length m = 24

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).

    Returns
    -------
    X:  The time series data for the problem. If return_type is either
        "numpy2d"/"numpyflat", it returns 2D numpy array of shape (n,m), if "numpy3d" it
        returns 3D numpy array of shape (n,1,m) and if "nested_univ" or None it returns
        a nested pandas DataFrame of shape (n,1), where each cell is a pd.Series of
        length m.
    y: (optional) numpy array shape (n,1). The class labels for each case in X.
        If return_X_y is False, y is appended to X.

    Examples
    --------
    >>> from tsml.datasets import load_unequal_minimal_chinatown
    >>> X, y = load_unequal_minimal_chinatown()

    Details
    -------
    This is the Chinatown problem with a smaller test set, useful for rapid tests.
    Dimensionality:     univariate
    Series length:      24
    Train cases:        20
    Test cases:         20 (full dataset has 345)
    Number of classes:  2

     See
    http://timeseriesclassification.com/description.php?Dataset=Chinatown
    for the full dataset
    """
    return _load_provided_dataset("UnequalMinimalChinatown", split)


def load_equal_minimal_japanese_vowels(
    split: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load the JapaneseVowels time series classification problem.

    Example of a multivariate problem with unequal length series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from tsml.datasets import load_equal_minimal_japanese_vowels
    >>> X, y = load_equal_minimal_japanese_vowels()

    Notes
    -----
    Dimensionality:     multivariate, 12
    Series length:      7-29
    Train cases:        270
    Test cases:         370
    Number of classes:  9

    A UCI Archive dataset. 9 Japanese-male speakers were recorded saying
    the vowels 'a' and 'e'. A '12-degree
    linear prediction analysis' is applied to the raw recordings to
    obtain time-series with 12 dimensions and series lengths between 7 and 29.
    The classification task is to predict the speaker. Therefore,
    each instance is a transformed utterance,
    12*29 values with a single class label attached, [1...9]. The given
    training set is comprised of 30
    utterances for each speaker, however the test set has a varied
    distribution based on external factors of
    timing and experimental availability, between 24 and 88 instances per
    speaker. Reference: M. Kudo, J. Toyama
    and M. Shimbo. (1999). "Multidimensional Curve Classification Using
    Passing-Through Regions". Pattern
    Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.
    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=JapaneseVowels
    """
    return _load_provided_dataset("EqualMinimalJapaneseVowels", split)


def load_minimal_japanese_vowels(
    split: Union[None, str] = None
) -> Tuple[list, np.ndarray]:
    """Load the JapaneseVowels time series classification problem.

    Example of a multivariate problem with unequal length series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The class labels for each case in X

    Examples
    --------
    >>> from tsml.datasets import load_minimal_japanese_vowels
    >>> X, y = load_minimal_japanese_vowels()

    Notes
    -----
    Dimensionality:     multivariate, 12
    Series length:      7-29
    Train cases:        270
    Test cases:         370
    Number of classes:  9

    A UCI Archive dataset. 9 Japanese-male speakers were recorded saying
    the vowels 'a' and 'e'. A '12-degree
    linear prediction analysis' is applied to the raw recordings to
    obtain time-series with 12 dimensions and series lengths between 7 and 29.
    The classification task is to predict the speaker. Therefore,
    each instance is a transformed utterance,
    12*29 values with a single class label attached, [1...9]. The given
    training set is comprised of 30
    utterances for each speaker, however the test set has a varied
    distribution based on external factors of
    timing and experimental availability, between 24 and 88 instances per
    speaker. Reference: M. Kudo, J. Toyama
    and M. Shimbo. (1999). "Multidimensional Curve Classification Using
    Passing-Through Regions". Pattern
    Recognition Letters, Vol. 20, No. 11--13, pages 1103--1111.
    Dataset details: http://timeseriesclassification.com/description.php
    ?Dataset=JapaneseVowels
    """
    return _load_provided_dataset("MinimalJapaneseVowels", split)


def load_minimal_gas_prices(
    split: Union[None, str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset of last three months confirmed covid cases.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default, it loads both.

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The regression values for each case in X

    Examples
    --------
    >>> from tsml.datasets import load_minimal_gas_prices
    >>> X, y = load_minimal_gas_prices()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      84
    Train cases:        140
    Test cases:         61
    Number of classes:  -

    The goal of this dataset is to predict COVID-19's death rate on 1st April 2020 for
    each country using daily confirmed cases for the last three months. This dataset
    contains 201 time series with no missing values, where each time series is
    the daily confirmed cases for a country.
    The data was obtained from WHO's COVID-19 database.
    Please refer to https://covid19.who.int/ for more details

    Dataset details: https://zenodo.org/record/3902690#.Yy1z_HZBxEY
    =Covid3Month
    """
    return _load_provided_dataset("MinimalGasPrices", split)


def load_unequal_minimal_gas_prices(
    split: Union[None, str] = None
) -> Tuple[list, np.ndarray]:
    """Load dataset of last three months confirmed covid cases.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default, it loads both.

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The regression values for each case in X

    Examples
    --------
    >>> from tsml.datasets import load_unequal_minimal_gas_prices
    >>> X, y = load_unequal_minimal_gas_prices()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      84
    Train cases:        140
    Test cases:         61
    Number of classes:  -

    The goal of this dataset is to predict COVID-19's death rate on 1st April 2020 for
    each country using daily confirmed cases for the last three months. This dataset
    contains 201 time series with no missing values, where each time series is
    the daily confirmed cases for a country.
    The data was obtained from WHO's COVID-19 database.
    Please refer to https://covid19.who.int/ for more details

    Dataset details: https://zenodo.org/record/3902690#.Yy1z_HZBxEY
    =Covid3Month
    """
    return _load_provided_dataset("UnequalMinimalGasPrices", split)


def _load_provided_dataset(
    name: str,
    split: Union[None, str] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[list, np.ndarray]]:
    """Load baked in time series classification datasets (helper function).

    Loads data from the provided files from sktime/datasets/data only.

    Parameters
    ----------
    name : string, file name to load from
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem.
        By default it loads both train and test instances (in a single container).

    Returns
    -------
    X: sktime data container, following mtype specification `return_type`
        The time series data for the problem, with n instances
    y: 1D numpy array of length n, only returned if return_X_y if True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.
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
