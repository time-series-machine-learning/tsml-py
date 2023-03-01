# -*- coding: utf-8 -*-
"""Utilities for discovering estimators in tsml."""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["all_estimators"]

import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path
from typing import List, Union

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils._testing import ignore_warnings

_PACKAGES_TO_IGNORE = {
    "tests",
}


def all_estimators(type_filter: Union[str, List[str]] = None):
    """Get a list of all estimators from `tsml`.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.

    Uses the `scikit-learn` 1.2.1 `all_estimators` function as a base.

    Parameters
    ----------
    type_filter : {"classifier", "regressor", "clusterer", "transformer"} \
            or list of such str, default=None
        Which kind of estimators should be returned. If None, no filter is
        applied and all estimators are returned.  Possible values are
        'classifier', 'regressor', 'clusterer' and 'transformer' to get
        estimators only of these specific types, or a list of these to
        get the estimators that fit at least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.

    Examples
    --------
    >>> from tsml.utils.discovery import all_estimators
    >>> estimators = all_estimators()
    >>> classifiers = all_estimators(type_filter="classifier")
    """

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    root = str(Path(__file__).parent.parent)  # tsml package

    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for _, module_name, _ in pkgutil.walk_packages(path=[root], prefix="tsml."):
            module_parts = module_name.split(".")
            if (
                any(part in _PACKAGES_TO_IGNORE for part in module_parts)
                or "._" in module_name
            ):
                continue

            module = import_module(module_name)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (
            issubclass(c[1], BaseEstimator)
            and c[1].__module__.startswith("tsml.")
            and c[0] != "BaseTimeSeriesEstimator"
        )
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {
            "classifier": ClassifierMixin,
            "regressor": RegressorMixin,
            "transformer": TransformerMixin,
            # accept both clusterer inputs
            "clusterer": ClusterMixin,
            "cluster": ClusterMixin,
        }
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'classifier', "
                "'regressor', 'transformer', 'cluster' or "
                f"None, got {repr(type_filter)}."
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))
