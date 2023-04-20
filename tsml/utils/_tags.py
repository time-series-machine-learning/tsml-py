# -*- coding: utf-8 -*-
"""Utilities for tsml tags."""

__author__ = ["MatthewMiddlehurst"]

from typing import Union

import numpy as np
from sklearn.base import BaseEstimator

_DEFAULT_TAGS = {
    # sklearn tags
    "non_deterministic": False,
    "X_types": ["3darray"],
    "no_validation": False,
    "allow_nan": False,
    "preserves_dtype": [np.float64],
    "requires_fit": True,
    "requires_y": False,
    "_skip_test": False,
    "_xfail_checks": False,
    # tsml tags
    "optional_dependency": False,
    "univariate_only": False,
    "equal_length_only": True,
}


def _safe_tags(
    estimator: BaseEstimator, key: Union[str, None] = None
) -> Union[dict, str]:
    """Safely get estimator tags.

    :class:`~sklearn.BaseEstimator` provides the estimator tags machinery.
    However, if an estimator does not inherit from this base class, we should
    fall-back to the default tags.

    For tsml built-in estimators, we should still rely on
    `self._get_tags()`. `_safe_tags(est)` should be used when we are not sure
    where `est` comes from: typically `_safe_tags(self.base_estimator)` where
    `self` is a meta-estimator, or in the common checks.

    Uses the `scikit-learn` 1.2.1 `_safe_tags` function as a base.

    Parameters
    ----------
    estimator : estimator object
        The estimator from which to get the tag.
    key : str, default=None
        Tag name to get. By default (`None`), all tags are returned.

    Returns
    -------
    tags : dict or tag value
        The estimator tags. A single value is returned if `key` is not None.
    """
    if hasattr(estimator, "_get_tags"):
        tags_provider = "_get_tags()"
        tags = estimator._get_tags()
    elif hasattr(estimator, "_more_tags"):
        tags_provider = "_more_tags()"
        tags = {**_DEFAULT_TAGS, **estimator._more_tags()}
    else:
        tags_provider = "_DEFAULT_TAGS"
        tags = _DEFAULT_TAGS

    if key is not None:
        if key not in tags:
            raise ValueError(
                f"The key {key} is not defined in {tags_provider} for the "
                f"class {estimator.__class__.__name__}."
            )
        return tags[key]
    return tags
