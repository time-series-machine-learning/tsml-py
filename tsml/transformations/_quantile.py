# from typing import List, Union
#
# import numpy as np
# from sklearn.base import TransformerMixin
#
# from tsml.base import BaseTimeSeriesEstimator
#
#
# class QuantileTransformer(TransformerMixin, BaseTimeSeriesEstimator):
#     """QuantileTransformer"""
#
#     def __init__(
#         self,
#         divisor=4,
#         subtract_mean=False,
#     ):
#         self.divisor = divisor
#         self.subtract_mean = subtract_mean
#
#         super(QuantileTransformer).__init__()
#
#     def fit(
#         self, X: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, None] = None
#     ) -> object:
#         """Unused. Validates X."""
#         self._validate_data(X=X)
#         return self
#
#     def transform(
#         self, X: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, None] = None
#     ) -> np.ndarray:
#         """Transform input cases in X.
#
#         Parameters
#         ----------
#         X : 3D np.ndarray of shape (n_instances, n_channels, n_timepoints)
#             The training data.
#         y : 1D np.ndarray of shape (n_instances)
#             The class labels for fitting, indices correspond to instance indices in X
#
#         Returns
#         -------
#         X_t : 2D np.ndarray of shape (n_instances, n_features)
#             Transformed data.
#         """
#         X = self._validate_data(X=X, reset=False)
#         X = self._convert_X(X)
#
#         num_quantiles = 1 + (X.shape[2] - 1) // self.divisor
#         if num_quantiles == 1:
#             return X.quantile(torch.tensor([0.5]), dim=-1).permute(1, 2, 0)
#         else:
#             quantiles = X.quantile(torch.linspace(0, 1, num_quantiles), dim=-1).permute(
#                 1, 2, 0
#             )
#             quantiles[..., 1::2] = quantiles[..., 1::2] - X.mean(-1, keepdims=True)
#             return quantiles
