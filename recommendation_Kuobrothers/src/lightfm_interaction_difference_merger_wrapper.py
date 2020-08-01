# -*- coding: utf-8 -*-
"""LightFMInteractionDifferenceMergerWrapper class."""
from typing import List

from scipy.sparse import csr_matrix

from dataloader_lightfm_interaction_difference import \
    LightFMInteractionDifferenceDataLoader
from interaction_difference_wrapper import \
    MatrixMergerInteractionDifferenceWrapper
from lightfm_wrapper import LightFMWrapper

from comm_func import CommFunc
from dataloader_helper import DataLoaderHelper


__author__ = 'haochun.fu'
__date__ = '2020-02-22'


class LightFMInteractionDifferenceMergerWrapper(
    MatrixMergerInteractionDifferenceWrapper):
    """Wrapper of LightFM interaction difference merger."""


    def get_dataloader_class(self) -> LightFMInteractionDifferenceDataLoader:
        """Get dataloader class.

        Returns:
            LightFMInteractionDifferenceDataLoader: 
                LightFMInteractionDifferenceDataLoader class.
        """
        return LightFMInteractionDifferenceDataLoader

    def get_model_wrapper_class(self) -> LightFMWrapper:
        """Get model wrapper class.

        Return:
            LightFMWrapper: LightFM wrapper class.
        """
        return LightFMWrapper 

    def _merge_split_model_data(
        self,
        params: dict,
        split_model_data: List[dict]
    ) -> dict:
        """Merge data of split models.

        Args: 
            params (dict): Parameters.
            split_model_data (list): List of training data and data related to
                training data of split models.

        Returns:
            dict: Merged split model data.
                {
                  'interactions': "np.float32 coo_matrix of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     matrix",
                  'contract_id_mapping': "[contract_id, ...], index of
                                         contract_id is id in matrix",
                  'item_features': "np.float64 csr_matrix of shape
                                   [n_items, item_dim]"
                }
        """
        ret = super()._merge_split_model_data(
            params['scale'],
            params['filter'],
            DataLoaderHelper.MATRIX_TYPE_COO,
            split_model_data)

        if ret['item_features']:
            ret['item_features'] = csr_matrix(ret['item_features'])

        return ret
