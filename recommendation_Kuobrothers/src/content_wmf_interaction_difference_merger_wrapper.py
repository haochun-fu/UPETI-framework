# -*- coding: utf-8 -*-
"""ContentWMFInteractionDifferenceMergerWrapper class."""
from typing import List

import numpy as np
from scipy.sparse import csr_matrix

from dataloader_content_wmf_interaction_difference import \
    ContentWMFInteractionDifferenceDataLoader
from dataloader_helper import DataLoaderHelper
from interaction_difference_wrapper import \
    MatrixMergerInteractionDifferenceWrapper
from content_wmf_wrapper import ContentWMFWrapper


__author__ = 'haochun.fu'
__date__ = '2020-02-25'


class ContentWMFInteractionDifferenceMergerWrapper(
    MatrixMergerInteractionDifferenceWrapper):
    """Wrapper of ContentWMF interaction difference merger."""


    def get_dataloader_class(self) -> ContentWMFInteractionDifferenceDataLoader:
        """Get dataloader class.

        Returns:
            ContentWMFInteractionDifferenceDataLoader: 
                ContentWMFInteractionDifferenceDataLoader class.
        """
        return ContentWMFInteractionDifferenceDataLoader

    def get_model_wrapper_class(self) -> ContentWMFWrapper:
        """Get model wrapper class.

        Return:
            ContentWMFWrapper: ContentWMF wrapper class.
        """
        return ContentWMFWrapper 

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
                  'item_features': "np.float64 array of shape
                                   [n_items, item_dim]". If not use, is None.
                }
        """
        ret = super()._merge_split_model_data(
            params['scale'],
            params['filter'],
            DataLoaderHelper.MATRIX_TYPE_CSR,
            split_model_data)

        if ret['item_features']:
            ret['item_features'] = np.asarray(
                ret['item_features'], dtype=np.float64)

        return ret
