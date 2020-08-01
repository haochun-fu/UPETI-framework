# -*- coding: utf-8 -*-
"""LightFMDataLoader class."""
from typing import (
    List,
    Tuple,
)

from dataloader import MatrixDataLoader
from scipy.sparse import csr_matrix

from dataloader_helper import DataLoaderHelper


__author__ = 'haochun.fu'
__date__ = '2019-10-08'


class LightFMDataLoader(MatrixDataLoader):
    """Data loader for LightFM."""


    def load_train_data(self) -> dict:
        """Load training data.

        Returns:
            dict: Training data and data related to training data.
                {
                  'interactions': "np.float32 coo_matrix of shape
                                   [n_users, n_items]).",
                   'user_id_mapping': "[user_id, ...], index of user_id is id in
                                      matrix",
                   'contract_id_mapping': "[contract_id, ...], index of
                                          contract_id is id in matrix",
                   'item_features': "np.float64 csr_matrix of shape
                                    [n_items, item_dim]",
                }
        """
        return self._load_train_data_by_date_range(self._get_train_range())

    def _get_item_features(
        self,
        contract_ids: List[int]
    ) -> Tuple[csr_matrix, List[int]]:
        """Get item features.

        Args:
            contract_ids (list): List of contract ids.

        Returns:
            tuple: Item features and contract ids in item features.
        """
        features, ids = super()._get_item_features(contract_ids)
        features = csr_matrix(features)
        return features, ids

    def _load_train_data_by_date_range(self, date_range: Tuple[str, str]):
        """Load training data by date range.

        Args:
            date_range (tuple): Start and end dates.

        Returns:
            {
              'interactions': "np.float32 coo_matrix of shape
                              [n_users, n_items]).",
              'user_id_mapping': "[user_id, ...], index of user_id is id in
                                 matrix",
              'contract_id_mapping': "[contract_id, ...], index of
                                     contract_id is id in matrix",
              'item_features': "np.float64 csr_matrix of shape
                               [n_items, item_dim]",
            }
        """
        data = self._get_interactive_data_by_range(date_range)

        user_ids, contract_ids = self._get_ids_from_interactive_data(data)

        item_features = None
        if self._param.get('product_vector') or \
            ('product_info' in self._param and \
             list(self._param['product_info'].values()).count(0) != \
                len(self._param['product_info'])):
            item_features, contract_ids = self._get_item_features(contract_ids)

        interactions = DataLoaderHelper.construct_matrix(
            data,
            user_ids,
            contract_ids,
            DataLoaderHelper.MATRIX_TYPE_COO)

        return {
            'interactions': interactions,
            'user_id_mapping': user_ids,
            'contract_id_mapping': contract_ids,
            'item_features': item_features,
        }
