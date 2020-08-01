# -*- coding: utf-8 -*-
"""ContentWMFDataLoader class."""
from typing import (
    List,
    Tuple,
)

import numpy as np

from content_wmf.content_wmf import log_surplus_confidence_matrix
from dataloader import MatrixDataLoader
from dataloader_helper import DataLoaderHelper


__author__ = 'haochun.fu'
__date__ = '2020-07-06'


class ContentWMFDataLoader(MatrixDataLoader):
    """Data loader for content WMF."""


    def load_train_data(self) -> dict:
        """Load training data.

        Returns:
            dict: Training data and data related to training data.
                {
                  'interactions': "np.float32 coo_matrix of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     matrix",
                  'item_id_mapping': "[item_id, ...], index of item_id is id in
                                     matrix",
                  'item_features': "np.float64 array of shape
                                   [n_items, item_dim]". If not use, is None.
                }
        """
        return self._load_train_data_by_date_range(self._get_train_range())

    def _get_item_features(
        self,
        item_ids: List[int]
    ) -> Tuple[np.ndarray, List[int]]:
        """Get item features.

        Args:
            item_ids (list): List of item ids.

        Returns:
            tuple: Item features and item ids in item features.
        """
        features, ids = super()._get_item_features(item_ids)

        return np.asarray(features, dtype=np.float64), ids

    def _load_train_data_by_date_range(self, date_range: Tuple[str, str]):
        """Load training data by date range.

        Args:
            date_range (tuple): Start and end dates.

        Returns:
            dict: Training data and data related to training data.
                {
                  'interactions': "np.float32 coo_matrix of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     matrix",
                  'item_id_mapping': "[item_id, ...], index of item_id is id in
                                     matrix",
                  'item_features': "np.float64 array of shape
                                   [n_items, item_dim]". If not use, is None.,
                }
        """
        data = self._get_interactive_data_by_range(date_range)

        user_ids, item_ids = self._get_ids_from_interactive_data(data)

        item_features = None
        param = self._param['item']
        if param.get('vector') or \
            param.get('genres') or \
            param.get('genome_tags'):
            item_features, item_ids = self._get_item_features(item_ids)

        interactions = DataLoaderHelper.construct_matrix(
            data,
            user_ids,
            item_ids,
            DataLoaderHelper.MATRIX_TYPE_CSR)

        log = self._param['feature']['log']
        if log:
            interactions = log_surplus_confidence_matrix(
                interactions,
                log['alpha'],
                log['epsilon'])

        return {
            'interactions': interactions,
            'user_id_mapping': user_ids,
            'item_id_mapping': item_ids,
            'item_features': item_features,
        }
