# -*- coding: utf-8 -*-
"""LightFMInteractionDifferenceMergerV2DataLoader class."""
from collections import defaultdict

from dataloader_helper import DataLoaderHelper
from dataloader_lightfm_interaction_difference import \
    LightFMInteractionDifferenceDataLoader
from time_decay import DecayFactory


__author__ = 'haochun.fu'
__date__ = '2020-06-26'


class LightFMInteractionDifferenceMergerV2DataLoader(
    LightFMInteractionDifferenceDataLoader):
    """Data loader for LightFM interaction difference merger version 2."""


    def load_train_data(self) -> dict:
        """Load training data.

        Returns:
            list: Training data and data related to training data.
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
        data = {}

        decay = DecayFactory().createByName(
            **self._param['interaction_difference']['time_decay'])
        for no, interval in enumerate(super().load_train_data(), 1):
            u_mapping = interval['user_id_mapping']
            i_mapping = interval['contract_id_mapping']
            for u_idx, row in enumerate(interval['interactions'].toarray()):
                i_data = data.setdefault(u_mapping[u_idx], defaultdict(float))
                for i_idx, val in enumerate(row):
                    if val == 0:
                        continue

                    i_data[i_mapping[i_idx]] += decay.weight(no, val)

        user_ids, item_ids = self._get_ids_from_interactive_data(data)

        item_features = None
        if self._param.get('product_vector') or \
            ('product_info' in self._param and \
             list(self._param['product_info'].values()).count(0) != \
                len(self._param['product_info'])):
            item_features, item_ids = self._get_item_features(item_ids)

        interactions = DataLoaderHelper.construct_matrix(
            data,
            user_ids,
            item_ids,
            DataLoaderHelper.MATRIX_TYPE_COO)

        return {
            'interactions': interactions,
            'user_id_mapping': user_ids,
            'contract_id_mapping': item_ids,
            'item_features': item_features,
        }
