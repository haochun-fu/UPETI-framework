# -*- coding: utf-8 -*-
"""ContentWMFInteractionDifferenceDataLoader class."""
from typing import List

from content_wmf_dataloader import ContentWMFDataLoader


__author__ = 'haochun.fu'
__date__ = '2020-07-06'


class ContentWMFInteractionDifferenceDataLoader(ContentWMFDataLoader):
    """Data loader for content WMF interaction difference."""


    def load_train_data(self) -> List[dict]:
        """Load training data.

        Returns:
            list: List of training data and data related to training data.
                [{
                  'interactions': "np.float32 coo_matrix of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     matrix",
                  'item_id_mapping': "[item_id, ...], index of item_id is id in
                                     matrix",
                  'item_features': "np.float64 array of shape
                                   [n_items, item_dim]". If not use, is None.
                }]
        """
        return [
            self._load_train_data_by_date_range(date_range) \
            for date_range in \
                self._get_interaction_difference_split_date_ranges()
        ]
