# -*- coding: utf-8 -*-
"""DINInteractionDifferenceDataLoader class."""
from typing import List

from din_dataloader import DINDataLoader


__author__ = 'haochun.fu'
__date__ = '2020-07-07'


class DINInteractionDifferenceDataLoader(DINDataLoader):
    """Data loader for DIN interaction difference."""


    def load_train_data(self) -> List[dict]:
        """Load training data.

        Returns:
            list: List of training data and data related to training data.
                [{
                  'x': {
                    'feature_name': [
                      value of row,
                    ],
                  },
                  'y': [
                    label of row,
                  ],
                  'history_feature_list': [
                    'feature name',
                  ],
                  'dnn_feature_columns': [
                    instance of feature,
                  ],
                  'encoders': {
                    'name': instance of encoder,
                  },
                  'genres': ['GENRE', ...],
                  'validation': {
                    'x': {
                      value of row,
                    },
                    'y': [
                      label of row,
                    ]
                  },
                  'dataloader': DataLoader,
                }]

                If it is training for test, 'validation' will not be given.
        """
        return [
            self._load_train_data_by_date_range(date_range) for date_range in \
                self._get_interaction_difference_split_date_ranges()
        ]
