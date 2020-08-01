# -*- coding: utf-8 -*-
"""DeepCTRHelper class."""
import copy
from typing import Union

from deepctr.feature_column import (
    SparseFeat,
    DenseFeat,
    VarLenSparseFeat,
)


__author__ = 'haochun.fu'
__date__ = '2020-02-14'


TypeDNNFeatureColumn = Union[DenseFeat, SparseFeat, VarLenSparseFeat]


class DeepCTRHelper(object):
    """Helper functions for DeepCTR."""


    @staticmethod
    def generate_dnn_feature_column(
        type_: str,
        params: dict,
        encoder: object = None
    ) -> TypeDNNFeatureColumn:
        """Generate DNN feature column.

        Args:
            type_ (str): Type, dense, sparse or sparse_var_len.
            params (dict): Parameters.
            encoder (object): Encoder for sparse features. It is used when type
                is sparse or sparse_var_len.

        Returns:
            deepctr.inputs.DenseFeat|deepctr.inputs.SparseFeat
            |deepctr.inputs.VarLenSparseFeat: DNN feature column.
        """
        params = copy.deepcopy(params)

        if type_ == 'dense':
            ret = DenseFeat(**params)
        elif type_ == 'sparse':
            size = encoder.classes_.shape[0]
            if params['vocabulary_size'] == 'auto+1':
                size += 1
            params['vocabulary_size'] = size
            ret = SparseFeat(**params)
        elif type_ == 'sparse_var_len':
            sparse_params = params['sparsefeat']
            size = encoder.classes_.shape[0]
            if sparse_params['vocabulary_size'] == 'auto+1':
                size += 1
            sparse_params['vocabulary_size'] = size
            params['sparsefeat'] = SparseFeat(**sparse_params)
            ret = VarLenSparseFeat(**params)

        return ret
