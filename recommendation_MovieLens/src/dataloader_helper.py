# -*- coding: utf-8 -*-
"""Helper classes for dataloader."""
import datetime
import math
from typing import (
    Union,
)

import numpy as np
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
)

from date_helper import DateHelper


__author__ = 'haochun.fu'
__date__ = '2020-07-05'


class InteractionHelper(object):
    """Generate interaction feature."""

    """Time format."""
    TIME_FMT = '%Y-%m-%d %H:%M:%S'

    """Name of 'name' of linear decay in time weight."""
    TIME_WEIGHT_NAME_DECREASE_BY_DAY = 'decrease_by_day'

    """Name of exponential decay in time weight."""
    TIME_WEIGHT_NAME_EXPONENTIAL_DECAY = 'exponential_decay'

    """Name of first decay in time weight."""
    TIME_WEIGHT_IS_USE_FIRST_DECAY = 'is_use_first_decay'


    def __init__(
        self,
        duration: int,
        end_time: str,
        time_weight: dict
    ) -> None:
        """Constructor.

        Args:
            duration (int): Duration of training data.
            end_time (str): End time but not included in format Y-m-d H:M:S.
            time_weight (dict): Time weight.
        """
        self.__duration = duration
        self.__end_time = end_time
        self.__time_weight = time_weight

        self.__end_time_obj = DateHelper.str_to_datetime(
            end_time, InteractionHelper.TIME_FMT)

    def preference(
        self,
        preference: Union[float, int],
        time_: str
    ) -> Union[float, int]:
        """Adjust preference of user and item from specified time.

        Args:
            preference (float|int): Preference.
            time_ (str): Time in format Y-m-d H:M:S.

        Returns:
            float|int: Adjusted preference.
        """
        return self.__apply_time_weight(
            preference,
            DateHelper.str_to_datetime(time_, InteractionHelper.TIME_FMT)
        )

    def __apply_time_weight(
        self,
        preference: Union[float, int],
        time_obj: datetime.datetime
    ) -> Union[float, int]:
        """Apply time weight.

        Args:
            preference (float|int): Preference.
            time_obj (datetime.datetime): Time.

        Returns:
            float|int: Preference.
        """
        ret = preference

        name = InteractionHelper.TIME_WEIGHT_IS_USE_FIRST_DECAY
        if name in self.__time_weight and self.__time_weight[name]:
            return ret

        name = self.__time_weight['name']
        param = self.__time_weight.get('param')
        if name == InteractionHelper.TIME_WEIGHT_NAME_DECREASE_BY_DAY:
            ret *= self.__duration - (self.__end_time_obj - time_obj).days
        elif name == InteractionHelper.TIME_WEIGHT_NAME_EXPONENTIAL_DECAY:
            ret *= math.exp(
                - param['lambda'] * (self.__end_time_obj- time_obj).days)

        return ret


class DataLoaderHelper(object):
    """Helper functions for data loader."""


    """Matrix type of coo matrix."""
    MATRIX_TYPE_COO = 'coo'

    """Matrix type of csr matrix."""
    MATRIX_TYPE_CSR = 'csr'


    @staticmethod
    def construct_matrix(
        values: dict,
        user_ids: list,
        item_ids: list,
        matrix_type: str
    ) -> Union[coo_matrix, csr_matrix]:
        """Construct matrix.

        Args:
            values (dict): Data of user and item.
            user_ids (list): List of user ids.
            item_ids (list): List of item ids.
            matrix_type (str): Type of matrix. DataLoaderHelper.MATRIX_TYPE_COO,
                DataLoaderHelper.MATRIX_TYPE_CSR.

        Returns:
            scipy.sparse.coo_matrix|scipy.sparse.csr_matrix: Matrix.
        """
        user_ids_idx = {id_: i for i, id_ in enumerate(user_ids)}
        item_ids_idx = {id_: i for i, id_ in enumerate(item_ids)}

        row_idx = []
        col_idx = []
        data = []
        for user, items in values.items():
            if user not in user_ids_idx:
                continue
            for item, value in items.items():
                if item not in item_ids_idx:
                    continue
                row_idx.append(user_ids_idx[user])
                col_idx.append(item_ids_idx[item])
                data.append(value)
        row_idx = np.asarray(row_idx)
        col_idx = np.asarray(col_idx)
        data = np.asarray(data)

        user_amount = len(user_ids)
        item_amount = len(item_ids)
        data_for_matrix = (data, (row_idx, col_idx))
        shape = (user_amount, item_amount)

        matrix_func = coo_matrix \
                      if matrix_type == DataLoaderHelper.MATRIX_TYPE_COO \
                      else csr_matrix

        return matrix_func(data_for_matrix, shape=shape)
