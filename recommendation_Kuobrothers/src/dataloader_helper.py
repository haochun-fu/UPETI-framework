# -*- coding: utf-8 -*-
"""Helper classes for dataloader."""
from collections import defaultdict
import copy
import datetime
import math
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

import numpy as np
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
)

from comm_func import CommFunc


__author__ = 'haochun.fu'
__date__ = '2019-10-15'


class ActionHelper(object):
    """Generate action feature."""

    """Actions."""
    ACTIONS = ('convert', 'favorite', 'pageview')

    """Action type 'ref_search'."""
    ACTION_TYPE_REF_SEARCH = 'ref_search'

    """Directory name of available product."""
    AVAILABLE_PRODUCT_DIR_NAME = 'available_product'

    """Name of 'name' of linear decay in time weight."""
    TIME_WEIGHT_NAME_DECREASE_BY_DAY = 'decrease_by_day'

    """Name of exponential decay in time weight."""
    TIME_WEIGHT_NAME_EXPONENTIAL_DECAY = 'exponential_decay'

    """Name of first decay in time weight."""
    TIME_WEIGHT_IS_USE_FIRST_DECAY = 'is_use_first_decay'


    def __init__(
        self,
        duration: int,
        action_weight: Dict[str, Dict[str, Union[float, int]]],
        time_weight: dict,
        data: Dict[int, Dict[int, Dict[str, Dict[str, List[str]]]]]
    ) -> None:
        """Constructor.

        Args:
            duration (int): Duration of training data.
            action_weight (dict): Action weight.
            time_weight (dict): Time weight.
            data (dict): Action time of user and item.
        """
        self.__duration = duration
        self.__action_weight = action_weight
        self.__time_weight = time_weight
        self.__data = data

    def action_amount(
        self,
        user: int,
        item: int,
        end_time: str
    ) -> Union[float, int]:
        """Compute action amount of user and item from specified time.

        Args:
            user (int): User.
            item (int): Item.
            end_time (str): End time but not included in format Y-m-d H:M:S.f or
                Y-m-d H:M:S.

        Returns:
            float|int: Action amount.
        """
        end_time_obj = CommFunc.str_to_datetime(end_time)
        start_time_obj = end_time_obj - datetime.timedelta(days=self.__duration)
        start_time = CommFunc.remove_microsecond_padding_zero(
            start_time_obj.strftime('%Y-%m-%d %H:%M:%S.%f'))
        times = self.__collect_action_time(user, item, start_time, end_time)
        return sum(self.__apply_time_weight(times, end_time_obj))

    def __apply_time_weight(
        self,
        times: List[Tuple[str, Union[float, int]]],
        end_time: datetime.datetime
    ) -> List[Union[float, int]]:
        """Apply time weight.

        Args:
            times (list): Times with weight.
            end_time (datetime.datetime): End time but not included.

        Returns:
            list: List of weights.
        """
        def __adjusted_day_by_first_decay():
            ret = 0

            name = ActionHelper.TIME_WEIGHT_IS_USE_FIRST_DECAY
            if name not in self.__time_weight or \
                not self.__time_weight[name] or \
                not times:
                return ret

            ret = (end_time - CommFunc.str_to_datetime(times[0][0])).days

            return ret

        ret = []
        param_name = self.__time_weight['name']
        param = self.__time_weight.get('param')
        if param_name == ActionHelper.TIME_WEIGHT_NAME_DECREASE_BY_DAY:
            def method(t, adjusted_d):
                return self.__duration \
                    - (end_time - CommFunc.str_to_datetime(t)).days \
                    + adjusted_d
        elif param_name == ActionHelper.TIME_WEIGHT_NAME_EXPONENTIAL_DECAY:
            def method(t, adjusted_d):
                diff_days = (end_time - CommFunc.str_to_datetime(t)).days + \
                    adjusted_d
                return math.exp(- param['lambda'] * diff_days)

        adjusted_d = __adjusted_day_by_first_decay()
        for (t, weight) in times:
            ret.append(weight * method(t, adjusted_d))

        return ret

    def __collect_action_time(
        self,
        user: int,
        item: int,
        start_time: str,
        end_time: str
    ) -> List[Tuple[str, Union[float, int]]]:
        """Count amount.

        Args:
            user (int): User.
            item (int): Item.
            start_time (str): Start time.
            end_time (str): Ending but not included time.

        Returns:
            list: List of times with weight.
        """
        ret = []
        user_actions = self.__data.get(user)
        if not user_actions: return ret
        item_actions = user_actions.get(item)
        if not item_actions:
            return ret

        for type_, actions in self.__action_weight.items():
            for action, weight in actions.items():
                if weight == 0:
                    continue

                item_types = item_actions.get(action)
                if not item_types:
                    continue
                item_times = item_types.get(type_)
                if not item_times:
                    continue

                for t in item_times:
                    if t < start_time:
                        continue
                    elif t >= end_time:
                        break
                    ret.append((t, weight))

        ret.sort(key=lambda row: row[0], reverse=True)

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
        contract_ids: list,
        matrix_type: str
    ) -> Union[coo_matrix, csr_matrix]:
        """Construct matrix.

        Args:
            values (dict): Data of user and contract.
            user_ids (list): List of user ids.
            contract_ids (list): List of contract ids.
            matrix_type (str): Type of matrix. DataLoaderHelper.MATRIX_TYPE_COO,
                DataLoaderHelper.MATRIX_TYPE_CSR.

        Returns:
            scipy.sparse.coo_matrix|scipy.sparse.csr_matrix: Matrix.
        """
        user_ids_idx = {id_: i for i, id_ in enumerate(user_ids)}
        contract_ids_idx = {id_: i for i, id_ in enumerate(contract_ids)}

        row_idx = []
        col_idx = []
        data = []
        for user, contracts in values.items():
            if user not in user_ids_idx:
                continue
            for contract, value in contracts.items():
                if contract not in contract_ids_idx:
                    continue
                row_idx.append(user_ids_idx[user])
                col_idx.append(contract_ids_idx[contract])
                data.append(value)
        row_idx = np.asarray(row_idx)
        col_idx = np.asarray(col_idx)
        data = np.asarray(data)

        user_amount = len(user_ids)
        contract_amount = len(contract_ids)
        data_for_matrix = (data, (row_idx, col_idx))
        shape = (user_amount, contract_amount)

        matrix_func = coo_matrix \
                      if matrix_type == DataLoaderHelper.MATRIX_TYPE_COO\
                      else csr_matrix
        result = matrix_func(data_for_matrix, shape=shape)

        return result
