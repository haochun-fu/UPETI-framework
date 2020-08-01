# -*- coding: utf-8 -*-
"""DataLoaderBase and MatrixDataLoader classes."""
import abc
import datetime
import glob
import os
from typing import (
    Dict,
    Generator,
    List,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from comm_func import CommFunc
from dataloader_helper import ActionHelper
import pandas_helper
from sklearn.preprocessing import MultiLabelBinarizer


__author__ = 'haochun.fu'
__date__ = '2019-10-15'


class DataLoaderException(Exception):
    """Base exception of DataLoader."""
    pass


class InvalidFeatureSetting(DataLoaderException):
    """Invalid feature setting."""
    pass


class DataLoaderBase(abc.ABC):
    """Basis of DataLoader to be inherited."""

    """Actions."""
    ACTIONS = ('convert', 'favorite', 'pageview')

    """Action type 'general'."""
    ACTION_TYPE_GENERAL = 'general'

    """Action type 'ref_search'."""
    ACTION_TYPE_REF_SEARCH = 'ref_search'

    """Directory name of available product."""
    AVAILABLE_PRODUCT_DIR_NAME = 'available_product'

    """Conversion of contract id 0."""
    CONTRACT_ID_0_CONVERSION = 100000

    """Directory name of features."""
    FEATURES_DIR_NAME = 'features'

    """Directory name of product information."""
    PRODUCT_INFO_DIR_NAME = 'product_info'

    """File name of product information."""
    PRODUCT_INFO_FILE_NAME = 'product_info.json'

    """Directory name of product vector."""
    PRODUCT_VECTOR_DIR_NAME = 'product_vector'

    """Prefix file name of product vector."""
    PRODUCT_VECTOR_PREFIX_FILE_NAME = 'product'

    """Items of product vector of feature setting."""
    FEATURE_SETTING_PRODUCT_VECTOR_ITEMS = (
        'model',
        'size',
        'normalization',
        'scale',
        'field',
        'field_operation',
    )


    def __init__(
        self,
        data_dir: str,
        param: dict,
        test_date: str,
        is_validation: bool = False,
        is_test: bool = False
    ) -> None:
        """Constructor.

        Args:
            data_dir (str): Data directory.
            param (dict): Parameter.
            test_date (str): Test date in format Y-m-d.
            is_validation (bool): Whether it is for validation or not.
            is_test (bool): Whether it is for test or not.
        """
        self.__data_dir = data_dir
        self._param = param
        self._test_date  = test_date
        self._is_validation = is_validation
        self._is_test = is_test

        self._available_product_dir = os.path.join(
            self.__data_dir,
            DataLoaderBase.AVAILABLE_PRODUCT_DIR_NAME
            )
        self._features_dir = os.path.join(self.__data_dir,
                                          DataLoaderBase.FEATURES_DIR_NAME)

        self._product_info_dir = os.path.join(
            self.__data_dir,
            DataLoaderBase.PRODUCT_INFO_DIR_NAME)
        self._product_info_file = os.path.join(
            self._product_info_dir,
            DataLoaderBase.PRODUCT_INFO_FILE_NAME)

        self._product_vector_dir = os.path.join(
            self.__data_dir,
            DataLoaderBase.PRODUCT_VECTOR_DIR_NAME)
        self._test_date_obj = datetime.datetime.strptime(self._test_date,
                                                         '%Y-%m-%d')

    def get_available_contract(self) -> List[int]:
        """Get available contract in test date.

        Returns:
            list: List of available contract_ids.
        """
        return self.get_available_contract_by_date(self._test_date)

    def get_available_contract_by_date(self, date: str) -> List[int]:
        """Get available contract in specified date.

        Returns:
            list: List of available contract_ids.
        """
        data = pd.read_csv(
            os.path.join(self._available_product_dir, date + '.csv'))
        return [int(id_) for id_ in data.contractid.unique()]

    def get_product_info(self) -> Dict[int, Dict[str, List[str]]]:
        """Get product information.

        Returns:
            dict: Product information.
                {
                  CONTRACT_ID: {
                    'category|tag|keyword': [
                      'value'
                    ]
                  }
                }
        """
        data = CommFunc.load_json(self._product_info_file)
        return {int(key): value for key, value in data.items()}

    def get_product_vector( 
        self,
        model: str,
        size: Union[int, str],
        normalization: str
    ) -> Dict[int, List[float]]:
        """Get product vectors.

        Args:
            model (str): Model name.
            size (int|str): Size.
            normalization (str): Normalization name.

        Returns:
            dict: Product vectors.
        """
        file = os.path.join(
            self._product_vector_dir,
            model,
            f"size_{size}",
            f'{DataLoaderBase.PRODUCT_VECTOR_PREFIX_FILE_NAME}'
            f"{'_' + normalization if normalization else ''}"
            '.json')
        return {int(id_): val for id_, val in CommFunc.load_json(file).items()}

    def get_product_vector_by_setting(
        self,
        model: str,
        size: Union[int, str],
        normalization: str,
        scale: float,
        field: List[str],
        field_operation: str
    ) -> Dict[int, List[float]]:
        """Get product vector by setting.

        Args:
            model (str): Model name.
            size (int|str): Size.
            normalization (str): Normalization name.
            scale (float): Scale.
            field (list): List of fields.
            field_operation (str): Field operation, add, concatenate.

        Returns:
            dict: Product vectors.
        """
        data = self.get_product_vector(model, size, normalization)
        data = self._scale_product_vector(data, scale)
        return self._operate_product_vector(data, field, field_operation)

    def get_user_item_action_time(
        self) -> Dict[int, Dict[int, Dict[str, Dict[str, List[str]]]]]:
        """Get action time of user and item.

        Returns:
            dict: Action time of user and item.
                {
                  USER_ID: {
                    CONTRACT_ID: {
                      'convert|favorite|pageview': {
                        'general|ref_search': [
                          'TIME'
                        ]
                      }
                    }
                  }
                }
        """
        ret = {}
        for data in self._iterate_all_data():
            for row in pandas_helper.iterate_with_dict(data):
                for action in DataLoaderBase.ACTIONS:
                    if row[action] == 0:
                        continue

                    ret_action = ret.setdefault(row['user_id'], {}) \
                                    .setdefault(row['contract_id'], {}) \
                                    .setdefault(action, {})
                    for type_ in (DataLoaderBase.ACTION_TYPE_GENERAL,
                                  DataLoaderBase.ACTION_TYPE_REF_SEARCH):
                        if type_ == DataLoaderBase.ACTION_TYPE_REF_SEARCH \
                           and row['ref_search'] == 0:
                            continue
                        ret_action.setdefault(type_, []) \
                                  .append(row['time'])
                    break

        return ret

    def _get_feature_by_date(self, date: str) -> pd.core.frame.DataFrame:
        """Get feature of certain date.

        Args:
            date (str): Date in format Y-m-d.

        Returns:
            pandas.core.frame.DataFrame: Data.
        """
        return pd.read_csv(self._get_feature_file_by_date(date))

    def _get_feature_file_by_date(self, date: str) -> str:
        """Get feature file path of certain date.

        Args:
            date (str): Date in format Y-m-d.

        Returns:
            str: Feature file path.
        """
        return os.path.join(self._features_dir, f"{date}.csv")

    def _get_interaction_difference_split_date_ranges(
        self
    ) -> List[Tuple[str, str]]:
        """Get split date ranges according to interaction difference.

        If per split days < 0, will use 1.

        Returns:
            list: List of split date ranges.
        """
        ret = []

        fmt = '%Y-%m-%d'

        start, end = self._get_train_range()
        start_obj = datetime.datetime.strptime(start, fmt)
        end_obj = datetime.datetime.strptime(end, fmt)

        split_amount = 1
        if 'interaction_difference' in self._param:
            split_amount = self._param['interaction_difference']['split']
        split_days = int(((end_obj - start_obj).days + 1) / split_amount)

        if split_days == 0:
            split_days = 1

        tmp_obj = start_obj
        while tmp_obj <= end_obj:
            tmp_date = tmp_obj.strftime(fmt)
            tmp_end_obj = tmp_obj + datetime.timedelta(days=split_days - 1)
            if tmp_end_obj > end_obj:
                ret.append((tmp_date, end))
                break
            ret.append((tmp_date, tmp_end_obj.strftime(fmt)))

            tmp_obj = tmp_end_obj + datetime.timedelta(days=1)

        return ret

    def _get_test_data(self) -> dict:
        """Get convert contracts of each user in test date.

        Returns:
            dict: Convert contracts of each user in test date.
                {
                  USER_ID: [CONTRACT_ID, ...]
                }
        """
        result = {}
        for row in pandas_helper.iterate_with_dict(
            self._get_feature_by_date(self._test_date)):
            if row['convert'] != 1:
                continue
            user_id = int(row['user_id'])
            contract_id = int(row['contract_id'])
            contracts = result.setdefault(user_id, [])
            if contract_id not in contracts:
                contracts.append(contract_id)
        return result

    def _get_train_range(self) -> Tuple[str, str]:
        """Get date range for training data.

        Returns:
            tuple: Start and end date in format Y-m-d.
        """
        start_date_obj = self._test_date_obj \
                         - datetime.timedelta(days=self._param['duration'])
        end_date_obj = self._test_date_obj - datetime.timedelta(days=1)

        return start_date_obj.strftime("%Y-%m-%d"), \
               end_date_obj.strftime("%Y-%m-%d")

    def _iterate_all_data(
        self
    ) -> Generator[pd.core.frame.DataFrame, None, None]:
        """Generator of iterating all data in order of date.

        Yields:
            pandas.core.frame.DataFrame: Data in next date.

        Examples:
            >>> [for data in self._iterate_data_in_date_range('1970-01-01', '1970-01-03')]
            [data1, data2, ...]
        """
        for file in sorted(glob.glob(os.path.join(self._features_dir, '*'))):
            yield pd.read_csv(file)

    def _iterate_data_in_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> Generator[pd.core.frame.DataFrame, None, None]:
        """Generator of iterating data in date range.
        Args:
            start_date (str): Start date e.q. 1970-01-01.
            end_date (str): End date e.q. 1970-01-01.

        Yields:
            pandas.core.frame.DataFrame: Data in next date.

        Examples:
            >>> [for data in self._iterate_data_in_date_range('1970-01-01', '1970-01-03')]
            [data1, data2, ...]
        """
        for date in CommFunc.iterate_date_range(start_date, end_date):
            yield self._get_feature_by_date(date)

    def _operate_product_vector(
        self,
        data: Dict[int, Dict[str, List[float]]],
        fields: List[str],
        field_operation: str
    ) -> Dict[int, List[float]]:
        """Operate product vector.

        Args:
            data (dict): Product vector.
            fields (list): List of fields.
            field_operation (str): Field operation, add, concatenate.

        Returns:
            dict: Operated product vector. If some row failed to be operated,
                this row would be None.
        """
        assert field_operation in ('add', 'concatenate'), \
            'Invalid field operation, field operation should be "add" or' \
            ' "concatenate"'

        ret = {}
        for key, values in data.items():
            vec = None
            for field in fields:
                field_vec = values[field]
                if not field_vec:
                    vec = None
                    break
                field_vec = np.asarray(field_vec)
                if vec is None:
                    vec = field_vec
                else:
                    if field_operation == 'add':
                        vec += raw_vec
                    elif field_operation == 'concatenate':
                        vec = np.concatenate((vec, field_vec))
            ret[key] = vec.tolist() if vec is not None else vec
        return ret

    def _scale_product_vector(
        self,
        data: Dict[int, Dict[str, List[float]]],
        scale: float
    ) -> Dict[int, List[float]]:
        """Scale product vector.

        Args:
            data (dict): Product vector.
            scale (float): Scale.

        Returns:
            dict: Scaled product vector.
        """
        ret = {}
        for key, values in data.items():
            key_dict = ret.setdefault(key, {})
            for value_key, value in values.items():
                key_dict[value_key] = [val * scale for val in value] \
                                      if value else value
        return ret

    @abc.abstractmethod
    def load_test_data(self):
        """Load test data."""
        return NotImplemented

    @abc.abstractmethod
    def load_train_data(self):
        """Load training data."""
        return NotImplemented

    @property
    def test_date(self) -> str:
        return self._test_date


class MatrixDataLoader(DataLoaderBase):
    """Data loader for Matrix."""


    def load_test_data(self):
        """Load test data.

        Returns:
            dict: Test data.
        """
        return self._get_test_data()

    def _get_ids_from_interactive_data(self, data: dict) -> Tuple[list, list]:
        """Get user_ids and contract_ids from interactive data.

        Args:
            data (dict): Interactive data.

        Returns:
            tuple: List of sorted user_ids and list of sorted contract_ids.
        """
        user_ids = []
        contract_ids = set()
        for user, contracts in data.items():
            user_ids.append(user)
            for contract, _ in contracts.items():
                contract_ids.add(contract)

        user_ids = sorted(user_ids)
        contract_ids = sorted(contract_ids)

        return user_ids, contract_ids

    def _get_interactive_data_by_range(
        self,
        date_range: Tuple[str, str]
    ) -> Dict[int, Dict[int, Union[float, int]]]:
        """Get interactive data in training range.

        Args:
            date_range (tuple): Start and end dates.
        
        Returns:
            dict: Interactive data of users.
                {
                  USER_ID: {
                    CONTRACT_ID: ACTION_AMOUNT
                  }
                }
        """
        ret = {}

        pairs = set()
        for date in CommFunc.iterate_date_range(*date_range):
            for row in pandas_helper.iterate_with_dict(
                        self._get_feature_by_date(date)):
                pairs.add((row['user_id'], row['contract_id']))

        fmt = '%Y-%m-%d'
        start_date_obj = datetime.datetime.strptime(date_range[0], fmt)
        end_date_obj = datetime.datetime.strptime(date_range[1], fmt)

        label_param = self._param['label']
        action_helper = ActionHelper(
            (end_date_obj-start_date_obj).days+1,
            label_param['action_weight'],
            label_param['time_weight'],
            self.get_user_item_action_time())

        end_time = (end_date_obj + datetime.timedelta(days=1)).strftime(fmt) \
                   + ' 00:00:00'
        for user, item in pairs:
            amount = action_helper.action_amount(user, item, end_time)
            if amount > 0:
                ret.setdefault(user, {})[item] = amount

        return ret

    def _get_item_features(
        self,
        contract_ids: List[int]
    ) -> Tuple[List[List[float]], List[int]]:
        """Get item features.

        Args:
            contract_ids (list): List of contract ids.

        Returns:
            tuple: Item features and contract ids in item features.
        """
        ids = contract_ids
        features = None
        if self._param.get('product_vector'):
            vecs, ids = self._get_product_vector_by_id(ids)
            features = [vecs[id_] for id_ in ids]
        if 'product_info' in self._param:
            infos = self._get_product_info_by_id(ids)
            if features:
                for i, row in zip(range(len(ids)), infos):
                    features[i] += row
            else:
                features = infos
        return features, ids

    def _get_product_info_by_id(self, ids: List[int]) -> List[List[int]]:
        """Get product information of item ids.

        Args:
            ids (list): List of item ids.

        Returns:
            list: List of information.
        """
        info = self.get_product_info()

        items = {}
        for name, amount in self._param['product_info'].items():
            if amount == 0:
                continue

            data = []
            for id_ in ids:
                info_id = info.get(id_)
                if not info_id:
                    data.append([])
                    continue

                if amount < 0:
                    data.append(info_id[name])
                else:
                    data.append(info_id[name][:amount])

            items[name] = MultiLabelBinarizer().fit_transform(data)

        return np.concatenate(
            [items[name] for name in sorted(items.keys())], axis=1).tolist()

    def _get_product_vector_by_id(
        self,
        contract_ids: List[int]
    ) -> Tuple[dict, List[int]]:
        """Get product vectors of contract ids.

        Concate vectors of fields in param.

        If contract does not have product vector of any field, ignores.

        Args:
            contract_ids (list): List of contract ids.

        Returns:
            tuple: Product vectors and contract ids of product vectors.
        """
        ids = []
        vecs = {}

        data = self.get_product_vector_by_setting(**{
            item: self._param['product_vector'][item] for item in \
                DataLoaderBase.FEATURE_SETTING_PRODUCT_VECTOR_ITEMS
        })
        for id_ in contract_ids:
            if id_ not in data or data[id_] is None:
                continue
            vecs[id_] = data[id_]
            ids.append(id_)
        return vecs, ids
