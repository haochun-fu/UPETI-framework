# -*- coding: utf-8 -*-
"""DINDataLoader class."""
import copy
import datetime
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Union,
)

import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    MultiLabelBinarizer,
)

from comm_func import CommFunc
from dataloader import (
    DataLoaderBase,
    InvalidFeatureSetting,
)
from dataloader_helper import (
    ActionHelper,
    DataLoaderHelper,
)
from deepctr_helper import (
    DeepCTRHelper,
    TypeDNNFeatureColumn,
)
import pandas_helper


__author__ = 'haochun.fu'
__date__ = '2020-01-15'


class DINDataLoader(DataLoaderBase):
    """Data loader for Deep Interest Network (DIN)."""

    """Padding value of item for encoding. Can also be used for unknown item."""
    ENCODING_ITEM_PADDING_VALUE = 0

    """Padding value of product information for encoding. Can also be used for
    unknown product information."""
    ENCODING_PRODUCT_INFO_PADDING_VALUE = ''

    """Items of 'action' of feature setting."""
    FEATURE_SETTING_ACTION_ITEMS = (
        'action_weight',
        'duration',
        'time_weight',
    )

    """Items of 'product_info' information of feature setting."""
    FEATURE_SETTING_PRODUCT_INFO_ITEMS = (
        'amount',
        'name',
    )


    def __init__(
        self,
        data_dir: str,
        param: dict,
        test_date: str,
        is_validation: bool = False,
        is_test: bool = False,
        encoders: dict = None,
    ) -> None:
        """Constructor.

        Args:
            data_dir (str): Data directory.
            param (dict): Parameter.
            test_date (str): Test date in format Y-m-d.
            is_validation (bool): Whether it is for validation or not.
            is_test (bool): Whether it is for test or not.
            encoders (dict): Encoders. Default is None.
        """
        super().__init__(data_dir, param, test_date, is_validation, is_test)

        self.__encoders = encoders

    def load_test_data(self):
        """Load test data.

        Returns:
            dict: Test data.
        """
        return self._get_test_data()

    def load_train_data(self) -> dict:
        """Load training data.

        Returns:
            dict: Training data and data related to training data.
                {
                  'x': {
                    'feature_name': [
                      value of row,
                    ],
                  },
                  'y': [
                    label of row,
                  ],
                  'encoders': {
                    'name': instance of encoder,
                  },
                  'history_feature_list': [
                    'feature name',
                  ],
                  'dnn_feature_columns': [
                    instance of feature,
                  ],
                  'validation': {
                    'x': {
                      value of row,
                    },
                    'y': [
                      label of row,
                    ]
                  },
                  'dataloader': DataLoader,
                }

                If it is training for test, 'validation' will not be given.
        """
        return self._load_train_data_by_date_range(self._get_train_range())

    def to_x(self, rows: List[List[int]]) -> Tuple[dict, List[int]]:
        """Generate features of rows of user and item.

        Args:
            rows (list): Rows of user and item.
                [
                  [user, item],
                ]

        Returns:
            tuple: Features of rows of user and item, and indices of missing features.
        """
        t = self._test_date + ' 00:00:00'
        recs = [
            {
                'user': row[0],
                'item': row[1],
                'time': t,
            } for row in rows
        ]

        history_items = self._get_history_item(recs)

        features = {
            **self.__generate_feature_of_target(recs),
            **self.__generate_feature_of_history(history_items),
        }

        users, items = self.__get_user_item(recs)

        x = {
            'user': users,
            'item': items,
            'hist_item': history_items,
            **features,
        }
        x, _, miss_idxs = self.__filter_missing_feature(x)

        self.__process_special_item(x)

        self.__encode(x)

        for name, value in x.items():
            if isinstance(name, np.ndarray):
                continue
            x[name] = np.asarray(value)

        return x, miss_idxs

    def _add_negative_sample(
        self,
        recs: List[dict],
        labels: List[int],
        end_date: str,
        amount: int
    ) -> Tuple[List[dict], List[int]]:
        """Add negative sample.

        Sample from records before end date.

        If 'item' encoder was exist, item not encodable would be ignored.

        Args:
            recs (list): Records.
            labels (list): Labels.
            end_date (str): End date but not included in format Y-m-d.
            amount (int): Amount of adding negative sample.

        Returns:
            tuple: Records added negative samples and labels.
                (
                  [
                    {
                      'user': User id,
                      'item': Item id,
                      'time': 'Time',
                    }
                  ],
                  [
                    label,
                  ]
                )
        """
        class __DateAvailableItemSampler(object):
            def __init__(self, date: str, items: List[int]) -> None:
                self.__date = date
                self.__items = items
                self.__idx = 0
                self.__amount = len(items)

            def __eq__(self, that: str) -> bool:
                return self.__date == that

            def sample(self, excluded_items: Set[int]) -> int:
                ret = None
                if not self.__amount:
                    return ret

                start_idx = self.__idx
                while True:
                    item = self.__items[self.__idx]
                    if item not in excluded_items:
                        ret = item

                    self.__idx = (self.__idx + 1) % self.__amount
                    if ret is not None or self.__idx == start_idx:
                        break
                return ret

        if amount == 0:
            return recs, labels

        ret_recs = []
        ret_labels = []

        users = {rec['user'] for rec in recs}
        hist = self._user_history_action_item(users, end_date)
        encodable_items = self.__get_encodable_item_set()

        sampler = None
        for rec, label in zip(recs, labels):
            ret_recs.append(rec)
            ret_labels.append(label)

            date = rec['time'].split(' ')[0]
            if not sampler or sampler != date:
                available_items = self.get_available_contract_by_date(date)
                if encodable_items is not None:
                    available_items = [item for item in available_items \
                                            if item in encodable_items]

                sampler = __DateAvailableItemSampler(date, available_items)

            hist_user = hist.get(rec['user'])
            if hist_user is None:
                hist_user = set()
            marks = set()
            for _ in range(amount):
                item = sampler.sample(hist_user)
                if item is None or item in marks:
                    break

                neg_rec = copy.deepcopy(rec)
                neg_rec['item'] = item
                ret_recs.append(neg_rec)
                ret_labels.append(0)

                marks.add(item)

        return ret_recs, ret_labels

    def _get_history_item(self, recs: List[dict]) -> List[List[int]]:
        """Get history item for records.

        Args:
            recs (list): Records.

        Returns:
            list: History item of records.
        """
        class __UserItemHistory(object):
            def __init__(self, data: Dict[int, List[int]], amount: int) -> None:
                self.__data = data
                self.__amount = amount

            def get(self, user: int, time_: str) -> List[int]:
                ret = [None] * self.__amount
                items = self.__data.get(user)
                if not items:
                    return ret

                idx = None
                for i, (item, t) in enumerate(items):
                    if t >= time_:
                        break
                    idx = i
                if idx is None:
                    return ret
                idx += 1
                if idx >= self.__amount:
                    ret = [
                        item for item, _ in items[idx - self.__amount:idx][::-1]
                    ]
                else:
                    ret = [item for item, _ in items[:idx][::-1]] \
                          + [None] * (self.__amount - idx)

                return ret

        ret = []

        user_item_history = __UserItemHistory(
            self._get_positive_user_item(),
            self._param['feature']['history']['amount'])

        prev = {
            'user': None,
            'time': None,
            'history': None,
        }
        for rec in recs:
            if prev['user'] == rec['user'] and prev['time'] == rec['time']:
                ret.append(prev['history'])
                continue

            history = user_item_history.get(rec['user'], rec['time'])
            ret.append(history)

            prev['user'] = rec['user']
            prev['time'] = rec['time']
            prev['history'] = history

        return ret

    def _get_positive_recs(
        self,
        date_range: Tuple[str, str]
    ) -> Tuple[List[dict], List[int]]:
        """Get positive records.

        If 'user' and 'item' encoders were exist, filter record of which user or
            item is not encodable.

        Args:
            date_range (tuple): Start and end date.

        Returns:
            tuple: Positive records and labels.
                (
                  [
                    {
                      'user': User id,
                      'item': Item id,
                      'time': 'Time',
                    }
                  ],
                  [
                    label,
                  ]
                )
        """
        recs = []

        encodable_users = self.__get_encodable_user_set()
        encodable_items = self.__get_encodable_item_set()

        actions = self._param['label']['action']
        for data in self._iterate_data_in_date_range(*date_range):
            selector = None
            for action in actions:
                selects = data[action] == 1
                if selector is None:
                    selector = selects
                else:
                    selector |= selects
            for row in pandas_helper.iterate_with_dict(data[selector]):
                if encodable_users is not None:
                    if row['user_id'] not in encodable_users \
                        or row['contract_id'] not in encodable_items:
                        continue
                recs.append({
                    'user': row['user_id'],
                    'item': row['contract_id'],
                    'time': row['time'],
                })

        labels = [1] * len(recs)

        return recs, labels

    def _get_positive_user_item(self) -> Dict[int, List[Tuple[int, str]]]:
        """Get positive items with time of users.

        If 'user' and 'item' encoders were exist, filter record of which user or
            item is not encodable.

        Returns:
            dict: Positive items with time of users.
        """
        ret = {}

        encodable_users = self.__get_encodable_user_set()
        encodable_items = self.__get_encodable_item_set()

        actions = self._param['label']['action']
        for data in self._iterate_all_data():
            selector = None
            for action in actions:
                selects = data[action] == 1
                if selector is None:
                    selector = selects
                else:
                    selector |= selects
            for row in pandas_helper.iterate_with_dict(data[selector]):
                if encodable_users is not None:
                    if row['user_id'] not in encodable_users \
                        or row['contract_id'] not in encodable_items:
                        continue
                ret.setdefault(row['user_id'], []) \
                   .append((row['contract_id'], row['time']))

        return ret

    def _load_feature_data(self, features: List[dict]) -> dict:
        """Load feature data.

        Args:
            features (list): List of feature settings.

        Returns:
            dict: Feature data.
        """
        ret = {}
        for feature in features:
            name = feature['feature_name']
            if name == 'action' and 'user_item_action_time' not in ret:
                ret['user_item_action_time'] = self.get_user_item_action_time()
            elif name == 'product_vector':
                ret[name] = self.get_product_vector_by_setting(**{
                    item: feature[item] for item in \
                        DataLoaderBase.FEATURE_SETTING_PRODUCT_VECTOR_ITEMS
                })
            elif name == 'product_info' and 'product_info' not in ret:
                ret[name] = self.get_product_info()
        return ret

    def _load_train_data_by_date_range(self, date_range: Tuple[str, str]):
        """Load training data by date range.

        Args:
            date_range (tuple): Start and end dates.

        Returns:
            dict: Training data and data related to training data.
                {
                  'x': {
                    'feature_name': [
                      value of row,
                    ],
                  },
                  'y': [
                    label of row,
                  ],
                  'encoders': {
                    'name': instance of encoder,
                  },
                  'history_feature_list': [
                    'feature name',
                  ],
                  'dnn_feature_columns': [
                    instance of feature,
                  ],
                  'validation': {
                    'x': {
                      value of row,
                    },
                    'y': [
                      label of row,
                    ]
                  },
                  'dataloader': DataLoader,
                }

                If it is training for test, 'validation' will not be given.
        """
        self.__encoders = None

        feature_param = self._param['feature']
        hist_amount = feature_param['history']['amount']
        feature_param['hist_item']['deepctr']['param']['maxlen'] = hist_amount
        for setting in feature_param['history']['features']:
            setting['deepctr']['param']['maxlen'] = hist_amount

        recs, y = self._get_positive_recs(date_range)
        recs, y = self._add_negative_sample(
            recs,
            y,
            self._test_date,
            self._param['feature']['negative_sample_amount'])
        history_items = self._get_history_item(recs)

        features = {
            **self.__generate_feature_of_target(recs),
            **self.__generate_feature_of_history(history_items),
        }

        users, items = self.__get_user_item(recs)

        x = {
            'user': users,
            'item': items,
            'hist_item': history_items,
            **features,
        }
        x, y, _ = self.__filter_missing_feature(x, y)

        self.__process_special_item(x)

        self.__encoders = self.__generate_encoder(x)
        self.__encode(x)

        for name, value in x.items():
            if isinstance(name, np.ndarray):
                continue
            x[name] = np.asarray(value)

        ret = {
            'x': x,
            'y': y,
            'encoders': self.__encoders,
            'dnn_feature_columns': self.__generate_dnn_feature_columns(),
            'history_feature_list': self.__get_history_feature_list(),
            'dataloader': copy.deepcopy(self),
        }

        if self._is_validation:
            ret['validation'] = self.__get_validation()

        return ret

    def _user_history_action_item(
        self,
        users: Union[List[int], Set[int]],
        end_date: str
    ) -> Dict[int, Set[int]]:
        """Get history action items of users before end date.

        Args:
            users (list|set): User IDs.
            end_date (str): End date but not included in format Y-m-d.

        Returns:
            dict: History action items of users.
                {
                  user_id: {item_id, ...},
                }
        """
        if not isinstance(users, set):
            users = set(users)

        ret = {}
        for data in self._iterate_all_data():
            for row in pandas_helper.iterate_with_dict(data):
                if row['time'].split(' ')[0] >= end_date:
                    break
                elif row['user_id'] not in users:
                    continue

                ret.setdefault(row['user_id'], set()).add(row['contract_id'])
        return ret

    def _val_history_feature_setting(self, features: List[dict]) -> None:
        """Validate history feature setting.

        Args:
            feature (list): List of features.

        Raises:
            InvalidFeatureSetting: If feature setting is invalid.
        """
        items = ('product_info',)
        for feature in features:
            feature_name = feature['feature_name']
            if feature_name not in items:
                raise InvalidFeatureSetting(f"Invalid feature '{feature_name}'")

    def _val_target_feature_setting(self, features: List[dict]) -> None:
        """Validate target feature setting.

        Args:
            feature (list): List of features.

        Raises:
            InvalidFeatureSetting: If feature setting is invalid.
        """
        product_vector_count = 0
        items = ('action', 'product_vector', 'product_info')
        for feature in features:
            feature_name = feature['feature_name']
            if feature_name not in items:
                raise InvalidFeatureSetting(f"Invalid feature '{feature_name}'")
            if feature_name == 'product_vector':
                product_vector_count += 1

        if product_vector_count >= 2:
            raise InvalidFeatureSetting('Amount of product vector exceeds 1')

    def __encode(self, x: Dict[str, list]) -> None:
        """Encode feature(s).

        Args:
            x (dict): x.
        """
        for item in ('user', 'item'):
            x[item] = self.__encoders[item].transform(x[item])

        encoder = self.__encoders['item']
        hist_item = x['hist_item']
        padding_value = DINDataLoader.ENCODING_ITEM_PADDING_VALUE
        for i, row in enumerate(hist_item):
            for j, value in enumerate(row):
                if value is None:
                    row[j] = padding_value
            hist_item[i] = encoder.transform(row)

        self.__encode_product_info(x)

    def __encode_product_info(self, x: Dict[str, list]) -> None:
        """Encode product information.

        Args:
            x (dict): x.
        """
        padding_value = DINDataLoader.ENCODING_PRODUCT_INFO_PADDING_VALUE
        for setting in self._param['feature']['target']:
            if setting['feature_name'] != 'product_info':
                continue

            param_name = setting['deepctr']['param']['name']
            encoder = self.__encoders[param_name]
            features = x[param_name]
            amount = setting['amount']
            for i, feature in enumerate(features):
                diff_amount = amount - len(feature)
                if diff_amount > 0:
                    feature += [padding_value] * diff_amount
                if setting['encoding'] in ('one_hot', 'multi_hot'):
                    features[i] = encoder.transform([feature]).toarray()[0]
                elif setting['encoding'] == 'label':
                    features[i] = encoder.transform(feature)[0]

        for setting in self._param['feature']['history']['features']:
            if setting['feature_name'] != 'product_info':
                continue

            param_name = setting['deepctr']['param']['sparsefeat']['name']
            encoder = self.__encoders[param_name]
            features = x[param_name]
            amount = setting['amount']
            for row in features:
                for i, feature in enumerate(row):
                    diff_amount = amount - len(feature)
                    if diff_amount > 0:
                        feature += [padding_value] * diff_amount
                    if setting['encoding'] in ('one_hot', 'multi_hot'):
                        row[i] = encoder.transform([feature]).toarray()[0]
                    elif setting['encoding'] == 'label':
                        row[i] = encoder.transform(feature)[0]

    def __filter_missing_feature(
        self,
        x: Dict[str, list],
        y: List[int] = None
    ) -> Tuple[Dict[str, list], List[int], List[int]]:
        """Filter missing features.

        Args:
            x (dict): x.
            y (list): y. Default is None.

        Returns:
            tuple: x, y and indices of missing featurey. If y is not given, y is
                None.
        """
        ret_x = {}
        ret_y = None
        ret_miss = []

        check_names = []
        for row in self._param['feature']['target']:
            if row['feature_name'] == 'product_vector':
                check_names.append(row['deepctr']['param']['name'])

        if not check_names:
            return x, y, ret_miss

        idxs = []
        for idx, rows in enumerate(zip(*[x[name] for name in check_names])):
            is_pass = True
            for row in rows:
                if row is None:
                    is_pass = False
                    break
            (idxs if is_pass else ret_miss).append(idx)

        for key, values in x.items():
            ret_x[key] = [values[idx] for idx in idxs]
        if y:
            ret_y = [y[idx] for idx in idxs]

        return ret_x, ret_y, ret_miss

    def __generate_dnn_feature_columns(self) -> List[TypeDNNFeatureColumn]:
        """Generate DNN feature columns.

        Returns:
            list: List of DNN feature columns.
        """
        ret = []

        feature_setting = self._param['feature']

        hist_settings = feature_setting['history']['features']
        settings = [
            *[feature_setting[item] for item in ('user', 'item', 'hist_item')],
            *[setting for setting in feature_setting['target']],
            *[setting for setting in hist_settings],
        ]

        for setting in settings:
            deepctr = setting['deepctr']
            name = deepctr['param']['sparsefeat']['name'] \
                    if deepctr['type'] == 'sparse_var_len' \
                    else deepctr['param']['name']
            if deepctr['type'] in ('sparse', 'sparse_var_len'):
                encoder = self.__encoders[
                    'item' if name == 'hist_item' else name]
            else:
                encoder = None

            ret.append(
                DeepCTRHelper.generate_dnn_feature_column(
                    deepctr['type'], deepctr['param'], encoder))

        return ret

    def __generate_encoder(self, x: Dict[str, list]) -> dict:
        """Generate encoder(s) of feature(s).

        Args:
            x (dict): x.

        Returns:
            dict: Encoders.
        """
        def __iterate_feature_value(feature):
            for row in feature:
                if isinstance(row, list):
                    for value in __iterate_feature_value(row):
                        yield value
                else:
                    yield row

        ret = {}

        encoder = LabelEncoder()
        encoder.fit(x['user'])
        ret['user'] = encoder

        items = [DINDataLoader.ENCODING_ITEM_PADDING_VALUE] \
                + x['item'] \
                + [
                    value for row in x['hist_item'] \
                        for value in row if value is not None
                ]
        encoder = LabelEncoder()
        encoder.fit(items)
        ret['item'] = encoder

        for name in ('category', 'tag', 'keyword'):
            feature = set()
            feature_names = []
            encoding = None
            for item in ('target', 'history'):
                settings = self._param['feature'][item]
                if item == 'history':
                    settings = settings['features']
                for setting in settings:
                    if setting['feature_name'] != 'product_info' \
                       or setting['name'] != name:
                        continue

                    deepctr = setting['deepctr']
                    param_name = deepctr['param']['sparsefeat']['name'] \
                                    if deepctr['type'] == 'sparse_var_len' \
                                    else deepctr['param']['name']
                    for value in __iterate_feature_value(x[param_name]):
                        feature.add(value)

                    if not feature_names:
                        encoding = setting['encoding']
                    feature_names.append(param_name)
            if not feature_names:
                continue
            feature.add(DINDataLoader.ENCODING_PRODUCT_INFO_PADDING_VALUE)

            feature = sorted(value for value in feature)
            if encoding in ('one_hot', 'multi_hot'):
                feature = [[value] for value in feature]

            if encoding == 'label':
                encoder = LabelEncoder()
            elif encoding == 'one_hot':
                encoder = OneHotEncoder(handle_unknown='ignore')
            elif encoding == 'multi_hot':
                encoder = MultiLabelBinarizer()
            encoder.fit(feature)
            
            for feature_name in feature_names:
                ret[feature_name] = encoder

        return ret

    def __generate_feature_action(
        self,
        user: int,
        item: Union[int, List[int]],
        time: str,
        action_helper: ActionHelper
    ) -> Union[List[str], List[List[str]]]:
        """Generate feature of action for user and item.

        Args:
            user (int|list): User(s).
            item (int|list): Item(s).
            time (str): Time in format Y-m-d H:M:S.f or Y-m-d H:M:S.
            action_helper (ActionHelper): Action helper.

        Returns:
            list: Feature(s).
        """
        ret = []
        is_single = not isinstance(item, list)

        for item in [item] if is_single else item:
            ret.append(action_helper.action_amount(user, item, time))

        return ret[0] if is_single else ret

    def __generate_feature_of_history(
        self,
        recs: List[List[int]]
    ) -> Dict[str, list]:
        """Generate features of records of history.

        Args:
            recs (list): Records of history.

        Returns:
            dict: Features of records.
                {
                  'feature name': [value of row, ...],
                }
        """
        ret = {}
        features = self._param['feature']['history']['features']
        if not features:
            return ret
        self._val_history_feature_setting(features)
        data = self._load_feature_data(features)

        for feature in features:
            name = feature['feature_name']
            if name == 'product_info':
                method = self.__generate_feature_product_info
                params = lambda rec: {
                    'data': data['product_info'],
                    'item': rec,
                    **{
                        item: feature[item] for item in \
                            DINDataLoader.FEATURE_SETTING_PRODUCT_INFO_ITEMS
                    },
                }

            ret[feature['deepctr']['param']['sparsefeat']['name']] = [
                method(**params(rec)) for rec in recs
            ]

        return ret

    def __generate_feature_of_target(self, recs: List[dict]) -> Dict[str, list]:
        """Generate features of records of target.

        Args:
            recs (list): Records.

        Returns:
            dict: Features of records.
                {
                  'feature name': [value of row, ...],
                }
        """
        ret = {}
        features = self._param['feature']['target']
        if not features:
            return ret
        self._val_target_feature_setting(features)
        data = self._load_feature_data(features)

        for feature in features:
            name = feature['feature_name']
            if name == 'action':
                method = self.__generate_feature_action
                params = lambda rec: {
                    'user': rec['user'],
                    'item': rec['item'],
                    'time': rec['time'],
                    'action_helper': ActionHelper(
                        **{
                            **{
                                item: feature[item] for item in \
                                    DINDataLoader.FEATURE_SETTING_ACTION_ITEMS
                            },
                            'data': data['user_item_action_time'],
                        }
                    ),
                }
            elif name == 'product_vector':
                method = self.__generate_feature_product_vector
                params = lambda rec: {
                    'data': data['product_vector'],
                    'item': rec['item'],
                }
            elif name == 'product_info':
                method = self.__generate_feature_product_info
                params = lambda rec: {
                    'data': data['product_info'],
                    'item': rec['item'],
                    **{
                        item: feature[item] for item in \
                            DINDataLoader.FEATURE_SETTING_PRODUCT_INFO_ITEMS
                    },
                }

            ret[feature['deepctr']['param']['name']] = [
                method(**params(rec)) for rec in recs
            ]

        return ret

    def __generate_feature_product_info(
        self,
        item: Union[int, List[int]],
        data: Dict[int, Dict[str, List[str]]],
        amount: int,
        name: str
    ) -> Union[List[str], List[List[str]]]:
        """Generate feature of product information for item.

        Args:
            item (int|list): Item(s).
            data (dict): Product information.
            amount (int): Amount of choosing in order of frequency from high to
                low.
            name (str): Name, category, tag or keyword.

        Returns:
            list: Feature(s).
        """
        ret = []
        is_single = not isinstance(item, list)

        for item in [item] if is_single else item:
            info = data.get(item)
            if not info:
                ret.append([])
                continue
            info = info[name]
            if not info:
                ret.append([])
                continue

            info = info[:amount]
            ret.append(info)

        return ret[0] if is_single else ret

    def __generate_feature_product_vector(
        self,
        item: Union[int, List[int]],
        data: Dict[str, List[float]]
    ) -> Union[List[float], List[List[float]]]:
        """Generate feature of product vector for item.

        Args:
            item (int|list): Item(s).
            data (dict): Product vector.

        Returns:
            list: Feature(s).
        """
        ret = []
        is_single = not isinstance(item, list)

        for item in [item] if is_single else item:
            ret.append(data.get(item))

        return ret[0] if is_single else ret

    def __get_encodable_item_set(self) -> Union[Set[int], None]:
        """Get set of encodable items.

        Returns:
            set|None: Set of encodable items. If 'item' encoder was not exist,
                return None.
        """
        ret = None
        if not self.__encoders or 'item' not in self.__encoders:
            return ret
        ret = self.__encoders['item'].classes_.tolist()
        if DataLoaderBase.CONTRACT_ID_0_CONVERSION in ret:
            ret[ret.index(DataLoaderBase.CONTRACT_ID_0_CONVERSION)] = 0
        return set(ret)

    def __get_encodable_user_set(self) -> Union[Set[int], None]:
        """Get set of encodable users.

        Returns:
            set|None: Set of encodable users. If 'user' encoder was not exist,
                return None.
        """
        ret = None
        if not self.__encoders or 'user' not in self.__encoders:
            return ret
        return set(self.__encoders['user'].classes_.tolist())

    def __get_history_feature_list(self) -> List[str]:
        """Get list of history features.

        Returns:
            list: List of history features.
        """
        return [
            'item',
            *[setting['deepctr']['param']['sparsefeat']['embedding_name'] \
                for setting in self._param['feature']['history']['features']]
        ]

    def __get_user_item(self, recs: List[dict]) -> Tuple[List[int], List[int]]:
        """Get user and item from records.

        Args:
            recs (list): Records.

        Returns:
            tuple: List of Users and list of items.
        """
        users = [rec['user'] for rec in recs]
        items = [rec['item'] for rec in recs]

        return users, items

    def __get_validation(self) -> dict:
        """Get validation.

        Returns:
            dict: x and y.
        """
        recs, y = self._get_positive_recs((self._test_date, self._test_date))
        recs, y = self._add_negative_sample(
            recs,
            y,
            (self._test_date_obj + datetime.timedelta(days=1)) \
                .strftime('%Y-%m-%d'),
            1)

        history_items = self._get_history_item(recs)

        features = {
            **self.__generate_feature_of_target(recs),
            **self.__generate_feature_of_history(history_items),
        }

        users, items = self.__get_user_item(recs)

        x = {
            'user': users,
            'item': items,
            'hist_item': history_items,
            **features
        }
        x, y, _ = self.__filter_missing_feature(x, y)

        self.__process_special_item(x)

        self.__encode(x)

        for name, value in x.items():
            if isinstance(name, np.ndarray):
                continue
            x[name] = np.asarray(value)

        return {
            'x': x,
            'y': y,
        }

    def __process_special_item(self, x: Dict[str, list]) -> None:
        """Process special item.

            Change contract_id 0 to DataLoaderBase.CONTRACT_ID_0_CONVERSION.

        Args:
            x (dict): x.
        """
        items = x['item']
        for i, item in enumerate(items):
            if item == 0:
                items[i] = DataLoaderBase.CONTRACT_ID_0_CONVERSION

        for row in x['hist_item']:
            for i, item in enumerate(row):
                if item == 0:
                    row[i] = DataLoaderBase.CONTRACT_ID_0_CONVERSION

