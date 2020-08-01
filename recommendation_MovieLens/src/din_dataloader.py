# -*- coding: utf-8 -*-
"""DINDataLoader class."""
import copy
import datetime
import itertools
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Union,
)

import numpy as np
from sklearn.preprocessing import LabelEncoder

from dataloader import (
    MovielensDataLoaderBase,
    InvalidFeatureSetting,
)
from deepctr_helper import (
    DeepCTRHelper,
    TypeDNNFeatureColumn,
)
import pandas_helper


__author__ = 'haochun.fu'
__date__ = '2020-07-06'


class DINDataLoader(MovielensDataLoaderBase):
    """Data loader for Deep Interest Network (DIN)."""


    """Padding value of item for encoding. Can also be used for unknown item."""
    ENCODING_ITEM_PADDING_VALUE = 0

    """Padding value of genome tags for encoding. Can also be used for empty."""
    ENCODING_ITEM_INFO_GENOME_TAGS_PADDING_VALUE = 0

    """Name of genome tags for item information."""
    ITEM_INFO_GENOME_TAGS = 'item_info_genome_tags'

    """Name of genres for item information."""
    ITEM_INFO_GENRES = 'item_info_genres'


    def __init__(
        self,
        data_dir: str,
        param: dict,
        test_date: str,
        is_validation: bool = False,
        is_test: bool = False,
        encoders: dict = None,
        genres: list = None
    ) -> None:
        """Constructor.

        Args:
            data_dir (str): Data directory.
            param (dict): Parameter.
            test_date (str): Test date in format Y-m-d.
            is_validation (bool): Whether it is for validation or not.
            is_test (bool): Whether it is for test or not.
            encoders (dict): Encoders. Default is None.
            genres (list): Genres being used.
        """
        super().__init__(data_dir, param, test_date, is_validation, is_test)

        self.__encoders = encoders
        self.__genres = genres

        self.__cache_positive_user_item = None
        self.__cache_target_feature_data = None
        self.__cache_history_feature_data = None

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
        class __CandidatesItemSampler(object):
            def __init__(self, items: List[int]) -> None:
                self.__items = items
                self.__idx = 0
                self.__amount = len(items)

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

        if amount <= 0:
            return recs, labels

        ret_recs = []
        ret_labels = []

        users = {rec['user'] for rec in recs}
        hist = self._user_history_interaction_item(users, end_date)

        encodable_items = self.__get_encodable_item_set()
        candidates = self.get_candidates()
        if encodable_items is not None:
            candidates = [
                item for item in candidates if item in encodable_items
            ]
        sampler = __CandidatesItemSampler(candidates)

        for rec, label in zip(recs, labels):
            ret_recs.append(rec)
            ret_labels.append(label)

            hist_user = hist.get(rec['user'], set())
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

        for data in self._iterate_data_in_date_range(*date_range):
            for row in pandas_helper.iterate_with_dict(data):
                if not self._is_like_rating(row['rating']):
                    continue

                if encodable_users is not None:
                    if row['userId'] not in encodable_users \
                        or row['movieId'] not in encodable_items:
                        continue
                recs.append({
                    'user': row['userId'],
                    'item': row['movieId'],
                    'time': row['timestamp'],
                })

        labels = [1] * len(recs)

        return recs, labels

    def _get_positive_user_item(self) -> Dict[int, List[Tuple[int, str]]]:
        """Get positive items with time of users.

        If 'user' and 'item' encoders were exist, filter record of which user or
            item is not encodable.

        Returns:
            dict: Positive items with time of users.
                {
                  USER_ID: [(ITEM_ID, 'TIME')],
                }
        """
        ret = {}

        if self.__cache_positive_user_item:
            return self.__cache_positive_user_item

        encodable_users = self.__get_encodable_user_set()
        encodable_items = self.__get_encodable_item_set()

        for data in self._iterate_all_data():
            for row in pandas_helper.iterate_with_dict(data):
                if not self._is_like_rating(row['rating']):
                    continue

                if encodable_users is not None:
                    if row['userId'] not in encodable_users \
                        or row['movieId'] not in encodable_items:
                        continue

                ret.setdefault(row['userId'], []) \
                    .append((row['movieId'], row['timestamp']))

        if encodable_users is not None:
            self.__cache_positive_user_item = ret

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
            if name == DINDataLoader.ITEM_INFO_GENOME_TAGS:
                ret[name] = {
                    id_: val[0]['tagId'] if val else None for id_, val in \
                        self.get_item_genome_tags().items()
                }
            elif name == DINDataLoader.ITEM_INFO_GENRES:
                ret[name] = self.get_item_genres()

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
                }

                If it is training for test, 'validation' will not be given.
        """
        self.__encoders = None
        self.__cache_positive_user_item = None

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
            feature_param['negative_sample_amount'])
        history_items = self._get_history_item(recs)

        features = {
            **self.__generate_feature_of_target(recs),
            **self.__generate_feature_of_history(history_items),
        }
        genres_mapping = None
        if self.__is_feature_genres_in_setting():
            features, genres_mapping = self.__process_genres(features)
            self.__genres = genres_mapping['genres']

        users, items = self.__get_user_item(recs)

        x = {
            'user': users,
            'item': items,
            'hist_item': history_items,
            **features,
        }

        self.__encoders = self.__generate_encoder(x)
        self.__encode(x)

        self.__convert_x_type(x)

        ret = {
            'x': x,
            'y': y,
            'history_feature_list': self.__get_history_feature_list(genres_mapping),
            'dnn_feature_columns': self.__generate_dnn_feature_columns(genres_mapping),
            'encoders': self.__encoders,
            'genres': self.__genres,
            'dataloader': copy.deepcopy(self),
        }

        if self._is_validation:
            ret['validation'] = self.__get_validation()

        return ret

    def _user_history_interaction_item(
        self,
        users: Union[List[int], Set[int]],
        end_date: str
    ) -> Dict[int, Set[int]]:
        """Get history interaction items of users before end date.

        Args:
            users (list|set): User IDs.
            end_date (str): End date but not included in format Y-m-d.

        Returns:
            dict: History action items of users.
                {
                  USER_ID: {ITEM_ID, ...},
                }
        """
        if not isinstance(users, set):
            users = set(users)

        ret = {}
        for data in self._iterate_all_data():
            for row in pandas_helper.iterate_with_dict(data):
                if row['timestamp'].split(' ')[0] >= end_date:
                    break
                elif row['userId'] not in users:
                    continue

                ret.setdefault(row['userId'], set()).add(row['movieId'])

        return ret

    def _val_feature_setting_comm(self, features: List[dict]) -> None:
        """Validate feature setting in common.

        Args:
            feature (list): List of features.

        Raises:
            InvalidFeatureSetting: If feature setting is invalid.
        """
        items = {
            DINDataLoader.ITEM_INFO_GENOME_TAGS,
            DINDataLoader.ITEM_INFO_GENRES,
        }
        marks = set()
        for feature in features:
            name = feature['feature_name']
            if name not in items:
                raise InvalidFeatureSetting(f"Invalid feature '{name}'")
            elif name in marks:
                raise InvalidFeatureSetting(
                    f"Feature '{name}' should be only one")

            marks.add(name)

    def _val_history_feature_setting(self, features: List[dict]) -> None:
        """Validate history feature setting.

        Args:
            feature (list): List of features.

        Raises:
            InvalidFeatureSetting: If feature setting is invalid.
        """
        self._val_feature_setting_comm(features)

    def _val_target_feature_setting(self, features: List[dict]) -> None:
        """Validate target feature setting.

        Args:
            feature (list): List of features.

        Raises:
            InvalidFeatureSetting: If feature setting is invalid.
        """
        self._val_feature_setting_comm(features)

    def __convert_x_type(self, x: Dict[str, list]) -> None:
        """Convert types of values in x into numpy.array.

        Args:
            x (dict): x.
        """
        for key, val in x.items():
            if isinstance(val, np.ndarray):
                continue
            x[key] = np.asarray(val)

    def __encode(self, x: Dict[str, list]) -> None:
        """Encode feature(s).

        Args:
            x (dict): x.
        """
        for item in ('user', 'item'):
            x[item] = self.__encoders[item].transform(x[item])

        encoder = self.__encoders['item']
        hist_item = x['hist_item']
        for i, row in enumerate(hist_item):
            for j, value in enumerate(row):
                if value is None:
                    row[j] = DINDataLoader.ENCODING_ITEM_PADDING_VALUE
            hist_item[i] = encoder.transform(row)

        self.__encode_item_info_genome_tags(x)

    def __encode_item_info_genome_tags(self, x: Dict[str, list]) -> None:
        """Encode genome tags.

        Args:
            x (dict): x.
        """
        NAME = DINDataLoader.ITEM_INFO_GENOME_TAGS

        padding_val = DINDataLoader.ENCODING_ITEM_INFO_GENOME_TAGS_PADDING_VALUE
        for setting in self._param['feature']['target']:
            if setting['feature_name'] != NAME:
                continue

            param_name = setting['deepctr']['param']['name']
            encoder = self.__encoders[param_name]
            encodable_set = set(encoder.classes_)
            features = x[param_name]
            for i, feature in enumerate(features):
                if feature is None or feature not in encodable_set:
                    feature = padding_val
                features[i] = encoder.transform([feature])[0]

        for setting in self._param['feature']['history']['features']:
            if setting['feature_name'] != NAME:
                continue

            param_name = setting['deepctr']['param']['sparsefeat']['name']
            encoder = self.__encoders[param_name]
            encodable_set = set(encoder.classes_)
            features = x[param_name]
            for row in features:
                for i, feature in enumerate(row):
                    if feature is None or feature not in encodable_set:
                        feature = padding_val
                    row[i] = encoder.transform([feature])[0]

    def __generate_dnn_feature_columns(
        self,
        genres_mapping: dict
    ) -> List[TypeDNNFeatureColumn]:
        """Generate DNN feature columns.

        Args:
            genres_mapping (dict): mapping of original genres feature name to genre
                feature names.
                {
                  'genres': [GENRE, ...],
                  'target': {
                    'name': 'NAME',
                    'mapping': ['GENRE_NAME', ...],
                  },
                  'hist': {
                    'name': 'NAME',
                    'mapping': [{
                        'name': 'GENRE_NAME',
                        'embedding_name': 'EMBEDDING_NAME',
                    }],
                  },
                }

        Returns:
            list: List of DNN feature columns.
        """
        ret = []

        feature_setting = self._param['feature']
        settings = [
            feature_setting[item] for item in ('user', 'item', 'hist_item')
        ]
        for setting in feature_setting['target']:
            if setting['feature_name'] == DINDataLoader.ITEM_INFO_GENRES:
                for row in genres_mapping['target']['mapping']:
                    new_setting = copy.deepcopy(setting)
                    new_setting['deepctr']['param']['name'] = row
                    settings.append(new_setting)

                continue

            settings.append(setting)

        for setting in feature_setting['history']['features']:
            if setting['feature_name'] == DINDataLoader.ITEM_INFO_GENRES:
                for row in genres_mapping['hist']['mapping']:
                    new_setting = copy.deepcopy(setting)
                    sparsefeat = new_setting['deepctr']['param']['sparsefeat']
                    sparsefeat['name'] = row['name']
                    sparsefeat['embedding_name'] = row['embedding_name']
                    settings.append(new_setting)

                continue

            settings.append(setting)

        for setting in settings:
            deepctr = setting['deepctr']
            name = deepctr['param']['sparsefeat']['name'] \
                    if deepctr['type'] == 'sparse_var_len' \
                    else deepctr['param']['name']

            if 'feature_name' in setting and \
                setting['feature_name'] == DINDataLoader.ITEM_INFO_GENRES:
                encoder = LabelEncoder()
                encoder.fit([0, 1])
            elif deepctr['type'] in ('sparse', 'sparse_var_len'):
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
        def __gen_item_info_genome_tags_encoder():
            ret = {}

            data = set()
            feature_names = []
            for item in ('target', 'history'):
                settings = self._param['feature'][item]
                if item == 'history':
                    settings = settings['features']
    
                for setting in settings:
                    if setting['feature_name'] != \
                        DINDataLoader.ITEM_INFO_GENOME_TAGS:
                        continue
    
                    deepctr = setting['deepctr']
                    feature_name = deepctr['param']['sparsefeat']['name'] \
                        if deepctr['type'] == 'sparse_var_len' \
                        else deepctr['param']['name']
                    for val in __iterate_feature_value(x[feature_name]):
                        if val is None:
                            continue

                        data.add(val)
    
                    feature_names.append(feature_name)
    
            if not feature_names:
                return ret
    
            data.add(DINDataLoader.ENCODING_ITEM_INFO_GENOME_TAGS_PADDING_VALUE)
            data = sorted(data)
    
            encoder = LabelEncoder()
            encoder.fit(data)
    
            for feature_name in feature_names:
                ret[feature_name] = encoder

            return ret

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

        for key, val in __gen_item_info_genome_tags_encoder().items():
            ret[key] = val

        return ret

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
        if not self.__cache_history_feature_data:
            self.__cache_history_feature_data = self._load_feature_data(features)
        data = self.__cache_history_feature_data

        for feature in features:
            name = feature['feature_name']
            if name == DINDataLoader.ITEM_INFO_GENOME_TAGS:
                method = self.__generate_feature_item_info_genome_tags
                params = lambda rec: {
                    'data': data[name],
                    'item': rec,
                }
            elif name == DINDataLoader.ITEM_INFO_GENRES:
                method = self.__generate_feature_item_info_genres
                params = lambda rec: {
                    'data': data[name],
                    'item': rec,
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
        if not self.__cache_target_feature_data:
            self.__cache_target_feature_data = self._load_feature_data(features)
        data = self.__cache_target_feature_data

        for feature in features:
            name = feature['feature_name']
            if name == DINDataLoader.ITEM_INFO_GENOME_TAGS:
                method = self.__generate_feature_item_info_genome_tags
                params = lambda rec: {
                    'data': data[name],
                    'item': rec['item'],
                }
            elif name == DINDataLoader.ITEM_INFO_GENRES:
                method = self.__generate_feature_item_info_genres
                params = lambda rec: {
                    'data': data[name],
                    'item': rec['item'],
                }

            ret[feature['deepctr']['param']['name']] = [
                method(**params(rec)) for rec in recs
            ]

        return ret

    def __generate_feature_item_info_genome_tags(
        self,
        item: Union[int, List[int]],
        data: Dict[int, Dict[str, List[str]]]
    ) -> Union[List[str], List[List[str]]]:
        """Generate feature of genome tags for item.

        Args:
            item (int|list): Item(s).
            data (dict): Item genome tags.

        Returns:
            list: Feature(s).
        """
        ret = []

        is_single = not isinstance(item, list)

        for item in [item] if is_single else item:
            ret.append(data.get(item, None))

        return ret[0] if is_single else ret

    def __generate_feature_item_info_genres(
        self,
        item: Union[int, List[int]],
        data: Dict[int, Dict[str, List[str]]]
    ) -> Union[List[str], List[List[str]]]:
        """Generate feature of genres for item.

        Args:
            item (int|list): Item(s).
            data (dict): Item genres.

        Returns:
            list: Feature(s).
        """
        ret = []

        is_single = not isinstance(item, list)

        for item in [item] if is_single else item:
            ret.append(data.get(item, []))

        return ret[0] if is_single else ret

    def __get_encodable_item_set(self) -> Union[Set[int], None]:
        """Get set of encodable items.

        Returns:
            set|None: Set of encodable items. If 'item' encoder was not exist,
                return None.
        """
        if not self.__encoders or 'item' not in self.__encoders:
            return None

        return set(self.__encoders['item'].classes_.tolist())

    def __get_encodable_user_set(self) -> Union[Set[int], None]:
        """Get set of encodable users.

        Returns:
            set|None: Set of encodable users. If 'user' encoder was not exist,
                return None.
        """
        if not self.__encoders or 'user' not in self.__encoders:
            return None

        return set(self.__encoders['user'].classes_.tolist())

    def __get_history_feature_list(self, genres_mapping: dict) -> List[str]:
        """Get list of history features.

        Args:
            genres_mapping (dict): mapping of original genres feature name to genre
                feature names.
                {
                  'genres': [GENRE, ...],
                  'target': {
                    'name': 'NAME',
                    'mapping': ['GENRE_NAME', ...],
                  },
                  'hist': {
                    'name': 'NAME',
                    'mapping': [{
                        'name': 'GENRE_NAME',
                        'embedding_name': 'EMBEDDING_NAME',
                    }],
                  },
                }

        Returns:
            list: List of history features.
        """
        ret = ['item']

        for setting in self._param['feature']['history']['features']:
            if setting['feature_name'] == DINDataLoader.ITEM_INFO_GENRES:
                for row in genres_mapping['hist']['mapping']:
                    ret.append(row['embedding_name'])

                continue

            ret.append(setting['deepctr']['param']['sparsefeat']['embedding_name'])

        return ret

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
        if self.__is_feature_genres_in_setting():
            features, _ = self.__process_genres(features)

        users, items = self.__get_user_item(recs)

        x = {
            'user': users,
            'item': items,
            'hist_item': history_items,
            **features
        }

        self.__encode(x)

        self.__convert_x_type(x)

        return {
            'x': x,
            'y': y,
        }

    def __is_feature_genres_in_setting(self) -> bool:
        """Check whether feature genres is in setting or not.

        Returns:
            Whether feature genres is in setting or not.
        """
        param = self._param['feature']
        for feature in itertools.chain(
            param['target'], param['history']['features']):
            if feature['feature_name'] == DINDataLoader.ITEM_INFO_GENRES:
                return True

        return False

    def __process_genres(self, features: dict) -> Tuple[dict, dict]:
        """Let each genre be a feature.

        Args:
            features (dict): Features.

        Returns:
            tuple: New features and mapping of original feature name to genre
                feature names.
                {
                  'FEATURE NAME': [...],
                },
                {
                  'genres': [GENRE, ...],
                  'target': {
                    'name': 'NAME',
                    'mapping': ['GENRE_NAME', ...],
                  },
                  'hist': {
                    'name': 'NAME',
                    'mapping': [{
                        'name': 'GENRE_NAME',
                        'embedding_name': 'EMBEDDING_NAME',
                    }],
                  },
                }
        """
        def __collect_genres(target, hist):
            ret = set()

            if self.__genres:
                return self.__genres

            if target:
                for row in features[target]:
                    for elem in row:
                        ret.add(elem)

            if hist:
                for row in features[hist]:
                    for hist_row in row:
                        for elem in hist_row:
                            ret.add(elem)

            return sorted(ret)

        def __generate_features(target, hist, hist_embedding, genres):
            ret_features = copy.deepcopy(features)
            ret_mapping = {}

            if target:
                mapping = [f'{target}_{genre}' for genre in genres]
                ret_mapping['target'] = {
                    'name': target,
                    'mapping': mapping,
                }

                template = {name: 0 for name in mapping}
                for row in features[target]:
                    vals = template.copy()
                    for genre in row:
                        vals[f'{target}_{genre}'] = 1

                    for name, val in vals.items():
                        ret_features.setdefault(name, []).append(val)

                del ret_features[target]

            if hist:
                mapping = [
                    {
                        'name': f'{hist}_{genre}',
                        'embedding_name': f'{hist_embedding}_{genre}',
                    } for genre in genres
                ]
                ret_mapping['hist'] = {
                    'name': hist,
                    'mapping': mapping,
                }

                template = {row['name']: 0 for row in mapping}
                for row in features[hist]:
                    new_row = {}
                    for hist_row in row:
                        vals = template.copy()
                        for genre in hist_row:
                            vals[f'{hist}_{genre}'] = 1

                        for name, val in vals.items():
                            new_row.setdefault(name, []).append(val)

                    for name, val in new_row.items():
                        ret_features.setdefault(name, []).append(val)

                del ret_features[hist]

            return ret_features, ret_mapping

        def __get_names():
            param = self._param['feature']
    
            target = None
            for row in param['target']:
                if row['feature_name'] == DINDataLoader.ITEM_INFO_GENRES:
                    target = row['deepctr']['param']['name']
                    break

            hist = None
            hist_embedding = None
            for row in param['history']['features']:
                if row['feature_name'] == DINDataLoader.ITEM_INFO_GENRES:
                    param = row['deepctr']['param']['sparsefeat']
                    hist = param['name']
                    hist_embedding = param['embedding_name']
                    break

            return target, hist, hist_embedding

        target, hist, hist_embedding = __get_names()
        genres = __collect_genres(target, hist)

        ret_features, ret_mapping = __generate_features(
            target, hist, hist_embedding, genres)

        ret_mapping['genres'] = genres

        return ret_features, ret_mapping

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
            tuple: Features of rows of user and item, and indices of missing
                features.
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
        if self.__is_feature_genres_in_setting():
            features, _ = self.__process_genres(features)

        users, items = self.__get_user_item(recs)

        x = {
            'user': users,
            'item': items,
            'hist_item': history_items,
            **features,
        }

        self.__encode(x)

        self.__convert_x_type(x)

        return x, []
