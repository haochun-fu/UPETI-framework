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

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from comm_func import CommFunc
from dataloader_helper import InteractionHelper
from date_helper import DateHelper
from movielens_helper import MovielensHelper
import pandas_helper


__author__ = 'haochun.fu'
__date__ = '2020-07-04'


TypeAUCItem = Dict[int, Dict[str, List[int]]]


class DataLoaderException(Exception):
    """Base exception of DataLoader."""
    pass


class InvalidFeatureSetting(DataLoaderException):
    """Invalid feature setting."""
    pass


class DataLoaderBase(abc.ABC):
    """Basis of DataLoader to be inherited."""


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
        self._data_dir = data_dir
        self._param = param
        self._test_date  = test_date
        self._is_validation = is_validation
        self._is_test = is_test

        self._test_date_obj = DateHelper.str_to_datetime(self._test_date)

    def _get_train_range(self) -> Tuple[str, str]:
        """Get date range for training data.

        Returns:
            tuple: Start and end date in format Y-m-d.
        """
        start_date_obj = self._test_date_obj \
                         - datetime.timedelta(days=self._param['duration'])
        end_date_obj = self._test_date_obj - datetime.timedelta(days=1)

        return start_date_obj.strftime('%Y-%m-%d'), \
               end_date_obj.strftime('%Y-%m-%d')

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


class MovielensDataLoaderBase(DataLoaderBase):
    """Basis of data loader for movielens to be inherited."""


    """Directory name of AUC item."""
    AUC_ITEM_DIR_NAME = 'auc_item'

    """Extension of AUC item file."""
    AUC_ITEM_FILE_EXT = '.json'

    """Directory name of candidates."""
    CANDIDATES_DIR_NAME = 'candidates'

    """File name of candidates."""
    CANDIDATES_FILE_NAME = 'movies.csv'

    """Directory name of filter for candidates."""
    CANDIDATES_FILTER_DIR_NAME = 'filter'

    """Extension of candidates filter file."""
    CANDIDATES_FILTER_FILE_EXT = '.json'

    """Directory name of features."""
    FEATURES_DIR_NAME = 'features'

    """Extension of feature file."""
    FEATURE_FILE_EXT = '.csv'

    """Timestamp format of feature."""
    FEATURE_TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S'

    """Directory name of item information."""
    ITEM_INFO_DIR_NAME = 'item_info'

    """File name of genome tags for item information."""
    ITEM_INFO_GENOME_TAGS_FILE_NAME = 'genome_tags.json'

    """File name of genres for item information."""
    ITEM_INFO_GENRES_FILE_NAME = 'genres.json'

    """Directory name of item vector for item information."""
    ITEM_INFO_VECTOR_DIR_NAME = 'vector/item'

    """Items of item vector setting."""
    ITEM_VECTOR_SETTING_ITEMS = (
        'model',
        'exp_no',
        'normalization',
        'scale',
    )

    """Extension of item vector."""
    ITEM_VECTOR_FILE_EXT = '.json'

    """Prefix file name of item vector."""
    ITEM_VECTOR_FILE_NAME_PREFIX = 'item'

    """Lower bound of rating for liking."""
    LIKE_RATING_LOWER_BOUND = 4


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
        super().__init__(data_dir, param, test_date, is_validation, is_test)

        candidates_dir = os.path.join(
            self._data_dir, MovielensDataLoaderBase.CANDIDATES_DIR_NAME)
        self._candidates_file = os.path.join(
            candidates_dir, MovielensDataLoaderBase.CANDIDATES_FILE_NAME)
        self._candidates_filter_dir = os.path.join(
            candidates_dir, MovielensDataLoaderBase.CANDIDATES_FILTER_DIR_NAME)
        self._features_dir = os.path.join(
            self._data_dir, MovielensDataLoaderBase.FEATURES_DIR_NAME)
        self._item_info_dir = os.path.join(
            self._data_dir, MovielensDataLoaderBase.ITEM_INFO_DIR_NAME)
        self._item_genome_tags_file = os.path.join(
            self._item_info_dir,
            MovielensDataLoaderBase.ITEM_INFO_GENOME_TAGS_FILE_NAME)
        self._item_genres_file = os.path.join(
            self._item_info_dir,
            MovielensDataLoaderBase.ITEM_INFO_GENRES_FILE_NAME)
        self._item_vector_dir = os.path.join(
            self._item_info_dir,
            MovielensDataLoaderBase.ITEM_INFO_VECTOR_DIR_NAME)
        self._auc_item_dir = os.path.join(
            self._data_dir, MovielensDataLoaderBase.AUC_ITEM_DIR_NAME)

    def _get_auc_item_by_date(self, date: str) -> TypeAUCItem:
        """Get AUC item by date.

        Args:
            date (str): Date.

        Returns:
            dict: AUC item for date.
                {
                  USER_ID: {
                    'neg': [ITEM_ID, ...],
                    'pos': [ITEM_ID, ...],
                  },
                }
        """
        ret = {}

        for key, val in CommFunc.load_json(
            os.path.join(
                self._auc_item_dir,
                date + MovielensDataLoaderBase.AUC_ITEM_FILE_EXT)).items():
            ret[int(key)] = val

        return ret

    def _get_candidates_filter_file_by_date(self, date: str) -> str:
        """Get candidates filter file path on certain date.

        Args:
            date (str): Date in format Y-m-d.

        Returns:
            str: Candidates filter file path.
        """
        return os.path.join(
            self._candidates_filter_dir,
            f'{date}{MovielensDataLoaderBase.CANDIDATES_FILTER_FILE_EXT}')

    def _get_candidates_filter_by_date(self, date: str) -> Dict[int, List[int]]:
        """Get feature on certain date.

        Args:
            date (str): Date in format Y-m-d.

        Returns:
            dict: Data.
        """
        return {
            int(key): val for key, val in \
                CommFunc \
                    .load_json(self._get_candidates_filter_file_by_date(date)) \
                    .items()
        }

    def _get_feature_by_date(self, date: str) -> pd.core.frame.DataFrame:
        """Get feature on certain date.

        Args:
            date (str): Date in format Y-m-d.

        Returns:
            pandas.core.frame.DataFrame: Data.
        """
        return MovielensHelper.convert_ratings_column_type(
            pd.read_csv(self._get_feature_file_by_date(date)))

    def _get_feature_file_by_date(self, date: str) -> str:
        """Get feature file path on certain date.

        Args:
            date (str): Date in format Y-m-d.

        Returns:
            str: Feature file path.
        """
        return os.path.join(
            self._features_dir,
            f'{date}{MovielensDataLoaderBase.FEATURE_FILE_EXT}')

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

    def _get_test_data(self) -> Dict[int, List[int]]:
        """Get item each user likes on test date.

        Filter 

        Returns:
            dict: Item each user likes on test date.
                {
                  USER_ID: [ITEM_ID, ...],
                }
        """
        ret = {}

        for row in pandas_helper.iterate_with_dict(
            self._get_feature_by_date(self._test_date)):
            if not self._is_like_rating(row['rating']):
                continue

            ret.setdefault(int(row['userId']), set()).add(int(row['movieId']))

        for key, val in ret.items():
            ret[key] = sorted(val)

        return ret

    def _get_test_auc_item(self) -> TypeAUCItem:
        """Get AUC item for test date.

        Filter 

        Returns:
            dict: AUC item for test date.
                {
                  USER_ID: {
                    'neg': [ITEM_ID, ...],
                    'pos': [ITEM_ID, ...],
                  },
                }
        """
        return self._get_auc_item_by_date(self._test_date)

    def _is_like_rating(self, rating: float) -> bool:
        """Check whether rating represent liking or not.

        Args:
            rating (float): Rating.

        Returns:
            bool: Whether rating represent liking or not.
        """
        return rating >= MovielensDataLoaderBase.LIKE_RATING_LOWER_BOUND

    def _iterate_all_data(
        self
    ) -> Generator[pd.core.frame.DataFrame, None, None]:
        """Generator of iterating all data in order of date.

        Yields:
            pandas.core.frame.DataFrame: Data on next date.
        """
        for file in sorted(glob.glob(os.path.join(self._features_dir, '*'))):
            yield MovielensHelper.convert_ratings_column_type(pd.read_csv(file))

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

    def _scale_item_vector(
        self,
        data: Dict[int, Dict[str, List[float]]],
        scale: float
    ) -> Dict[int, List[float]]:
        """Scale item vector.

        Args:
            data (dict): Item vector.
            scale (float): Scale.

        Returns:
            dict: Scaled item vector.
                {
                    ITEM_ID: [...],
                }
        """
        ret = {}
        for key, val in data.items():
            ret[key] = [val_ * scale for val_ in val] if val else val

        return ret

    def get_candidates(self) -> List[int]:
        """Get candidates.

        Returns:
            dict: Candidates for each user.
        """
        return pd.read_csv(self._candidates_file)['movieId'].tolist()

    def get_candidates_filter(self) -> Dict[int, List[int]]:
        """Get candidates filter on test date.

        Returns:
            dict: Candidates filter for each user on test date.
        """
        return self._get_candidates_filter_by_date(self._test_date)

    def get_item_genome_tags(self) -> Dict[int, List[dict]]:
        """Get genome tags of items.

        Returns:
            dict: Genome tags of items.
                {
                  MOVIE_ID: [{
                    'tagId': TAG_ID,
                    'relevance': RELEVANCE_SCORE,
                  }]
                }
        """
        return {
            int(id_): val for id_, val in \
                CommFunc.load_json(self._item_genome_tags_file).items()
        }

    def get_item_genres(self) -> Dict[int, List[str]]:
        """Get genres of items.

        Returns:
            dict: Genres of items.
                {
                  MOVIE_ID: [GENRE, ...],
                }
        """
        return {
            int(key): val for key, val in \
                CommFunc.load_json(self._item_genres_file).items()
        }

    def get_item_vector( 
        self,
        model: str,
        exp_no: Union[int, str],
        normalization: str
    ) -> Dict[int, List[float]]:
        """Get item vectors.

        Args:
            model (str): Model name.
            exp_no (int|str): Experiment no.
            normalization (str): Normalization name.

        Returns:
            dict: Item vectors.
                {
                    ITEM_ID: [...],
                }
        """
        file = os.path.join(
            self._item_vector_dir,
            model,
            f"exp{exp_no}",
            f'{MovielensDataLoaderBase.ITEM_VECTOR_FILE_NAME_PREFIX}'
            f"{'_' + normalization if normalization else ''}"
            f'{MovielensDataLoaderBase.ITEM_VECTOR_FILE_EXT}')
        return {int(id_): val for id_, val in CommFunc.load_json(file).items()}

    def get_item_vector_by_setting(
        self,
        model: str,
        exp_no: Union[int, str],
        normalization: str,
        scale: float
    ) -> Dict[int, List[float]]:
        """Get item vector by setting.

        Args:
            model (str): Model name.
            exp_no (int|str): Experiment no.
            normalization (str): Normalization name.
            scale (float): Scale.

        Returns:
            dict: Item vectors.
                {
                    ITEM_ID: [...],
                }
        """
        return self._scale_item_vector(
            self.get_item_vector(model, exp_no, normalization), scale)

    def get_user_item_interaction_time(
        self
    ) -> Dict[int, Dict[int, str]]:
        """Get interaction time of user and item.

        Returns:
            dict: Interaction time of user and item.
                {
                  USER_ID: {
                    ITEM_ID: 'TIME',
                  },
                }
        """
        ret = {}

        for data in self._iterate_all_data():
            for row in pandas_helper.iterate_with_dict(data):
                ret.setdefault(row['userId'], {})['movieId'] = row['timestamp']

        return ret

    def load_test_auc_data(self) -> dict:
        """Load test AUC data.

        Returns:
            dict: Test AUC data.
                {
                  USER_ID: {
                    'items': [ITEM_ID, ...],
                    'labels': [LABEL, ...]
                  },
                }
        """
        ret = {}

        auc_item = self._get_test_auc_item()
        for user, pos_items in self._get_test_data().items():
            user_auc_item = auc_item.get(user)
            if not user_auc_item and user_auc_item['neg']:
                continue

            ret[user] = {
                'items': pos_items + user_auc_item['neg'],
                'labels': [1] * len(pos_items) + \
                          [0] * len(user_auc_item['neg']),
            }

        return ret


class MatrixDataLoader(MovielensDataLoaderBase):
    """Data loader for Matrix."""


    def _get_ids_from_interactive_data(self, data: dict) -> Tuple[list, list]:
        """Get user ids and item ids from interactive data.

        Args:
            data (dict): Interactive data.

        Returns:
            tuple: List of sorted user ids and list of sorted item ids.
        """
        user_ids = []
        item_ids = set()
        for user, items in data.items():
            user_ids.append(user)
            for item in items:
                item_ids.add(item)

        return sorted(user_ids), sorted(item_ids)

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
                    ITEM_ID: PREFERENCE,
                  },
                }
        """
        ret = {}

        fmt = '%Y-%m-%d'

        start_date_obj = datetime.datetime.strptime(date_range[0], fmt)
        end_date_obj = datetime.datetime.strptime(date_range[1], fmt)

        label_param = self._param['label']

        interaction_helper = None
        if label_param['time_weight']:
            interaction_helper = InteractionHelper(
                (end_date_obj-start_date_obj).days+1,
                (end_date_obj + datetime.timedelta(days=1)).strftime(fmt) \
                    + ' 00:00:00',
                label_param['time_weight'])
 
        for date in CommFunc.iterate_date_range(*date_range):
            for row in pandas_helper.iterate_with_dict(
                        self._get_feature_by_date(date)):
                if row['rating'] < label_param['rating_lower_bound']:
                    continue

                ret.setdefault(row['userId'], {})[row['movieId']] = \
                    interaction_helper.preference(
                        row['rating'], row['timestamp']) \
                        if interaction_helper \
                        else row['rating']

        return ret

    def _get_item_features(
        self,
        item_ids: List[int]
    ) -> Tuple[List[List[float]], List[int]]:
        """Get item features.

        Args:
            item_ids (list): List of item ids.

        Returns:
            tuple: Item features and item ids in item features.
        """
        param = self._param['item']

        ids = item_ids
        features = None
        if param.get('vector'):
            features, ids = self._get_item_vector_by_id(ids)

        for info_item, method in \
            (
                ('genres', self._get_item_genres_by_id),
                ('genome_tags', self._get_item_genome_tags_by_id),
            ):
            if not param.get(info_item):
                continue

            infos = method(ids)
            if not features:
                features = infos
                continue

            for i, row in enumerate(infos):
                features[i] += row

        return features, ids

    def _get_item_genome_tags_by_id(self, ids: List[int]) -> List[List[int]]:
        """Get item genome tags of item ids.

        Args:
            ids (list): List of item ids.

        Returns:
            list: List of encoded genome tags.
        """
        lower_bound = \
            self._param['item']['genome_tags']['relevance_lower_bound']

        info = self.get_item_genome_tags()

        data = []
        for id_ in ids:
            id_info = info.get(id_)
            if not id_info:
                data.append([])
                continue

            data.append([
                row['tagId'] for row in id_info \
                    if row['relevance'] >= lower_bound
            ])

        return MultiLabelBinarizer().fit_transform(data).tolist()

    def _get_item_genres_by_id(self, ids: List[int]) -> List[List[int]]:
        """Get item genres of item ids.

        Args:
            ids (list): List of item ids.

        Returns:
            list: List of encoded genres.
        """
        info = self.get_item_genres()

        data = [info[id_] for id_ in ids]

        return MultiLabelBinarizer().fit_transform(data).tolist()

    def _get_item_vector_by_id(
        self,
        item_ids: List[int]
    ) -> Tuple[dict, List[int]]:
        """Get item vectors of item ids.

        If item does not have vector, ignores.

        Args:
            item_ids (list): List of item ids.

        Returns:
            tuple: Item vectors and item ids of item vectors.
        """
        ids = []
        vecs = []

        data = self.get_item_vector_by_setting(**{
            item: self._param['item']['vector'][item] for item in \
                MovielensDataLoaderBase.ITEM_VECTOR_SETTING_ITEMS
        })
        for id_ in item_ids:
            if id_ not in data or not data[id_]:
                continue

            vecs.append(data[id_])
            ids.append(id_)

        return vecs, ids

    def load_test_data(self):
        """Load test data.

        Returns:
            dict: Test data.
        """
        return self._get_test_data()
