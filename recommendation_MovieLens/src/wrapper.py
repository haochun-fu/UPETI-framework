# -*- coding: utf-8 -*-
"""WrapperBase class."""
import abc
import copy
import multiprocessing as mp
import os
import time
from typing import (
    Any,
    Dict,
    List,
    Union,
)

import numpy as np

from comm_func import CommFunc
from encoder import Encoder
from dataloader import TypeAUCItem


__author__ = 'haochun.fu'
__date__ = '2020-07-06'


class WrapperBase(abc.ABC):
    """Basis of Wrapper to be implemented."""


    """Model information filename."""
    MODEL_INFO_FILENAME = "model_info.json"

    """Default prediction for new user."""
    DEFAULT_PREDICTION_FILENAME = "default_prediction.json"


    def __init__(self) -> None:
        """Constructor."""
        self._model = None
        self._model_info = None
        self._default_prediction = None

    def evaluate_auc(
        self,
        data: dict,
        dataloader: Any = None,
        n_processes: int = 1,
        data_info: Union[dict, None] = None
    ) -> dict:
        """Evaluate AUC.

        Args:
            data (dict): Items each user likes.
                {
                  USER_ID: {
                    'items': [ITEM_ID, ...],
                    'labels': [LABEL, ...]
                  },
                }
            dataloader (any): Data loader. Default is None.
            n_processes (int): The number of processes. Default is 1.
            data_info (dict): Data information.
                {
                  'data_dir': 'Data directory',
                  'test_date': 'Test date',
                  'data_config': Data configuration,
                }

        Returns:
            dict: Result.
        """
        def __multiproces(rows):
            total = len(rows)
            pile_amount = int(total / n_processes)

            if n_processes == 1 or pile_amount == 0:
                ret = {}
                no = 0
                __multiproces_task(no, ret, rows)

                return ret[no]

            p_ret = mp.Manager().dict()
            ps = []
            for i in range(n_processes):
                start = i * pile_amount
                end = (i + 1) * pile_amount
                if i + 1 == n_processes:
                    end = total

                pile = rows[start:end]
                if not pile:
                    break

                p = mp.Process(target=__multiproces_task, args=(i, p_ret, pile))
                p.start()
                ps.append(p)

            for p in ps:
                p.join()

            idx = 0
            for i in range(len(ps)):
                for res_row in p_ret[i]:
                    rows[idx][3] = res_row[3]

                    idx += 1

            return rows

        def __multiproces_task(no, ret, rows):
            params = {
                'dataloader': dataloader,
            }
            if data_info:
                params['data_info'] = data_info

            for row in rows:
                user = row[0]

                params['user_ids'] = user
                params['item_ids'] = row[1]

                row[3] = self.predict(**params)
            
            ret[no] = rows

        total = sum(row['labels'].count(1) for row in data.values())
        ret = {
            'elapsed_time': {
                'evaluation': {
                    'second': None,
                    'human_readable': None,
                },
            },
            'evaluation': {
                'total': total,
                'recommendable_amount': 0,
                'auc': None,
            },
            'amount': {
                'user': len(data),
                'new_user': 0,
            },
            # user: {
            #   'auc': AUC,
            # }
            'record': {},
        }

        ret_amount = ret['amount']
        ret_record = ret['record']
        ret_evaluation = ret['evaluation']

        start_time = time.time()

        rows = []
        for user, row in data.items():
            # user, items, labels, prediction
            rows.append([user, row['items'], row['labels'], None])

        rows = __multiproces(rows)

        auc_total = 0
        for user, _, labels, predictions in rows:
            if predictions is None:
                ret_amount['new_user'] += 1
                continue

            # prediction, label
            org_predictions = []
            for label, prediction in zip(labels, predictions):
                if prediction is None:
                    continue

                org_predictions.append((prediction, label))
            org_predictions.sort(key=lambda row: row[0])

            pos_rank_sum = 0
            pos_amount = 0
            for rank, (_, label) in enumerate(org_predictions, 1):
                if label == 0:
                    continue

                pos_amount += 1
                pos_rank_sum += rank

            if pos_amount == 0 or len(org_predictions) == pos_amount:
                continue

            auc = (pos_rank_sum - pos_amount * (pos_amount + 1) / 2) \
                  / (pos_amount * (len(org_predictions) - pos_amount))

            ret_record[user] = {
                'auc': auc,
            }

            ret_evaluation['recommendable_amount'] += pos_amount

            auc_total += auc

        amount = len(ret_record.keys())
        ret_evaluation['auc'] = auc_total / amount if amount else 0

        elapsed = ret['elapsed_time']['evaluation']
        elapsed['second'] = time.time() - start_time
        elapsed['human_readable'] = CommFunc.second_to_time(elapsed['second'])

        return ret

    def evaluate_recall(
        self,
        data: dict,
        candidates: List[int],
        top_ns: List[int],
        candidates_filter: Dict[int, List[int]] = {},
        dataloader: Any = None,
        n_processes: int = 1
    ) -> dict:
        """Evaluate recall.

        Args:
            data (dict): Items each user likes.
                {
                  USER_ID: [ITEM_ID, ...]
                }
            candidates (list): Candidates.
            top_ns (list): List of top N.
            candidates_filter (dict): Candidates filter for each user. Default
                is empty.
            dataloader (any): Data loader. Default is None.
            n_processes (int): The number of processes. Default is 1.

        Returns:
            dict: Result.
        """
        def __multiproces(rows):
            total = len(rows)
            pile_amount = int(total / n_processes)

            if n_processes == 1 or pile_amount == 0:
                ret = {}
                no = 0
                __multiproces_task(no, ret, rows)

                return ret[no]

            p_ret = mp.Manager().dict()
            ps = []
            for i in range(n_processes):
                start = i * pile_amount
                end = (i + 1) * pile_amount
                if i + 1 == n_processes:
                    end = total

                pile = rows[start:end]
                if not pile:
                    break

                p = mp.Process(target=__multiproces_task, args=(i, p_ret, pile))
                p.start()
                ps.append(p)

            for p in ps:
                p.join()

            ret_users_candidates = {}
            idx = 0
            for i in range(len(ps)):
                res_rows, res_users_candidates = p_ret[i]

                for res_row in res_rows:
                    row = rows[idx]

                    row[2] = res_row[2]
                    row[3] = res_row[3]

                    idx += 1

                ret_users_candidates.update(res_users_candidates)

            return rows, ret_users_candidates

        def __multiproces_task(no, ret, rows):
            ret_users_candidates = {}

            for row in rows:
                user = row[0]

                user_candidates = self.filter_candidates(
                    candidates, candidates_filter.get(user, []))
            
                row[2] = self.predict(
                    user, user_candidates, dataloader=dataloader)
                if self.has_default_prediction():
                    row[3] = self.predict(
                        user,
                        user_candidates,
                        use_default=True,
                        dataloader=dataloader)
            
                ret_users_candidates[user] = user_candidates

            ret[no] = (rows, ret_users_candidates)

        total = sum(len(items) for items in data.values())
        result = {
            'elapsed_time': {
                'evaluation': {
                    'second': None,
                    'human_readable': None,
                },
            },
            'evaluation': {
                'total': total,
                'recommendable_amount': 0,
                'top_ns': {
                    # n: {'hit': HIT, 'recall': RECALL}
                    'predict_without_new_user': {},
                    'recommendable': {},
                },
            },
            'amount': {
                'user': len(data),
                'new_user': 0,
                'avg_user_candidates': 0,
            },
            # USER_ID: {
            #   'new_user': TRUE|FALSE,
            #   'recall': {TOP_N: RECALL, ...},
            #   'like': [[ITEM_ID, ORDER], ...],
            #   'recommendation': [ITEM_ID, ...],
            # }
            'record': {},
        }

        res_amount = result['amount']

        result_top_ns = result['evaluation']['top_ns']
        if self.has_default_prediction():
            top_ns_including_new_user = {}
            result_top_ns['predict_including_new_user'] = \
                top_ns_including_new_user

        start_time = time.time()

        top_ns = sorted(top_ns)
        max_top_n = max(top_ns)
        top_ns_without_new_user = result_top_ns['predict_without_new_user']
        for result_top_n in result_top_ns.values():
            for n in top_ns:
                result_top_n[n] = {
                    'hit': 0,
                    'recall': None,
                }

        rows = []
        for user, items in data.items():
            rows.append([user, items, None, None])

        #users_candidates = {}
        #for row in rows:
        #    user = row[0]
        #    user_candidates = self._filter_candidates(
        #        candidates, candidates_filter.get(user, []))

        #    row[2] = self.predict(user, user_candidates, dataloader=dataloader)
        #    if self.has_default_prediction():
        #        row[3] = self.predict(
        #            user,
        #            user_candidates,
        #            use_default=True,
        #            dataloader=dataloader)

        #    users_candidates[user] = user_candidates
        rows, users_candidates = __multiproces(rows)

        recommendable_amount = 0
        for user, items, predictions, predictions_default in rows:
            new_user = False
            if predictions is None:
                new_user = True
                res_amount['new_user'] += 1
                if self.has_default_prediction():
                    predictions = predictions_default
                else:
                    continue

            user_candidates = users_candidates[user]
            user_candidates_len = len(user_candidates)
            res_amount['avg_user_candidates'] += user_candidates_len
            no_order_val = user_candidates_len + 1

            # [(ITEM_ID, PREDICTION), ...]
            org_predictions = []
            for item, prediction in zip(user_candidates, predictions):
                if prediction is None:
                    continue
                org_predictions.append((item, prediction))
            org_predictions.sort(key=lambda row: row[1], reverse=True)

            user_top_ns = {n: 0 for n in top_ns}
            like = {item: None for item in items}
            for order, (item, _) in enumerate(org_predictions, 1):
                if item not in like:
                    continue
                like[item] = order
                if not new_user:
                    recommendable_amount += 1
                if order > max_top_n:
                    continue
                for n in top_ns:
                    if order <= n:
                        user_top_ns[n] += 1

            like_amount = len(like)
            like = [(id_, order) for id_, order in like.items()]
            like.sort(key=lambda row: row[1] if row[1] is not None \
                                         else no_order_val)
            result['record'][user] = {
                'new_user': new_user,
                'recall': {n: amount / like_amount \
                           for n, amount in user_top_ns.items()},
                'like': like,
                'recommendation': [row[0] \
                                   for row in org_predictions[:max_top_n]],
            }

            result_top_ns_s = []
            if self.has_default_prediction():
                result_top_ns_s.append(top_ns_including_new_user)
            if not new_user:
                result_top_ns_s.append(top_ns_without_new_user)
            for result_top_ns in result_top_ns_s:
                for n, row in result_top_ns.items():
                    row['hit'] += user_top_ns[n]

        result_top_ns_s = [top_ns_without_new_user]
        if self.has_default_prediction():
            result_top_ns_s.append(top_ns_including_new_user)
        for result_top_ns in result_top_ns_s:
            for row in result_top_ns.values():
                row['recall'] = row['hit'] / total

        top_ns_recommendable = copy.deepcopy(top_ns_without_new_user)
        for row in top_ns_recommendable.values():
            row['recall'] = row['hit'] / recommendable_amount
        result['evaluation']['top_ns']['recommendable'] = top_ns_recommendable
        result['evaluation']['recommendable_amount'] = recommendable_amount

        res_amount['avg_user_candidates'] /= \
            res_amount['user'] - res_amount['new_user']

        elapsed_time = result['elapsed_time']['evaluation']
        elapsed_time['second'] = time.time() - start_time
        elapsed_time['human_readable'] = \
            CommFunc.second_to_time(elapsed_time['second'])

        return result

    def filter_candidates(self, candidates: list, filter_: list) -> list:
        """Filter candidates.

        Args:
            candidates (list): Candidates.
            filter_ (list): Lite of items to be filtered.

        Returns:
            list: Filtered Candidates.
        """
        filter_ = set(filter_)

        return [item for item in candidates if item not in filter_]

    @abc.abstractmethod
    def fit(
        self,
        data: dict,
        config: dict,
        is_gen_default_prediction: bool = False
    ) -> None:
        """Train.

        Args:
            data (dict): Training data and data related to training data.
            config (dict): Training configuration.
            is_gen_default_prediction (bool): Whether generate default
                prediction. Default is False.
        """
        return NotImplemented

    def get_data_for_dataloader(self) -> dict:
        """Get data for dataloader.

        Returns:
            dict: Data for dataloader.
        """
        return {}

    @abc.abstractmethod
    def get_dataloader_class(self) -> Any:
        """Get dataloader class.

        Returns:
            DataLoaderClass: DataLoader class.
        """
        return NotImplemented

    def has_default_prediction(self) -> bool:
        """Whether has default prediction or not.

        Returns:
            bool: Whether has default prediction or not.
        """
        return self._default_prediction is not None

    def load(self, model_dir: str) -> None:
        """Load model.

        Args:
            model_dir (str): Model directory.
        """
        self._model_info = CommFunc.load_json(
            os.path.join(model_dir, WrapperBase.MODEL_INFO_FILENAME))

        file = os.path.join(model_dir,
                            WrapperBase.DEFAULT_PREDICTION_FILENAME)
        self._default_prediction = CommFunc.load_json(file) \
                                   if os.path.isfile(file) \
                                   else None

    @abc.abstractmethod
    def predict(
        self,
        user_ids: int,
        item_ids: np.ndarray,
        use_default: bool = False,
        dataloader: Any = None
    ) -> Union[List, None]:
        """Predict.

        Args:
            user_id (int): User id.
            item_ids (np.int32 array): Item ids.
            use_default: Whether use default prediction for new user or not.
                Default is false.
            dataloader (any): Data loader to generate features for prediction.

        Returns:
            list|None: Prediction. If some item is not able to be predicted,
                this value is set to None. None if user_id is not able to
                predict.
        """
        return NotImplemented

    def save(self, model_dir: str) -> None:
        """Save model.

        Args:
            model_dir (str): Model directory.
        """
        file = os.path.join(model_dir, WrapperBase.MODEL_INFO_FILENAME)
        CommFunc.save_data(self._model_info, file, save_type='json')

        if self.has_default_prediction() \
            and self._default_prediction is not None:
            file = os.path.join(model_dir,
                                WrapperBase.DEFAULT_PREDICTION_FILENAME)
            CommFunc.save_data(self._default_prediction,
                               file,
                               save_type='json')

    @property
    def model_info(self) -> dict:
        return copy.deepcopy(self._model_info)


class MatrixWrapper(WrapperBase):
    """Wrapper for matrix."""


    """Model filename."""
    MODEL_FILENAME = "model.pickle"

    """User ID encoder filename."""
    USER_ID_ENCODER_FILENAME = "user_id_encoder.pickle"

    """Item ID encoder filename."""
    ITEM_ID_ENCODER_FILENAME = "item_id_encoder.pickle"

    """Item features filename."""
    ITEM_FEATURES_FILENAME = "item_features.pickle"


    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

        self._user_id_encoder = None
        self._item_id_encoder = None
        self._item_features = None

    def gen_default_prediction(self) -> None:
        """Generate default prediction for new user."""
        start_time = time.time()

        user_ids = self._user_id_encoder.origin
        item_ids = self._item_id_encoder.origin

        predictions = self.predict(user_ids, item_ids)

        self._default_prediction = np.asarray(predictions, dtype=np.float64) \
                                    .mean(axis=0) \
                                    .tolist()

        sec = time.time() - start_time
        self._model_info['elapsed_time']['generate_default_prediction'] = {
            'second': sec,
            'human_readable': CommFunc.second_to_time(sec),
        }

    def load(self, model_dir: str) -> None:
        """Load model.

        Args:
            model_dir (str): Model directory.
        """
        super().load(model_dir)

        self._model = CommFunc.load_pickle(
            os.path.join(model_dir, MatrixWrapper.MODEL_FILENAME))

        self._user_id_encoder = CommFunc.load_pickle(
            os.path.join(model_dir, MatrixWrapper.USER_ID_ENCODER_FILENAME))

        self._item_id_encoder = CommFunc.load_pickle(
            os.path.join(model_dir, MatrixWrapper.ITEM_ID_ENCODER_FILENAME))

        file = os.path.join(model_dir, MatrixWrapper.ITEM_FEATURES_FILENAME)
        self._item_features = CommFunc.load_pickle(file) \
                              if os.path.isfile(file) \
                              else None

    def save(self, model_dir: str) -> None:
        """Save model.

        Args:
            model_dir (str): Model directory.
        """
        super().save(model_dir)

        CommFunc.save_data(
            self._model,
            os.path.join(model_dir, MatrixWrapper.MODEL_FILENAME),
            save_type='pickle')

        CommFunc.save_data(
            self._user_id_encoder,
            os.path.join(model_dir, MatrixWrapper.USER_ID_ENCODER_FILENAME),
            save_type='pickle')

        CommFunc.save_data(
            self._item_id_encoder,
            os.path.join(model_dir, MatrixWrapper.ITEM_ID_ENCODER_FILENAME),
            save_type='pickle')

        if self._item_features is not None:
            CommFunc.save_data(
                self._item_features,
                os.path.join(model_dir, MatrixWrapper.ITEM_FEATURES_FILENAME),
                save_type='pickle')

    def _organize_user_item_for_predict(
        self,
        user_ids: Union[int, List[int]],
        item_ids: Union[int, List[int]]
    ) -> Dict[str, dict]:
        """Organize user ids and item ids for prediction.

        Args:
            user_ids (int|list): User id(s).
            item_ids (int|list): Item id(s).

        Returns:
            dict: Organized data.
                {
                  'miss': {
                    'user': [user_index],
                    'item': [item_index],
                  },
                  'encode': {
                    'user': [encoded_user_id],
                    'item': [encoded_item_id],
                  },
                }
        """
        miss = {
            'user': [],
            'item': [],
        }
        encode = {
            'user': [],
            'item': [],
        }

        if isinstance(user_ids, int):
            user_ids = [user_ids]
        if isinstance(item_ids, int):
            item_ids = [item_ids]

        for name, ids, encoder in (
            ('user', user_ids, self._user_id_encoder),
            ('item', item_ids, self._item_id_encoder)):
            for i, id_ in enumerate(ids):
                if not encoder.isAbleEncode(id_):
                    miss[name].append(i)
                else:
                    encode[name].append(encoder.encode(id_))

        return {
            'miss': miss,
            'encode': encode,
        }

    @property
    def item_features(self) -> Any:
        return self._item_features

    @property
    def item_id_encoder(self) -> Encoder:
        return self._item_id_encoder

    @property
    def user_id_encoder(self) -> Encoder:
        return self._user_id_encoder
