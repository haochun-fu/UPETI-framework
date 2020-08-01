# -*- coding: utf-8 -*-
"""WrapperBase class."""
import abc
import copy
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


__author__ = 'haochun.fu'
__date__ = '2019-10-18'


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

    def evaluate_recall(
        self,
        data: dict,
        available_products: List[int],
        top_ns: List[int],
        dataloader: Any = None,
    ) -> dict:
        """Evaluate recall.

        Args:
            data (dict): Converted products of each user.
                {
                    USER_ID: [CONTRACT_ID, ...]
                }
            available_products (list): Available products.
            top_ns (list): List of top N.
            dataloader (any): Data loader. Default is None.

        Returns:
            dict: Result.
        """
        total = sum(len(products) for products in data.values())
        result = {
            'elapsed_time': {
                'evaluation': {
                    'second': None,
                    'human_readable': None,
                }
            },
            'evaluation': {
                'total': total,
                'recommendable_amount': 0,
                'top_ns': {
                    # n: {'hit': HIT, 'recall': RECALL}
                    'predict_without_new_user': {},
                    'recommendable': {},
                }
            },
            'amount': {
                'user': len(data),
                'new_user': 0,
            },
            # USER_ID: {
            #   'new_user': TRUE|FALSE,
            #   'recall': {TOP_N: RECALL, ...},
            #   'convert': [[CONTRACT_ID, ORDER], ...],
            #   'recommendation': [CONTRACT_ID, ...],
            # }
            'record': {},
        }

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
        user_ids = []
        for user, buy_items in data.items():
            user_ids.append(user)
            rows.append([user, buy_items, None, None])

        predictions = self.predict(user_ids,
                                   available_products,
                                   dataloader=dataloader)
        for row, prediction in zip(rows, predictions):
            row[2] = prediction

        if self.has_default_prediction():
            predictions = self.predict(user_ids,
                                       available_products,
                                       use_default=True,
                                       dataloader=dataloader)
            for row, prediction in zip(rows, predictions):
                row[3] = prediction

        no_order_val = len(available_products) + 1
        recommendable_amount = 0
        for user, buy_items, predictions, predictions_default in rows:
            new_user = False
            if predictions is None:
                new_user = True
                result['amount']['new_user'] += 1
                if self.has_default_prediction():
                    predictions = predictions_default
                else:
                    continue

            # [(CONTRACT_ID, PREDICTION), ...]
            org_predictions = []
            for contract, prediction in zip(available_products, predictions):
                if prediction is None:
                    continue
                org_predictions.append((contract, prediction))
            org_predictions.sort(key=lambda row: row[1], reverse=True)

            user_top_ns = {n: 0 for n in top_ns}
            convert = {contract: None for contract in buy_items}
            for order, (contract, _) in enumerate(org_predictions, 1):
                if contract not in convert:
                    continue
                convert[contract] = order
                if not new_user:
                    recommendable_amount += 1
                if order > max_top_n:
                    continue
                for n in top_ns:
                    if order <= n:
                        user_top_ns[n] += 1

            convert_amount = len(convert)
            convert = [(id_, order) for id_, order in convert.items()]
            convert.sort(key=lambda row: row[1] if row[1] is not None \
                                         else no_order_val)
            result['record'][user] = {
                'new_user': new_user,
                'recall': {n: amount / convert_amount \
                           for n, amount in user_top_ns.items()},
                'convert': convert,
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

        elapsed_time = result['elapsed_time']['evaluation']
        elapsed_time['second'] = time.time() - start_time
        elapsed_time['human_readable'] = \
            CommFunc.second_to_time(elapsed_time['second'])

        return result

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
        user_id: int,
        item_ids: np.ndarray,
        use_default: bool = False,
        dataloader: Any = None
    ) -> Union[List, None]:
        """Predict.

        Args:
            user_id (int): user_id.
            item_ids (np.int32 array): item ids.
            use_default: Whether use default prediction for new user or not.
                Default is false.
            dataloader (any): Data loader to generate features for prediction.

        Returns:
            list|None: Prediction. If some contract is not able to be predicted,
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

    """Contract ID encoder filename."""
    CONTRACT_ID_ENCODER_FILENAME = "contract_id_encoder.pickle"

    """Item features filename."""
    ITEM_FEATURES_FILENAME = "item_features.pickle"


    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

        self._user_id_encoder = None
        self._contract_id_encoder = None
        self._item_features = None

    def gen_default_prediction(self) -> None:
        """Generate default prediction for new user."""
        start_time = time.time()

        user_ids = self._user_id_encoder.origin
        item_ids = self._contract_id_encoder.origin

        predictions = self.predict(user_ids, item_ids)

        self._default_prediction = np.asarray(predictions, dtype=np.float64) \
                                    .mean(axis=0) \
                                    .tolist()

        sec = time.time() - start_time
        self._model_info['elapsed_time']['generate_default_prediction'] = {
            'second': sec,
            'human_readable': CommFunc.second_to_time(sec)
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

        self._contract_id_encoder = CommFunc.load_pickle(
            os.path.join(model_dir, MatrixWrapper.CONTRACT_ID_ENCODER_FILENAME))

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
            self._contract_id_encoder,
            os.path.join(model_dir, MatrixWrapper.CONTRACT_ID_ENCODER_FILENAME),
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
            ('item', item_ids, self._contract_id_encoder)):
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
    def contract_id_encoder(self) -> Encoder:
        return self._contract_id_encoder

    @property
    def user_id_encoder(self) -> Encoder:
        return self._user_id_encoder

    @property
    def item_features(self) -> Any:
        return self._item_features
