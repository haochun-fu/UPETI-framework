# -*- coding: utf-8 -*-
"""InteractionDifferenceWrapper and MatrixInteractionDifferenceWrapper classes
which are base class for exntension."""
import abc
import copy
import logging
import os
import time
from typing import (
    Any,
    List,
    Tuple,
    Union,
)

import numpy as np

from comm_func import CommFunc
from wrapper import WrapperBase


__author__ = 'haochun.fu'
__date__ = '2020-02-24'


class InteractionDifferenceWrapper(WrapperBase):
    """Wrapper of interaction difference."""

    """Prefix name of model directory."""
    SPLIT_MODELS_DIR = 'split_models'

    """Split models list file name."""
    SPLIT_MODELS_LIST_FILE_NAME = 'split_models_list.json'


    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

        self._split_models = None
        self._split_models_list = None

    @abc.abstractmethod
    def get_model_wrapper_class(self) -> Any:
        """Get model wrapper class.

        Return:
            Any: Model wrapper class.
        """
        return NotImplemented

    def load(self, model_dir: str) -> None:
        """Load model.

        Args:
            model_dir (str): Model directory.
        """
        super().load(model_dir)

        self._split_models_list = CommFunc.load_json(
            os.path.join(
                model_dir,
                InteractionDifferenceWrapper.SPLIT_MODELS_LIST_FILE_NAME
            ))

        self._split_models = []
        ModelClass = self.get_model_wrapper_class()
        for no in self._split_models_list:
            model = ModelClass()
            model.load(
                os.path.join(
                    model_dir,
                    InteractionDifferenceWrapper.SPLIT_MODELS_DIR,
                    str(no)
                )
            )
            self._split_models.append(model)

    def save(self, model_dir: str) -> None:
        """Save model.

        Args:
            model_dir (str): Model directory.
        """
        super().save(model_dir)

        for no, model in zip(self._split_models_list, self._split_models):
            model.save(
                os.path.join(
                    model_dir,
                    InteractionDifferenceWrapper.SPLIT_MODELS_DIR,
                    str(no)
                )
            )

        CommFunc.save_data(
            self._split_models_list,
            os.path.join(
                model_dir,
                InteractionDifferenceWrapper.SPLIT_MODELS_LIST_FILE_NAME
            ),
            save_type='json'
        )

    @abc.abstractmethod
    def _fit_split_model(self, datas: List[dict], config: dict) -> dict:
        """Train split models.

        Args:
            datas (list): List of training data and data related to training
                data.
            config (dict): Training configuration.

        Returns:
            dict: Information.
        """
        return NotImplemented


class EnsemblerInteractionDifferenceWrapper(InteractionDifferenceWrapper):
    """Wrapper of interaction difference for ensember."""

    def fit(
        self,
        data: List[dict],
        config: dict,
        is_gen_default_prediction: bool = False
    ) -> None:
        """Train.

        Args:
            data (list): List of training data and data related to training
                data.
            config (dict): Training configuration.  is_gen_default_prediction (bool): Whether generate default
                prediction. Default is False.
        """
        info = self._fit_split_model(data, config)

        self._model_info = {
            'elapsed_time': {
                'train_split': info['elapsed_time'],
            },
            'amount': info['amount'],
        }

        if is_gen_default_prediction:
            logging.info('Generate default prediction ...')
            self.gen_default_prediction()

    def gen_default_prediction(self) -> None:
        """Generate default prediction for new user."""
        start_time = time.time()

        user_ids, item_ids = self._get_users_items_from_split_model()

        predictions = np.asarray(
                        self.predict(user_ids, item_ids),
                        dtype=np.float64) \
                        .mean(axis=0)

        self._default_prediction = { 
            id_: value for id_, value in zip(item_ids, predictions)
        }

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

        if self._default_prediction:
            self._default_prediction = {
                int(key): value \
                    for key, value in self._default_prediction.items()
            }

    def predict(
        self,
        user_ids: Union[int, List[int], np.ndarray],
        item_ids: Union[List[int], np.ndarray],
        use_default: bool = False,
        dataloader: Any = None,
        **kwargs,
    ) -> Union[list, None]:
        """Predict.

        Args:
            user_ids (np.int32 array|int|list): User id(s).
            item_ids (np.int32 array|list): Item ids.
            use_default: Whether use default prediction for new user or not.
                Default is false.
            dataloader (any): Data loader to generate features for prediction.

        Returns:
            list|None: Prediction. If some item is not able to be predicted,
                this value is set to None. None if user id provided only one is
                not able to predict.
        """
        is_single_user = False
        if isinstance(user_ids, int):
            is_single_user = True
            user_ids = [user_ids]
        elif isinstance(user_ids, np.ndarray):
            user_ids = user_ids.tolist()
        is_able_use_default = use_default and self.has_default_prediction()

        users_len = len(user_ids)
        items_len = len(item_ids)

        ret_tmp = [[[0, 0] for _ in range(items_len)] for _ in range(users_len)]
        for w, model in enumerate(self._split_models, 1):
            rows = model.predict(
                user_ids,
                item_ids,
                use_default=use_default,
                **kwargs)

            for prediction_u, row_u in zip(ret_tmp, rows):
                if row_u is None:
                    continue
                for prediction, row in zip(prediction_u, row_u):
                    if row is None:
                        continue
                    prediction[0] += w * row
                    prediction[1] += 1

        default = None
        if is_able_use_default:
            default = [self._default_prediction.get(id_) for id_ in item_ids]

        ret = []
        for row_u in ret_tmp:
            row = [row[0] / row[1] if row[1] > 0 else None for row in row_u]
            if len(set(row)) == 1 and row[0] is None:
                row = default
            ret.append(row)

        return ret[0] if is_single_user else ret


class MatrixInteractionDifferenceWrapperHelper(object):
    """Helper functions for matrix interaction difference wrapper."""

    @staticmethod
    def fit_split_model(
        ModelClass: Any,
        datas: List[dict],
        config: dict
    ) -> Tuple[List[Any], dict]:
        """Train split models.

        Args:
            ModelClass (Any): Model class.
            datas (list): List of training data and data related to training
                data.
                [{
                  'interactions': "np.float32 coo_matrix of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     matrix",
                  'item_id_mapping': "[item_id, ...], index of
                                         item_id is id in matrix",
                  'item_features': "np.float64 csr_matrix of shape
                                   [n_items, item_dim]"
                }]
            config (dict): Training configuration.

        Returns:
            Tuple: List of models, and information.
        """
        ret_info = {
            'elapsed_time': {
                'second': None,
                'human_readable': None,
            },
            'amount': {
                'user': set(),
                'item': set(),
                'pair': 0,
            },
        }
        elapsed_time = ret_info['elapsed_time']
        amount = ret_info['amount']

        config = copy.deepcopy(config)
        config['model']['param'] = config['model']['param']['split_model']

        start_time = time.time()

        total = len(datas)
        ret_models = []
        for i, data in enumerate(datas, 1):
            logging.info(f'Train split model {i}/{total} ...')

            amount['user'] |= set(data['user_id_mapping'])
            amount['item'] |= set(data['item_id_mapping'])
            amount['pair'] += data['interactions'].count_nonzero()

            model = ModelClass()
            model.fit(data, config)

            ret_models.append(model)

        elapsed_time['second'] = time.time() - start_time
        elapsed_time['human_readable'] = CommFunc.second_to_time(
            elapsed_time['second'])

        amount['user'] = len(amount['user'])
        amount['item'] = len(amount['item'])

        return ret_models, ret_info

    @staticmethod
    def get_users_items_from_split_model(models) -> Tuple[List[int], List[int]]:
        """Get overall user ids and item ids from split models.

        Args:
            models (list): List of split models.

        Returns:
            tuple: Overall user ids and item ids.
        """
        user_ids = set()
        item_ids = set()
        for model in models:
            user_ids |= set(model.user_id_encoder.origin)
            item_ids |= set(model.item_id_encoder.origin)
        user_ids = sorted(user_ids)
        item_ids = sorted(item_ids)

        return user_ids, item_ids


class MatrixEnsemblerInteractionDifferenceWrapper(
    EnsemblerInteractionDifferenceWrapper):
    """Wrapper of ensembler interaction difference for matrix."""


    def _fit_split_model(self, datas: List[dict], config: dict) -> dict:
        """Train split models.

        Args:
            datas (list): List of training data and data related to training
                data.
                [{
                  'interactions': "np.float32 coo_matrix of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     matrix",
                  'item_id_mapping': "[item_id, ...], index of
                                         item_id is id in matrix",
                  'item_features': "np.float64 csr_matrix of shape
                                   [n_items, item_dim]"
                }]
            config (dict): Training configuration.

        Returns:
            dict: Information.
        """
        self._split_models, ret = \
            MatrixInteractionDifferenceWrapperHelper.fit_split_model(
                self.get_model_wrapper_class(),
                datas,
                config)
        self._split_models_list = list(range(len(self._split_models)))

        return ret

    def _get_users_items_from_split_model(self) -> Tuple[List[int], List[int]]:
        """Get overall user ids and item ids from split model.

        Returns:
            tuple: Overall user ids and item ids.
        """
        return MatrixInteractionDifferenceWrapperHelper \
                .get_users_items_from_split_model(self._split_models)
