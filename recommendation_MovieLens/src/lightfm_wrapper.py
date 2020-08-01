# -*- coding: utf-8 -*-
"""LightFMWrapper class."""
import itertools
import logging
import time
from typing import (
    Any,
    Dict,
    List,
    Union,
)

from lightfm import LightFM
import numpy as np

from comm_func import CommFunc
from lightfm_dataloader import LightFMDataLoader
from encoder import Encoder
from wrapper import MatrixWrapper


__author__ = 'haochun.fu'
__date__ = '2020-07-06'


class LightFMWrapper(MatrixWrapper):
    """Wrapper of LightFM."""


    def fit(
        self,
        data: dict,
        config: dict,
        is_gen_default_prediction: bool = False
    ) -> None:
        """Train.

        Args:
            data (dict): Training data and data related to training data.
                {
                  'interactions': "np.float32 coo_matrix of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     matrix",
                  'item_id_mapping': "[item_id, ...], index of item_id is id in
                                     matrix",
                  'item_features': "np.float64 csr_matrix of shape
                                   [n_items, item_dim]"
                }
            config (dict): Training configuration.
            is_gen_default_prediction (bool): Whether generate default
                prediction. Default is False.
        """
        param = config['model']['param']
        model_param, fit_param = param['model'], param['fit']

        interactions = data['interactions']

        model_info = {
            'elapsed_time': {
                'train': {
                    'second': None,
                    'human_readable': None,
                },
            },
            'amount': {
                'user': interactions.shape[0],
                'item': interactions.shape[1],
                'pair': interactions.count_nonzero(),
            },
        }

        logging.info('Train ...')
        start_time = time.time()
        items = (
            'no_components',
            'k',
            'n',
            'learning_schedule',
            'loss',
            'learning_rate',
            'rho',
            'epsilon',
            'item_alpha',
            'user_alpha',
            'max_sampled',
            'random_state',
        )
        params = {item: model_param[item] for item in items}
        model = LightFM(**params)

        items = (
            'epochs',
            'num_threads'
        )
        params = {item: fit_param[item] for item in items}
        params['item_features'] = data['item_features']
        params['verbose'] = True
        model.fit(interactions, **params)

        elapsed_time = model_info['elapsed_time']['train']
        elapsed_time['second'] = time.time() - start_time 
        elapsed_time['human_readable'] = \
            CommFunc.second_to_time(elapsed_time['second'])

        self._model = model
        self._model_info = model_info

        self._user_id_encoder = Encoder(data['user_id_mapping'])
        self._item_id_encoder = Encoder(data['item_id_mapping'])
        self._item_features = data['item_features']

        if is_gen_default_prediction:
            logging.info('Generate default prediction ...')
            self.gen_default_prediction()

    def get_dataloader_class(self) -> LightFMDataLoader:
        """Get dataloader class.

        Returns:
            DataLoaderLightFM: DataLoaderLightFM class.
        """
        return LightFMDataLoader

    def predict(
        self,
        user_ids: Union[int, List[int], np.ndarray],
        item_ids: Union[List[int], np.ndarray],
        use_default: bool = False,
        dataloader: Any = None,
        num_threads: int = CommFunc.get_cpu_count()
    ) -> Union[list, None]:
        """Predict.

        Args:
            user_ids (np.int32 array|int|list): User id(s).
            item_ids (np.int32 array|list): Item ids.
            use_default: Whether use default prediction for new user or not.
                Default is false.
            dataloader (any): Data loader to generate features for prediction.
            num_threads (int): The number of threads used to predict. Default
                is the number of CPUs in the system.

        Returns:
            list|None: Prediction. If some item is not able to be predicted,
                this value is set to None. None if user id provided only one is
                not able to predict.
        """
        is_single_user = isinstance(user_ids, int)
        is_able_use_default = use_default and self.has_default_prediction()
        if isinstance(user_ids, np.ndarray):
            user_ids = user_ids.tolist()
        if isinstance(item_ids, np.ndarray):
            item_ids = item_ids.tolist()

        data = self._organize_user_item_for_predict(user_ids, item_ids)
        pair = data['pair']
        miss = data['miss']
        encode = data['encode']

        encode_user_len = len(encode['user'])
        encode_item_len = len(encode['item'])

        users_len = 1 if is_single_user else len(user_ids)
        items_len = len(item_ids)
        if encode_user_len == 0 and not is_able_use_default:
            return None if is_single_user else [None] * users_len
        elif encode_item_len == 0:
            ret = [None] * items_len
            return ret if is_single_user \
                   else [ret.copy() for _ in range(users_len)]

        predictions = []
        if pair['user'].shape[0] > 0:
            predictions_tmp = self._model.predict(
                pair['user'],
                pair['item'],
                item_features=self._item_features,
                num_threads=num_threads)
            for i, value in enumerate(predictions_tmp, 1):
                no = i % encode_item_len

                if encode_item_len == 1:
                    predictions.append([value])
                    continue

                if no == 1:
                    row = []
                row.append(value)

                if no == 0:
                    predictions.append(row)
        
        miss['user'] = set(miss['user'])
        miss['item'] = set(miss['item'])

        default = None
        if is_able_use_default:
            default = []
            item_i = 0
            for i in range(items_len):
                value = None
                if i not in miss['item']:
                    value = self._default_prediction[encode['item'][item_i]]
                    item_i += 1
                default.append(value)

        ret = []
        prediction_i = 0
        for i in range(users_len):
            if i in miss['user']:
                ret.append(default)
                continue

            prediction = predictions[prediction_i]
            item_i = 0
            row = []
            for j in range(items_len):
                value = None
                if j not in miss['item']:
                    value = prediction[item_i]
                    item_i += 1
                row.append(value)
            ret.append(row)
            prediction_i += 1

        return ret[0] if is_single_user else ret

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
                  'pair': {
                    'user': [encoded_user_id],
                    'item': [encoded_item_id],
                  },
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
        ret = super()._organize_user_item_for_predict(user_ids, item_ids)

        encode = ret['encode']

        items_len = len(encode['item'])
        pair = {}
        pair['user'] = np.asarray(
            list(
                itertools.chain(
                    *[[user] * items_len for user in encode['user']])
            )
        )
        pair['item'] = encode['item']  * len(encode['user'])

        ret['pair'] = pair

        return ret
