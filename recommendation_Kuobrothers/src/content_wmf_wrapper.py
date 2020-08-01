# -*- coding: utf-8 -*-
"""ContentWMFWrapper class."""
import logging
import time
from typing import (
    Any,
    List,
    Union,
)

from content_wmf import content_wmf
import numpy as np

from comm_func import CommFunc
from dataloader_content_wmf import ContentWMFDataLoader
from encoder import Encoder
from wrapper import MatrixWrapper


__author__ = 'haochun.fu'
__date__ = '2019-12-30'


class ContentWMFWrapper(MatrixWrapper):
    """Wrapper of content WMF."""


    def __predict_by_idx(
        self,
        user_idx: int,
        item_idxs: np.ndarray
    ) -> np.ndarray:
        """Predict.

        Args:
            user_idx (int): User index.
            item_idxs (np.int32 array): Item indices.

        Returns:
            np.float32 array: Prediction.
        """
        model = self._model
        return np.dot(model['u'][user_idx], model['v'].T)[item_idxs]

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
                  'interactions': "np.float32 coo_model of shape
                                  [n_users, n_items]).",
                  'user_id_mapping': "[user_id, ...], index of user_id is id in
                                     model",
                  'contract_id_mapping': "[contract_id, ...], index of
                                         contract_id is id in model",
                  'item_features': "np.float64 array of shape
                                   [n_items, item_dim]". If not use, is None.
                }
            config (dict): Training configuration.
            is_gen_default_prediction (bool): Whether generate default
                prediction. Default is False.
        """
        model_param = config['model']['param']['model']

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
                'contract': interactions.shape[1],
                'pair': interactions.count_nonzero(),
            },
        }

        logging.info('Train ...')
        start_time = time.time()

        items = (
            'num_factors',
            'lambda_V_reg',
            'lambda_U_reg',
            'lambda_W_reg',
            'init_std',
            'beta',
            'num_iters',
            'batch_size',
            'random_state',
            'dtype',
            'n_jobs',
        )
        params = {item: model_param[item] for item in items}
        params['S'] = data['interactions']
        params['X'] = data['item_features']
        params['computeW'] = True if params['X'] is not None else False
        params['verbose'] = True
        u, v, w = content_wmf.factorize(**params)

        elapsed_time = model_info['elapsed_time']['train']
        elapsed_time['second'] = time.time() - start_time 
        elapsed_time['human_readable'] = CommFunc.second_to_time(
            elapsed_time['second'])

        self._model = {
            'u': u,
            'v': v,
            'w': w,
        }
        self._model_info = model_info

        self._user_id_encoder = Encoder(data['user_id_mapping'])
        self._contract_id_encoder = Encoder(data['contract_id_mapping'])
        self._item_features = data['item_features']

        if is_gen_default_prediction:
            logging.info('Generate default prediction ...')
            start_time = time.time()

            self.gen_default_prediction()

            sec = time.time() - start_time
            model_info['elapsed_time']['generate_default_prediction'] = {
                'second': sec,
                'human_readable': CommFunc.second_to_time(sec),
            }

    def get_dataloader_class(self) -> ContentWMFDataLoader:
        """Get dataloader class.

        Returns:
            ContentWMFDataLoader: ContentWMFDataLoader class.
        """
        return ContentWMFDataLoader

    def predict(
        self,
        user_ids: Union[int, List[int], np.ndarray],
        item_ids: Union[List[int], np.ndarray],
        use_default: bool = False,
        dataloader: Any = None
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
        is_single_user = isinstance(user_ids, int)
        is_able_use_default = use_default and self.has_default_prediction()
        if isinstance(user_ids, np.ndarray):
            user_ids = user_ids.tolist()
        if isinstance(item_ids, np.ndarray):
            item_ids = item_ids.tolist()

        data = self._organize_user_item_for_predict(user_ids, item_ids)
        miss = data['miss']
        encode = data['encode']
        encode['item'] = np.asarray(encode['item'])

        encode_user_len = len(encode['user'])
        encode_item_len = encode['item'].shape[0]

        users_len = 1 if is_single_user else len(user_ids)
        items_len = len(item_ids)
        if encode_user_len == 0 and not is_able_use_default:
            return None if is_single_user else [None] * users_len
        elif encode_item_len == 0:
            ret = [None] * items_len
            return ret if is_single_user \
                   else [ret.copy() for _ in range(users_len)]

        predictions = []
        if encode_user_len > 0:
            predictions = [
                self.__predict_by_idx(user_id, encode['item']) \
                    for user_id in encode['user'] 
            ]

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
