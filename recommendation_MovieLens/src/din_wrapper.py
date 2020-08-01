# -*- coding: utf-8 -*-
"""DINWrapper class."""
import itertools
import logging
import os
import time
from typing import (
    Dict,
    List,
    Union,
)

from deepctr.models import DIN
from deepctr.layers import custom_objects
import numpy as np
from tensorflow.python.keras.models import (
    load_model,
    save_model,
)

from comm_func import CommFunc
from din_dataloader import DINDataLoader
from keras_helper import KerasHelper
from wrapper import WrapperBase


__author__ = 'haochun.fu'
__date__ = '2020-07-07'


class DINWrapper(WrapperBase):
    """Wrapper of DIN."""


    """Encoders filename."""
    ENCODERS_FILENAME = 'encoders.pickle'

    """Genres filename."""
    GENRES_FILENAME = 'genres.json'

    """Model filename."""
    MODEL_FILENAME = 'model.h5'

    """Training history filename."""
    TRAIN_HISTORY_FILENAME = 'train_history.json'


    def __init__(self):
        """Constructor."""
        super().__init__()

        self.encoders = None
        self.genres = None
        self.__train_history = None

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
            config (dict): Training configuration.
            is_gen_default_prediction (bool): Whether generate default
                prediction. Default is False.
        """
        param = config['model']['param']
        model_param = param['model']
        compile_param = param['compile']
        fit_param = param['fit']

        x = data['x']

        model_info = {
            'elapsed_time': {
                'train': {
                    'second': None,
                    'human_readable': None,
                },
            },
            'amount': {
                'user': len(set(x['user'])),
                'item': len(set(x['item'])),
                'pair': len(x['user']),
            },
        }

        logging.info('Train ...')
        start_time = time.time()

        items = (
            'dnn_use_bn',
            'dnn_hidden_units',
            'dnn_activation',
            'att_hidden_size',
            'att_activation',
            'att_weight_normalization',
            'l2_reg_dnn',
            'l2_reg_embedding',
            'dnn_dropout',
            'seed',
            'task',
        )
        params = {item: model_param[item] for item in items}
        params['dnn_feature_columns'] = data['dnn_feature_columns']
        params['history_feature_list'] = data['history_feature_list']

        model = DIN(**params)

        items = (
            'loss',
            'metrics',
        )
        params = {item: compile_param[item] for item in items}
        optimizer_params = compile_param['optimizer']
        params['optimizer'] = KerasHelper.generate_optimizer(
            optimizer_params['name'], optimizer_params['param'])
        model.compile(**params)

        items = (
            'batch_size',
            'epochs',
            'verbose',
        )
        params = {item: fit_param[item] for item in items}
        params['x'] = x
        params['y'] = data['y']
        if 'validation' in data:
            validation_data = data['validation']
            params['validation_data'] = (
                validation_data['x'],
                validation_data['y'],
            )
        self.__train_history = model.fit(**params).history
        for values in self.__train_history.values():
            for i, value in enumerate(values):
                values[i] = float(value)

        elapsed_time = model_info['elapsed_time']['train']
        elapsed_time['second'] = time.time() - start_time 
        elapsed_time['human_readable'] = CommFunc.second_to_time(
            elapsed_time['second'])

        self._model = model
        self._model_info = model_info
        self.encoders = data['encoders']
        self.genres = data['genres']

        if is_gen_default_prediction:
            logging.info('Generate default prediction ...')
            start_time = time.time()

            self.gen_default_prediction(data['dataloader'])

            sec = time.time() - start_time
            model_info['elapsed_time']['generate_default_prediction'] = {
                'second': sec,
                'human_readable': CommFunc.second_to_time(sec),
            }

    def gen_default_prediction(self, dataloader: DINDataLoader)-> None:
        """Generate default prediction for new user.

        Args:
            dataloader (DINDataLoader): Data loader.
        """
        start_time = time.time()

        users = self.encoders['user'].classes_
        items = self.encoders['item'].classes_.tolist()
        if 0 in items:
            items.remove(0)

        predictions = self.predict(users, items, dataloader=dataloader)

        self._default_prediction = np.asarray(predictions, dtype=np.float64) \
                                    .mean(axis=0) \
                                    .tolist()

        sec = time.time() - start_time
        self._model_info['elapsed_time']['generate_default_prediction'] = {
            'second': sec,
            'human_readable': CommFunc.second_to_time(sec),
        }

    def get_data_for_dataloader(self) -> dict:
        """Get data for dataloader.

        Returns:
            dict: Data for dataloader.
        """
        return {
            'encoders': self.encoders,
            'genres': self.genres,
        }

    def get_dataloader_class(self) -> DINDataLoader:
        """Get dataloader class.

        Returns:
            DINDataLoader: DINDataLoader class.
        """
        return DINDataLoader

    def load(self, model_dir: str) -> None:
        """Load model.

        Args:
            model_dir (str): Model directory.
        """
        super().load(model_dir)

        self._model = load_model(
            os.path.join(model_dir, DINWrapper.MODEL_FILENAME), custom_objects)

        self.encoders = CommFunc.load_pickle(
            os.path.join(model_dir, DINWrapper.ENCODERS_FILENAME))

        self.genres = CommFunc.load_json(
            os.path.join(model_dir, DINWrapper.GENRES_FILENAME))

    def predict(
        self,
        user_ids: Union[int, List[int], np.ndarray],
        item_ids: Union[List[int], np.ndarray],
        use_default: bool = False,
        dataloader: DINDataLoader = None
    ) -> Union[List, None]:
        """Predict.

        Args:
            user_ids (np.int32 array|int|list): User id(s).
            item_ids (np.int32 array|list): Item ids.
            use_default: Whether use default prediction for new user or not.
                Default is false.
            dataloader (DINDataLoader): Data loader to generate features for prediction.

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

        data = self._organize_user_item_for_predict(user_ids, item_ids, dataloader)
        x = data['x']
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
        if encode_user_len != 0 and encode_item_len != 0:
            predictions_tmp = self._model.predict(x)
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

    def save(self, model_dir: str) -> None:
        """Save model.

        Args:
            model_dir (str): Model directory.
        """
        super().save(model_dir)

        save_model(
            self._model, os.path.join(model_dir, DINWrapper.MODEL_FILENAME))

        CommFunc.save_data(
            self.encoders,
            os.path.join(model_dir, DINWrapper.ENCODERS_FILENAME),
            save_type='pickle')

        CommFunc.save_data(
            self.genres,
            os.path.join(model_dir, DINWrapper.GENRES_FILENAME),
            save_type='json')

        CommFunc.save_data(
            self.__train_history,
            os.path.join(model_dir, DINWrapper.TRAIN_HISTORY_FILENAME),
            save_type='json')

    def _organize_user_item_for_predict(
        self,
        user_ids: Union[int, List[int]],
        item_ids: Union[int, List[int]],
        dataloader: DINDataLoader
    ) -> Dict[str, dict]:
        """Organize user ids and item ids for prediction.

        Args:
            user_ids (int|list): User id(s).
            item_ids (int|list): Item id(s).
            dataloader (DINDataLoader): Data loader.

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
                  'x': {
                    Generated by data loader
                  }
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


        ok = {
            'user': [],
            'item': [],
        }
        for name, ids, enc_ids in (
            ('user', user_ids, self.encoders['user'].classes_),
            ('item', item_ids, self.encoders['item'].classes_)):
            enc_ids_map = {id_: i for i, id_ in enumerate(enc_ids)}
            for i, id_ in enumerate(ids):
                if id_ not in enc_ids_map:
                    miss[name].append(i)
                else:
                    encode[name].append(enc_ids_map[id_])
                    ok[name].append(id_)
            
        rows = itertools.product(ok['user'], ok['item'])
        x, miss_idxs = dataloader.to_x(rows)

        if miss_idxs:
            raise ValueError('There are missing features.')

        return {
            'miss': miss,
            'encode': encode,
            'x': x,
        }
