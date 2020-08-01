# -*- coding: utf-8 -*-
"""DINInteractionDifferenceEnsemblerWrapper class."""
import copy
import logging
import time
from typing import (
    Any,
    List,
    Tuple,
    Union,
)

import numpy as np

from comm_func import CommFunc
from dataloader_din_interaction_difference import \
    DINInteractionDifferenceDataLoader
from din_wrapper import DINWrapper
from interaction_difference_wrapper import EnsemblerInteractionDifferenceWrapper


__author__ = 'haochun.fu'
__date__ = '2020-03-31'


class DINInteractionDifferenceEnsemblerWrapper(
    EnsemblerInteractionDifferenceWrapper):
    """Wrapper of DIN interaction difference ensembler."""

    def evaluate_recall(
        self,
        data: dict,
        available_products: List[int],
        top_ns: List[int],
        dataloader: Any = None,
        data_info: dict = None
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
            data_info (dict): Data information.
                {
                  'data_dir': 'Data directory',
                  'test_date': 'Test date',
                  'data_config': Data configuration,
                }

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
                                   dataloader=dataloader,
                                   data_info=data_info)
        for row, prediction in zip(rows, predictions):
            row[2] = prediction

        if self.has_default_prediction():
            predictions = self.predict(user_ids,
                                       available_products,
                                       use_default=True,
                                       dataloader=dataloader,
                                       data_info=data_info)
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

    def get_dataloader_class(self) -> DINInteractionDifferenceDataLoader:
        """Get dataloader class.

        Returns:
            DINInteractionDifferenceDataLoader: 
                DINInteractionDifferenceDataLoader class.
        """
        return DINInteractionDifferenceDataLoader

    def get_model_wrapper_class(self) -> DINWrapper:
        """Get model wrapper class.

        Return:
            DINWrapper: DIN wrapper class.
        """
        return DINWrapper 

    def predict(
        self,
        user_ids: Union[int, List[int], np.ndarray],
        item_ids: Union[List[int], np.ndarray],
        use_default: bool = False,
        dataloader: Any = None,
        data_info: dict = None
    ) -> Union[list, None]:
        """Predict.

        Args:
            user_ids (np.int32 array|int|list): User id(s).
            item_ids (np.int32 array|list): Item ids.
            use_default: Whether use default prediction for new user or not.
                Default is false.
            dataloader (any): Data loader to generate features for prediction.
            data_info (dict): Data information.
                {
                  'data_dir': 'Data directory',
                  'test_date': 'Test date',
                  'data_config': Data configuration,
                }

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
            params = {
                'data_dir': data_info['data_dir'],
                'param': data_info['data_config'],
                'test_date': data_info['test_date'],
                **model.get_data_for_dataloader(),
            }
            dataloader = model.get_dataloader_class()(**params)

            rows = model.predict(
                user_ids,
                item_ids,
                use_default=use_default,
                dataloader=dataloader)

            for prediction_u, row_u in zip(ret_tmp, rows):
                if row_u is None:
                    continue
                for prediction, row in zip(prediction_u, row_u):
                    if row is None:
                        continue
                    prediction[0] += w * row[0]
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

    def _fit_split_model(self, datas: List[dict], config: dict) -> dict:
        """Train split models.

        Args:
            datas (list): List of training data and data related to training
                data.
                [{
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
                }]

                If it is training for test, 'validation' will not be given.
            config (dict): Training configuration.

        Returns:
            dict: Information.
        """
        ret = {
            'elapsed_time': {
                'second': None,
                'human_readable': None,
            },
            'amount': {
                'user': set(),
                'contract': set(),
                'pair': 0,
            },
        }
        elapsed_time = ret['elapsed_time']
        amount = ret['amount']

        config = copy.deepcopy(config)
        config['model']['param'] = config['model']['param']['split_model']

        start_time = time.time()

        ModelClass = self.get_model_wrapper_class()
        total = len(datas)
        self._split_models = []
        for i, data in enumerate(datas, 1):
            logging.info(f'Train split model {i}/{total} ...')

            x = data['x']

            amount['user'] |= set(x['user'])
            amount['contract'] |= set(x['item'])
            amount['pair'] += len(x['user'])

            model = ModelClass()
            model.fit(data, config)

            self._split_models.append(model)

        elapsed_time['second'] = time.time() - start_time
        elapsed_time['human_readable'] = CommFunc.second_to_time(
            elapsed_time['second'])

        amount['user'] = len(amount['user'])
        amount['contract'] = len(amount['contract'])

        self._split_models_list = list(range(len(self._split_models)))

        return ret

    def _get_users_items_from_split_model(self) -> Tuple[List[int], List[int]]:
        """Get overall user ids and item ids from split model.

        Returns:
            tuple: Overall user ids and item ids.
        """
        user_ids = set()
        item_ids = set()
        for model in models:
            user_ids |= set(model.encoders['user'].classes_)
            item_ids |= set(model.encoders['item'].classes_)
        item_ids.discard(0)
        user_ids = sorted(user_ids)
        item_ids = sorted(item_ids)

        return user_ids, item_ids
