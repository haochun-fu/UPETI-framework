# -*- coding: utf-8 -*-
"""ModelHelper class."""
import importlib
import os

from comm_func import CommFunc


__authro__ = 'haochun.fu'
__date__ = '2020-04-03'


class ModelHelper(object):
    "Helper functions of model."""

    @staticmethod
    def load_model(model_dir: str, config_name: str = 'config.json') -> dict:
        """Load, model, configuration and DataLoader class.

        Args:
            model_dir (str): Model directory.
            config_name (str): Configuration name. Default is config.json.

        Returns:
            dict: Model, configuration and DataLoader class.
        """
        config = CommFunc.load_json(os.path.join(model_dir, config_name))

        param = config['model']
        ModelClass = getattr(importlib.import_module(param['module']),
                             param['class'])
        model = ModelClass()
        model.load(model_dir)

        DataLoader = model.get_dataloader_class()
        dataloader_data = model.get_data_for_dataloader()

        return {
            'model': model,
            'config': config,
            'DataLoader': DataLoader,
            'dataloader_data': dataloader_data,
        }
