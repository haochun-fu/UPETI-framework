# -*- coding: utf-8 -*-
"""KerasHelper class."""
from typing import Any

import tensorflow as tf


__date__ = '2020-02-14'
__author__ = 'haochun.fu'


class KerasHelperException(Exception):
        pass


class KerasHelper(object):
    """Helper functions for TensorFlow."""

    @staticmethod
    def generate_optimizer(name: str, params: dict) -> Any:
        """Generate optimizer.
        
        Args:
            name (str): Optimizer name.
            params (dict): Parameters.

        Returns:
            object: Optimizer.

        Raises:
            KerasHelperException: If name is not supported.
        """
        optimizers = tf.keras.optimizers

        if name == 'adam':
            ret = optimizers.Adam(**params)
        else:
            raise TensorFlowHelperException("not support '{name}'")

        return ret
