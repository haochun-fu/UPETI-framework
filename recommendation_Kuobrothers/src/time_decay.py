# -*- coding: utf-8 -*-
"""Time decay classes.."""
from typing import (
    Any,
    Union,
)


__author__ = 'haochun.fu'
__date__ = '2020-06-26'


class DecayFactory(object):
    """Decay factory."""


    def createByName(self, name: str, param: dict) -> Any:
        """Create linear decay.

        Args:
            name (str): Name.
            param (dict): Parameters.

        Returns:
            Any: Object.

        Raises:
            ValueError: If name is invalid.
        """
        if name == 'linear':
            return LinearDecay(**param)
        else:
            raise ValueError(f"Name '{name}' is invalid")


class LinearDecay(object):
    """Linear decay."""


    def __init__(self, delta: float) -> None:
        """Constructor.

        Args:
            delta (float): Delta.
        """
        self.__delta = delta

    def weight(self, no: int, value: Union[int, float]) -> Union[int, float]:
        """Weight value.

        Args:
            no (int): No.
            value (int|float): Value.

        Returns:
            int|float: Weighted value.
        """
        return no * self.__delta * value
