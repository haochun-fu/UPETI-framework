# -*- coding: utf-8 -*-
"""Encoder class.

For example, transform ID into ID in matrix.
"""
import copy
from typing import Union


__author__ = 'haochun.fu'
__date__ = '2019-10-08'


class Encoder(object):
    """Encoder.
    
    Transform original ID into converted ID.
    Transform converted ID into original ID.
    """

    def __init__(self, data: list) -> None:
        """Constructor.

        Args:
            data (list): Element is original ID. Index of element is converted
                         ID.
        """
        self.__origin = copy.copy(data)
        self.__max_converted_idx = len(self.__origin) - 1
        self.__mapping = self.__gen_mapping(self.__origin)

    def decode(self, index: int) -> Union[float, int, str]:
        """Transform converted index into original value.

        Args:
            index (int): Index.

        Returns:
            float|int|str: Original value.

        Raises:
            LookupError: If index is not found.
        """
        if index > self.__max_converted_idx:
            raise LookupError(f"'{index}' is not found.")
        return self.__origin[index]

    def encode(self, value: Union[float, int, str]) -> Union[float, int, str]:
        """Transform value into converted index.

        Args:
            value (float|int|str): Value.

        Returns:
            float|int|str: Converted index.

        Raises:
            LookupError: If value is not found.
        """
        result = self.__mapping.get(value, None)
        if result is None:
            raise LookupError(f"'{value}' is not found.")
        return result

    def isAbleDecode(self, index: int) -> bool:
        """Whether index is able to be decoded or not.

        Args:
            index (int): Index.

        Returns:
            bool: Whether index is able to be decoded or not.
        """
        return index <= self.__max_converted_idx

    def isAbleEncode(self, value: Union[float, int, str]) -> bool:
        """Whether value is able to be encoded or not.

        Args:
            value (float|int|str): Value.

        Returns:
            bool: Whether value is able to be encoded or not.
        """
        return value in self.__mapping

    def __gen_mapping(self, data: list) -> dict:
        """Generate mapping of original ID to converted ID.
        
        Args:
            data (list): Element is original ID. Index of element is converted
                ID.
        """
        return {val: idx for idx, val in enumerate(data)}

    @property
    def mapping(self) -> dict:
        return self.__mapping

    @property
    def origin(self) -> list:
        return self.__origin
