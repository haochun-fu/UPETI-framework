# -*- coding: utf-8 -*-
"""argparse helper."""
import argparse
import datetime
import os


__author__ = 'haochun.fu'
__date__ = '2019-09-23'


class MissingOption(Exception):
    pass


def check_miss_item(args: argparse.Namespace, items: tuple) -> None:
    """Check missing items.

    Args:
        args (argparse.Namespace): Arguments.
        items (tuple): Items.

    Raises:
        MissingOption: If item is missing.
    """
    for item in items:
        if getattr(args, item) is None:
            raise MissingOption(f"requires '{', '.join(items)}'")


def positive_float(value: str) -> float:
    """Positive flaot.

    Args:
        value (str): Value.

    Returns:
        float: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    result = None

    try:
        result = float(value)
        if result <= 0:
            raise
    except:
        raise argparse.ArgumentTypeError('shoud be float number > 0.')

    return result


def positive_integer(value: str) -> int:
    """Positive integer.

    Args:
        value (str): Value.

    Returns:
        int: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    result = None

    try:
        result = int(value)
        if result <= 0:
            raise
    except:
        raise argparse.ArgumentTypeError('shoud be integer > 0.')

    return result


def nonnegative_integer(value: str) -> int:
    """Nonnegative integer.

    Args:
        value (str): Value.

    Returns:
        int: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    result = None

    try:
        result = int(value)
        if result < 0:
            raise
    except:
        raise argparse.ArgumentTypeError('shoud be integer >= 0.')

    return result


def year_month(value: str) -> str:
    """Year and month is in format YYYY-MM.

    For example, 1970-01.

    Args:
        value (str): Value.

    Returns:
        str: value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    try:
        datetime.datetime.strptime(value, '%Y-%m')
    except:
        raise argparse.ArgumentTypeError('should be in the format YYYY-MM, like'
                                         '1970-01.')

    return value


def date(value: str) -> str:
    """Date is in format YYYY-MM-DD.
    
    For example, 1970-01-01.

    Args:
        value (str): Value.

    Returns:
        str: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    try:
        datetime.datetime.strptime(value, '%Y-%m-%d')
    except:
        raise argparse.ArgumentTypeError('should be in the format YYYY-MM-DD,'
                                         'like 1970-01-01.')

    return value


def non_separator_date(value: str) -> str:
    """Date is in format YYYYMMDD.
    
    For example, 19700101.

    Args:
        value (str): Value.

    Returns:
        str: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    try:
        datetime.datetime.strptime(value, '%Y%m%d')
    except:
        raise argparse.ArgumentTypeError('should be in the format YYYYMMDD,'
                                         'like 19700101.')

    return value


def date_time(value: str) -> str:
    """Date and time is in format of Y-m-d H:M:S.
    
    For example, 1970-01-01 00:00:00.

    Args:
        value (str): Value.

    Returns:
        str: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    try:
        datetime.datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    except:
        raise argparse.ArgumentTypeError('should be in the format'
                                         ' YYYY-MM-DD hh:mm:ss, like 1970-01-01'
                                         ' 00:00:00.')

    return value


def files(value: str) -> str:
    """Check whether file(s) is exist or not.

    Multiple files are separated by ','.
    
    Args:
        value (str): Files are separated by ','.

    Returns:
        str: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    files = value.split(',')
    for file in files:
        if not os.path.isfile(file):
            raise argparse.ArgumentTypeError(f"'{file}' is not found.")

    return value


def dirs(value: str) -> str:
    """Check whether directory(s) is exist or not.

    Args:
        value (str): Value.

    Returns:
        str: Value.

    Raises:
        argparse.ArgumentTypeError: If value is invalid.
    """
    dirs = value.split(',')
    for dir_ in dirs:
        if not os.path.isdir(dir_):
            raise argparse.ArgumentTypeError(f"'{dir_}' is not found.")

    return value
