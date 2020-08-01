# -*- coding: utf-8 -*-
"""CommFunc class."""
import datetime
import json
import multiprocessing
import os
import pickle
import re
from typing import (
    Generator,
    List,
    Tuple,
    Union,
)


__author__ = 'haochun.fu'
__date__ = '2018-10-15'


class CommFunc(object):
    """Common functions."""


    @staticmethod
    def get_cpu_count() -> int:
        """Get the number of CPUs in the system.

        Returns:
            int: The number of CPUs in the system.
        """
        return multiprocessing.cpu_count()

    @staticmethod
    def save_data(
        data: Union[list, dict, object],
        file: str,
        save_type: str = 'text',
        mode: str = 'w'
    ) -> None:
        """Save data to file.

        Args:
            data (list|dict|object): Data.
            file (str): File.
            save_type (str): Save type, text, json, pickle. default is text.
            mode (str): Mode of saving, w or wb. Default is 'w'. If save_type
                is pickle, mode is set to wb.
        """
        CommFunc.create_dir(file, is_dir=False)

        if save_type == 'pickle':
            mode = 'wb'

        with open(file, mode) as f:
            if save_type == 'json':
                print(json.dumps(data), file=f)
            elif save_type == 'pickle':
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                for row in data:
                    print(row, file=f)

    @staticmethod
    def current_time(fmt: str = '%Y-%m-%d %H-%M-%S') -> str:
        """Current time.

        Args:
            fmt (str): Format.

        Returns:
            str: Current time.
        """
        return datetime.datetime.now().strftime(fmt)

    @staticmethod
    def second_to_time(second: Union[float, int]) -> str:
        """Convert second into time description.

        Args:
            second (float|int): Second.

        Returns:
            str: Time description of second.
        """
        units = ('s', 'm', 'h', 'd')
        items = (60, 60, 24) 

        values = []
        tmp = second
        for i, item in enumerate(items):
            value = tmp % item
            values.append(int(value) if float(value).is_integer() else value)
            tmp //= item

            if tmp == 0:
                break
        if tmp != 0:
            values.append(int(tmp))

        result = []
        for i, value in enumerate(values):
            result.append(f'{value}{units[i]}')
        result = ' '.join(result[::-1])

        return result

    @staticmethod
    def load_json(file: str) -> Union[list, dict]:
        """Load json data from file.

        Args:
            file (str): Data file path.

        Returns:
            (list|dict): Json data.
        """
        with open(file, 'r') as f:
            return json.load(f)

    @staticmethod
    def create_dir(path: str, is_dir: bool = True) -> None:
        """Create directory of path.

        Args:
            path (str): Path.
            is_dir (bool): Whether path is directory path or not.
                Default is True.
        """
        # Convert path into canonical path for letting os.makedirs not be
        # confused of case having '..'.
        path = os.path.realpath(path)

        if not is_dir:
            path = os.path.dirname(path)

        if path and not os.path.isdir(path):
            os.makedirs(path)

    @staticmethod
    def remove_microsecond_padding_zero(data: str) -> str:
        """Remove padding zero of microsecond.

        Args:
            data (str): Datetime.

        Returns:
            str: Result.
        """
        result = data.rstrip('0')
        if result[-1] == '.':
            result = result[:-1]

        return result

    @staticmethod
    def get_start_end_of_date_range(
        date_range: Tuple[str, str]
    ) -> Tuple[str, str]:
        """Get start date and end date of date range.
    
        Args:
            date_range (tuple): Date range, 0: start date, 1: end date. "-x" in
                start day means that the date of amount of days before end date.
    
        Returns:
            tuple: Start date and end date.
        """
        start = date_range[0]
        end = date_range[1]
    
        days_before = re.match('-(\d+)', start)
        if days_before is not None and end != '':
            end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
            date_before_end = end_date - datetime.timedelta(days=int(days_before.groups()[0]))
            start = date_before_end.strftime('%Y-%m-%d')
    
        return start, end

    @staticmethod
    def iterate_date_range(
        start_date: str,
        end_date: str
    ) -> Generator[str, None, None]:
        """Generator of iterating date range.

        Args:
            start_date (str): Start date e.q. 1970-01-01.
            end_date (str): End date e.q. 1970-01-01.

        Yields:
            str: The next date.

        Examples:
            >>> [for date in iterate_date_range('1970-01-01', '1970-01-03')]
            ['1970-01-01', '1970-01-02', '1970-01-03']
        """
        date_obj = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        while date_obj <= end_date_obj:
            yield date_obj.strftime('%Y-%m-%d')

            date_obj += datetime.timedelta(days=1)

    @staticmethod
    def str_to_datetime(datetime_str: str) -> datetime.datetime:
        """Convert datetime string into datetime object.

        Args:
            datetime_str (str): Date string in format, '%Y-%m-%d %H:%M:%S' or
                '%Y-%m-%d %H:%M:%S.%f'.

        Returns:
            datetime.datetime: Datetime object.
        """
        fmt = '%Y-%m-%d %H:%M:%S'
        if datetime_str.find('.') != -1:
            fmt += '.%f'

        result = datetime.datetime.strptime(datetime_str, fmt)

        return result

    @staticmethod
    def amount_in_range(
        times: List[str],
        start: str,
        end: str,
        is_include_start: bool = True,
        is_include_end: bool = True
    ) -> int:
        """Count amount in range in list of time.
    
        Args:
            times (list): List of time.
            start (str): Start time.
            end (str): End time.
            is_include_start (bool): Whether include start or not. Default is
                True.
            is_include_end (bool): Whether include end or not. Default is True.
    
        Returns:
            int: Amount.
        """
        counter = 0
        for time in times:
            if start != ''\
               and (time < start or (not is_include_start and time <= start)):
                continue
        
            if end != ''\
               and (time > end or (not is_include_end and time >= end)):
                break

            counter += 1
        
        return counter

    @staticmethod
    def load_pickle(file: str) -> object:
        """Load pickle from file.

        Args:
            file (str): File.

        Returns:
            object: Data.
        """
        with open(file, 'rb') as f:
            return pickle.load(f)
