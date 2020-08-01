# -*- coding: utf-8 -*-
"""DateHelper class."""
import datetime
from typing import (
    Generator,
    Tuple,
)


__author__ = 'haochun.fu'
__date__ = '2020-06-02'


DEFAULT_DATE_FORMAT = '%Y-%m-%d'


class DateHelper(object):
    """Date helper functions."""


    @staticmethod
    def modify_days(date: str, days: int, fmt='%Y-%m-%d') -> str:
        """Modify date with days.

        Args:
            date (str): Date in format fmt, e.q., 1970-01-01.
            fmt (str): Date format. Default is '%Y-%m-%d'.

        Returns:
            str: Modified date.

        Examples:
            >>> DateHelper.modify_days('1970-01-01', 1)
            '1970-01-02'

            >>> DateHelper.modify_days('1970-01-02', -1)
            '1970-01-01'
        """
        return (DateHelper.str_to_datetime(date, fmt) + \
                    datetime.timedelta(days=days)) \
                    .strftime(fmt)

    @staticmethod
    def iterate_date_range_split(
        start_date: str,
        end_date: str,
        split: int
    ) -> Generator[Tuple[str, str], None, None]:
        """Generator of iterating split of date range.

        Args:
            start_date (str): Start date in format %Y-%m-%d, e.q., 1970-01-01.
            end_date (str): End date in format %Y-%m-%d, e.q., 1970-01-01.
            split (int): Amount of split.

        Yields:
            tuple: The next date range of split.

        Examples:
            >>> [ \
                range_ for range_ in \
                    DateHelper.iterate_date_range_split( \
                        '1970-01-01', '1970-01-04', 2) \
            ]
            [('1970-01-01', '1970-01-02'), ('1970-01-03', '1970-01-04')]
        """
        fmt = '%Y-%m-%d'
        start_obj = DateHelper.str_to_datetime(start_date, fmt)
        end_obj = DateHelper.str_to_datetime(end_date, fmt)

        split_days = int(((end_obj - start_obj).days + 1) / split)

        if split_days == 0:
            split_days = 1

        tmp_obj = start_obj
        while tmp_obj <= end_obj:
            tmp_date = tmp_obj.strftime(fmt)
            tmp_end_obj = tmp_obj + datetime.timedelta(days=split_days - 1)
            if tmp_end_obj > end_obj:
                yield (tmp_date, end_date)
                break
            yield (tmp_date, tmp_end_obj.strftime(fmt))

            tmp_obj = tmp_end_obj + datetime.timedelta(days=1)

    @staticmethod
    def str_to_datetime(date: str, fmt: str='%Y-%m-%d') -> datetime.datetime: 
        """Convert date string into datetime object.

        Args:
            date (str): Date.
            fmt (str): Date format.

        Returns:
            datetime.datetime: Datetime object.
        """
        return datetime.datetime.strptime(date, fmt)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
