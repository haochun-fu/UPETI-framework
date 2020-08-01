# -*- coding: utf-8 -*-
"""Pandas helper."""
from typing import Generator

import pandas as pd

def iterate_with_dict(
    data: pd.core.frame.DataFrame
) -> Generator[dict, None, None]:
    """Generator of accessing each row in data.

    Args:
        data (pandas.core.frame.DataFrame): Data.

    Yields:
        dict: The next row.

    Examples:
        >>> [for row in iterate_with_dict(data)]
        [{'column1': value1, 'column2': value2, ...}, ...]
    """
    columns = data.columns

    for row in data.values:
        yield {column: row[i] for i, column in enumerate(columns)}
