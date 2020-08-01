# -*- coding: utf-8 -*-
"""MovielensHelper class."""
import pandas as pd


__author__ = 'haochun.fu'
__date__ = '2020-07-09'


class MovielensHelper(object):
    """Helper functions for dataset movielens."""


    @staticmethod
    def convert_ratings_column_type(
        data: pd.core.frame.DataFrame
    ) -> pd.core.frame.DataFrame:
        """Convert column types of ratings data.

        Args:
            data (pandas.core.frame.DataFrame): Data.
        """
        return data.astype({'userId': int, 'movieId': int})
