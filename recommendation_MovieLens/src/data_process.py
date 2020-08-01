#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data process."""
import argparse
import logging
import os
import random
import time

import pandas as pd

import argparse_helper
from comm_func import CommFunc
from date_helper import DateHelper
from movielens_helper import MovielensHelper
import pandas_helper


__author__ = 'haochun.fu'
__date__ = '2020-07-03'


def process_extract_auc_item_for_test_date(
    data: pd.core.frame.DataFrame,
    test_date_data: pd.core.frame.DataFrame
) -> dict:
    """Extract AUC item before test date.

    Positive: rating >= 4
    Negative: rating <= 2

    Positive and negaive items will be shuffled.

    Args:
        data (pandas.core.frame.DataFrame): Data.
        test_date_data (pandas.core.frame.DataFrame): Test date data.

    Returns:
        dict: Result.
            {
              USER_ID: {
                'pos': [ITEM_ID, ...],
                'neg': [ITEM_ID, ...],
              },
            }
    """
    def __extract_user(user, test_date):
        ret = {
            'neg': set(),
            'pos': set(),
        }

        items = (
            ('neg', __is_neg),
            ('pos', __is_pos),
        )
        for row in \
            pandas_helper.iterate_with_dict(data[data['userId'] == user]):
            if test_date <= \
                DateHelper.ts_to_datetime(row['timestamp']) \
                    .strftime('%Y-%m-%d'):
                continue

            for name, method in items:
                if method(row['rating']):
                    ret[name].add(int(row['movieId']))
                    break

        for key in ret.keys():
            ret[key] = list(ret[key])
            random.shuffle(ret[key])

        return ret

    def __is_neg(rating):
        return rating <= 2

    def __is_pos(rating):
        return rating >= 4

    ret = {}

    date = test_date_data.iloc[0]['timestamp'].split(' ')[0]

    logging.info('Organize data ...')
    for user in test_date_data['userId'].unique():
        ret[user.item()] = __extract_user(user, date)

    return ret


def process_extract_rating_date(
    data: pd.core.frame.DataFrame,
    output_dir: str,
    year: int = None
) -> None:
    """Extract ratings of each date and save them into output directory.

    Args:
        data (pandas.core.frame.DataFrame): Data.
        output_dir (str): Output directory.
        year (int): Year to be extracted. None is for all records. Default is
            None.
    """
    org = {}

    logging.info('Organize data ...')
    for row in pandas_helper.iterate_with_dict(data):
        ts = DateHelper.ts_to_datetime(row['timestamp'])

        if year is not None and year != int(ts.strftime('%Y')):
            continue

        row['timestamp'] = ts.strftime('%Y-%m-%d %H:%M:%S')

        org.setdefault(ts.strftime('%Y-%m-%d'), []).append(row)

    for rows in org.values():
        rows.sort(key=lambda row: row['timestamp'])

    CommFunc.create_dir(output_dir)
    cols = data.columns
    for date, rows in org.items():
        file = os.path.join(output_dir, date + '.csv')
        pd.DataFrame(rows, columns=cols). \
            to_csv(file, sep=',', encoding='utf-8', index=False)
        logging.info(f'Save {file}')


def process_extract_genome_tag(
    data: pd.core.frame.DataFrame,
    relevance_lower_bound: float = None
) -> dict:
    """Extract genome tag of movie.

    Args:
        data (pandas.core.frame.DataFrame): Data.
        relevance_lower_bound (float): Lower bound of relevance score. None for
            all. Default is None.

    Returns:
        dict: Result.
    """
    ret = {}

    logging.info('Organize data ...')
    for row in pandas_helper.iterate_with_dict(data):
        if relevance_lower_bound is not None and \
            row['relevance'] < relevance_lower_bound:
            continue

        ret.setdefault(int(row['movieId']), []). \
            append({'tagId': int(row['tagId']), 'relevance': row['relevance']})

    for rows in ret.values():
        rows.sort(key=lambda row: row['relevance'], reverse=True)

    return ret


def process_extract_genres(
    data: pd.core.frame.DataFrame,
    relevance_lower_bound: float = None
) -> dict:
    """Extract geners of movie.

    Args:
        data (pandas.core.frame.DataFrame): Data.

    Returns:
        dict: Result.
    """
    ret = {}

    logging.info('Organize data ...')
    for row in pandas_helper.iterate_with_dict(data):
        ret[row['movieId']] = [] if row['genres'] == '(no genres listed)' \
            else row['genres'].split('|')

    return ret


def process_extract_seen_movie_for_test_date(
    data: pd.core.frame.DataFrame,
    test_date_data: pd.core.frame.DataFrame
) -> dict:
    """Extract movies user having seen before test date.

    Args:
        data (pandas.core.frame.DataFrame): Data.
        test_date_data (pandas.core.frame.DataFrame): Test date data.

    Returns:
        dict: Result.
    """
    def __extract_user(user, test_date):
        ret = []

        for row in \
            pandas_helper.iterate_with_dict(data[data['userId'] == user]):
            if test_date <= \
                DateHelper.ts_to_datetime(row['timestamp']) \
                    .strftime('%Y-%m-%d'):
                continue

            ret.append(int(row['movieId']))

        ret.sort()

        return ret

    ret = {}

    date = test_date_data.iloc[0]['timestamp'].split(' ')[0]

    logging.info('Organize data ...')
    for user in test_date_data['userId'].unique():
        ret[user.item()] = __extract_user(user, date)

    return ret


def main(args: argparse.Namespace) -> None:
    """Execution.

    Args:
        args (argparse.Namespace): Arguments.
    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S')
    start_time = time.time()

    logging.info('Load data ...')
    args.data = pd.read_csv(args.data)
    if args.item in ('extract_auc_item_for_test_date',
        'extract_rating_date',
        'extract_seen_movie_for_test_date'):
        args.data = MovielensHelper.convert_ratings_column_type(args.data)

    if args.item in (
        'extract_auc_item_for_test_date',
        'extract_seen_movie_for_test_date'):
        args.test_date_data = MovielensHelper.convert_ratings_column_type(
            pd.read_csv(args.test_date_data))

    logging.info(f"Execute {args.item} ...")
    method = globals()['process_' + args.item]
    items_params = {
        'extract_auc_item_for_test_date': ('data', 'test_date_data'),
        'extract_rating_date': ('data', 'output_dir', 'year'),
        'extract_genome_tag': ('data', 'relevance_lower_bound'),
        'extract_genres': ('data',),
        'extract_seen_movie_for_test_date': ('data', 'test_date_data'),
    }
    res = method(
        **{name: getattr(args, name) for name in items_params[args.item]})

    if args.item in (
        'extract_auc_item_for_test_date',
        'extract_genome_tag',
        'extract_genres',
        'extract_seen_movie_for_test_date'):
        CommFunc.create_dir(args.output, is_dir=False)
        CommFunc.save_data(res, args.output, save_type='json')
        logging.info(f'Save to {args.output}')

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
            description='Data process.',
            formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        '--item',
        choices=[
            'extract_auc_item_for_test_date',
            'extract_rating_date',
            'extract_genome_tag',
            'extract_genres',
            'extract_seen_movie_for_test_date',
        ],
        help='\n'.join([
            'extract_auc_item_for_test_date: Extract AUC items for test date.',
            'extract_rating_date: Extract ratings of each date.',
            'extract_genome_tag: Extract genome tag of movie.',
            'extract_genres: Extract genres of movie.',
            'extract_seen_movie_for_test_date: Extract movie user having seed'
                ' for test date.',
        ]),
        required=True
    )
    arg_parser.add_argument(
        '--data',
        type=argparse_helper.files,
        help='\n'.join([
            'Data CSV file.',
            '  extract_auc_item_for_test_date: e.q., data/ml-25m/ratings.csv',
            '  extract_rating_date: e.q., data/ml-25m/ratings.csv',
            '  extract_genome_tag: e.q., data/ml-25m/genome-scores.csv',
            '  extract_genres: e.q., data/ml-25m/movies.csv',
            '  extract_seen_movie_for_test_date: e.q., data/ml-25m/ratings.csv',
        ])
    )
    arg_parser.add_argument(
        '--year',
        type=argparse_helper.positive_integer,
        help='Year to be extracted for item extract_rating_date.'
    )
    arg_parser.add_argument(
        '--relevance_lower_bound',
        type=argparse_helper.positive_float,
        help='Lower bound of relevance score for item extract_genome_tag.'
    )
    arg_parser.add_argument(
        '--test_date_data',
        type=argparse_helper.files,
        help='Test date data for item extract_auc_item_for_test_date,'
             ' extract_seen_movie_for_test_date, e.q.,'
             ' training_data/features/2019-11-20.csv.'
    )
    arg_parser.add_argument('--output', help='Output.')
    arg_parser.add_argument('--output_dir', help='Output directory.')
    args = arg_parser.parse_args()

    try:
        if args.item in (
            'extract_auc_item_for_test_date',
            'extract_seen_movie_for_test_date'):
            argparse_helper.check_miss_item(
                args, ('data', 'output', 'test_date_data'))
        elif args.item == 'extract_rating_date':
            argparse_helper.check_miss_item(args, ('data', 'output_dir',))
        elif args.item in ('extract_genome_tag', 'extract_genres'):
            argparse_helper.check_miss_item(args, ('data', 'output',))

        main(args)
    except argparse_helper.MissingOption as err:
        print(err)
