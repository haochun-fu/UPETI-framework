#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Organize results.."""
import argparse
from collections import defaultdict
import logging
import os
import sys
import time

import argparse_helper
from comm_func import CommFunc


__author__ = 'haochun.fu'
__date__ = '2019-10-17'


def organize_evaluation(files: list) -> list:
    """Organize evaluation.

    Args:
        files (list): List of data files.

    Returns:
        list: Result.
    """
    result = []
    rec_amount = {
        item: {
            'records': [],
            'average': 0,
        } for item in ('total', 'recommendable_amount')
    }
    # {
    #   ITEM: {
    #     'records': [
    #       {TOP_N: RECALL}
    #     ],
    #     'average': {TOP_N: RECALL}
    #   }
    # }
    items = {}
    user = {
        item: {
            'records': [],
            'average': 0,
        } for item in ('total', 'new')
    }

    for file in files:
        logging.info(f'Load {file} ...')

        data = CommFunc.load_json(file)

        evaluation = data['evaluation']

        for item, vals in rec_amount.items():
            vals['records'].append(evaluation[item])
            vals['average'] += evaluation[item]

        top_ns = evaluation['top_ns']
        for item_name, item_result in top_ns.items():
            item = items.setdefault(item_name, {})

            item_rec = item.setdefault('records', [])
            item_average = item.setdefault('average', defaultdict(float))

            rec = {}
            for top_n, values in item_result.items():
                rec[top_n] = values['recall']
                item_average[top_n] += values['recall']
            item_rec.append(rec)

        amount = data['amount']
        user['total']['records'].append(amount['user'])
        user['total']['average'] += amount['user']
        user['new']['records'].append(amount['new_user'])
        user['new']['average'] += amount['new_user']

    amount = len(files)
    for vals in rec_amount.values():
        vals['average'] /= amount
    for item in items.values():
        average = item['average']
        for top_n in average:
            average[top_n] /= amount
    for vals in user.values():
        vals['average'] /= amount

    result.append('# Record amount')
    for item in ('total', 'recommendable_amount'):
        vals = rec_amount[item]

        result.append(f'\n## {item}')
        for rec in vals['records']:
            result.append(rec)
        result.append(f"\naverage: {vals['average']}")

    result.append('\n# Evaluation Items')
    for name, item in ((name, items[name]) for name in sorted(items.keys())):
        result.append(f'\n## {name}')

        recs = item['records']
        top_ns = sorted(recs[0].keys(), key=lambda val: int(val))
        result.append(f"Recall@{','.join(val for val in top_ns)}")
        for rec in recs:
            result.append(','.join(str(rec[n]) for n in top_ns))

        average = item['average']
        result.append(f"\naverage: {','.join(str(average[n]) for n in top_ns)}")

    result.append('\n# user')
    for item_name in ('total', 'new'):
        result.append(f'\n## {item_name}')
        for rec in user[item_name]['records']:
            result.append(rec)
        result.append(f"\naverage: {user[item_name]['average']}")

    return result


def organize_model_info(files: list) -> list:
    """Organize model information.

    Args:
        files (list): List of data files.

    Returns:
        list: Result.
    """
    result = []
    # {
    #   ITEM: {
    #     'records': [TIME],
    #     'average': TIME
    #   }
    # }
    elapsed_time = {}
    # {
    #   ITEM: {
    #     'records': [AMOUNT],
    #     'average': AMOUNT 
    #   }
    # }
    amount = {}
    
    for file in files:
        logging.info(f'Load {file} ...')
        data = CommFunc.load_json(file)

        for name, values in data['elapsed_time'].items():
            item = elapsed_time.setdefault(name, {'records': [], 'average': 0})
            item['records'].append(values['human_readable'])
            item['average'] += values['second']

        for name, value in data['amount'].items():
            item = amount.setdefault(name, {'records': [], 'average': 0})
            item['records'].append(value)
            item['average'] += value

    total  = len(files)
    for items in (elapsed_time, amount):
        for item in items.values():
            item['average'] /= total

    for no, (name, items) in enumerate(zip(('Elapsed Time', 'Amount'),
                                           (elapsed_time, amount)),
                                       1):
        if no > 1:
            result.append('')
        result.append(f'# {name}')
        for item_name, item in items.items():
            result.append('')
            result.append(f'## {item_name}')
            for val in item['records']:
                result.append(val)

            result.append('')
            value = item['average']
            if name == 'Elapsed Time':
                value = CommFunc.second_to_time(value)
            result.append(f'average: {value}')

    return result


def main(args: dict) -> None:
    """Execution.

    Args:
        args (dict): Arguments..
    """
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    start_time = time.time()

    data_files = args['data'].split(',')
    if args['item'] == 'evaluation':
        result = organize_evaluation(data_files)
    elif args['item'] == 'model_info':
        result = organize_model_info(data_files)

    CommFunc.save_data(result, args['output'])
    logging.info(f"Save result to '{args['output']}'")

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


def parse_args(args: argparse.Namespace) -> dict:
    """Parse arguments.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        dict: Arguments.
    """
    items = ('data', 'item', 'output')
    options = {item: getattr(args, item) for item in items}
    return options


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Organize results..',
        formatter_class=argparse.RawTextHelpFormatter
        )
    arg_parser.add_argument(
        '--data',
        type=argparse_helper.files,
        required=True,
        help="Data json file. Multiple files are separated by ','.")
    arg_parser.add_argument(
        '--item',
        required=True,
        choices=['evaluation', 'model_info'],
        help='\n'.join([
            'evaluation: Evaluation.',
            'model_info: Model information.'
        ]))
    arg_parser.add_argument('--output', required=True, help='Output.')
    args = arg_parser.parse_args()

    main(parse_args(args))
