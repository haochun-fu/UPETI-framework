#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Organize results.."""
import argparse
from collections import defaultdict
import logging
import time

import argparse_helper
from comm_func import CommFunc


__author__ = 'haochun.fu'
__date__ = '2020-07-07'


def organize_evaluation_auc(data_list: list) -> list:
    """Organize evaluation of AUC.

    Args:
        data_list (list): List of data.

    Returns:
        list: Result.
    """
    ret = []
    rec_amount = {
        item: {
            'records': [],
            'average': 0,
        } for item in ('total', 'recommendable_amount')
    }
    auc = {
        'records': [],
        'average': 0,
    }
    user = {
        item: {
            'records': [],
            'average': 0,
        } for item in ('total', 'new')
    }

    for data in data_list:
        evaluation = data['evaluation']

        for item, vals in rec_amount.items():
            vals['records'].append(evaluation[item])
            vals['average'] += evaluation[item]

        auc['records'].append(evaluation['auc'])

        amount = data['amount']
        user['total']['records'].append(amount['user'])
        user['new']['records'].append(amount['new_user'])

    amount = len(data_list)
    for vals in rec_amount.values():
        vals['average'] = sum(vals['records']) / amount
    auc['average'] = sum(auc['records']) / amount
    for vals in user.values():
        vals['average'] = sum(vals['records']) / amount

    ret.append('# Record amount')
    for item in ('total', 'recommendable_amount'):
        vals = rec_amount[item]

        ret.append(f'\n## {item}')
        for rec in vals['records']:
            ret.append(rec)
        ret.append(f"\naverage: {vals['average']}")

    ret.append('\n# Evaluation')
    ret.append('\n## AUC')

    for row in auc['records']:
        ret.append(row)
    ret.append(f"\naverage: {auc['average']}")

    ret.append('\n# user')
    for item_name in ('total', 'new'):
        ret.append(f'\n## {item_name}')
        for rec in user[item_name]['records']:
            ret.append(rec)
        ret.append(f"\naverage: {user[item_name]['average']}")

    return ret


def organize_evaluation_recall(data_list: list) -> list:
    """Organize evaluation of recall.

    Args:
        data_list (list): List of data.

    Returns:
        list: Result.
    """
    ret = []
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
    user_candidates = {
        'records': [],
        'average': 0,
    }

    for data in data_list:
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
        user['new']['records'].append(amount['new_user'])
        user_candidates['records'].append(amount['avg_user_candidates'])

    amount = len(data_list)
    for vals in rec_amount.values():
        vals['average'] = sum(vals['records']) / amount
    for item in items.values():
        average = item['average']
        for top_n in average:
            average[top_n] /= amount
    for vals in user.values():
        vals['average'] = sum(vals['records']) / amount
    user_candidates['average'] = sum(user_candidates['records']) / amount

    ret.append('# Record amount')
    for item in ('total', 'recommendable_amount'):
        vals = rec_amount[item]

        ret.append(f'\n## {item}')
        for rec in vals['records']:
            ret.append(rec)
        ret.append(f"\naverage: {vals['average']}")

    ret.append('\n# Evaluation Items')
    for name, item in ((name, items[name]) for name in sorted(items.keys())):
        ret.append(f'\n## {name}')

        recs = item['records']
        top_ns = sorted(recs[0].keys(), key=lambda val: int(val))
        ret.append(f"Recall@{','.join(val for val in top_ns)}")
        for rec in recs:
            ret.append(','.join(str(rec[n]) for n in top_ns))

        average = item['average']
        ret.append(f"\naverage: {','.join(str(average[n]) for n in top_ns)}")

    ret.append('\n# user')
    for item_name in ('total', 'new'):
        ret.append(f'\n## {item_name}')
        for rec in user[item_name]['records']:
            ret.append(rec)
        ret.append(f"\naverage: {user[item_name]['average']}")

    ret.append('\n# user candidates')
    for rec in user_candidates['records']:
        ret.append(rec)
    ret.append(f"\naverage: {user_candidates['average']}")

    return ret


def organize_model_info(data_list: list) -> list:
    """Organize model information.

    Args:
        data_list (list): List of data.

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
    
    for data in data_list:
        for name, values in data['elapsed_time'].items():
            item = elapsed_time.setdefault(name, {'records': [], 'average': 0})
            item['records'].append(values['human_readable'])
            item['average'] += values['second']

        for name, value in data['amount'].items():
            item = amount.setdefault(name, {'records': [], 'average': 0})
            item['records'].append(value)
            item['average'] += value

    total  = len(data_list)
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


def main(args: argparse.Namespace) -> None:
    """Execution.

    Args:
        args (argparse.Namespace): Arguments..
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    start_time = time.time()

    data_list = [CommFunc.load_json(file) for file in args.data]
    method = globals()['organize_' + args.item]
    result = method(data_list)

    CommFunc.save_data(result, args.output)
    logging.info(f"Save result to '{args.output}'")

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Organize results.',
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        '--data',
        type=argparse_helper.files,
        required=True,
        help="Data json file. Multiple files are separated by ','.")
    arg_parser.add_argument(
        '--item',
        required=True,
        choices=['evaluation_auc', 'evaluation_recall', 'model_info'],
        help='\n'.join([
            'evaluation_auc: AUC evaluation.',
            'evaluation_recall: Recall evaluation.',
            'model_info: Model information.'
        ]))
    arg_parser.add_argument('--output', required=True, help='Output.')
    args = arg_parser.parse_args()

    args.data = args.data.split(',')
    for file in args.data:
        argparse_helper.files(file)

    main(args)
