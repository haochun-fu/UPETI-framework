#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate model."""
import argparse
import logging
import time

import argparse_helper
from comm_func import CommFunc
from model_helper import ModelHelper


__date__ = '2020-07-07'
__author__ = 'haochun.fu'


def main(args: argparse.Namespace) -> None:
    """Execution.

    Args:
        args (argparse.Namespace): Arguments.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    start_time = time.time()

    model_info = ModelHelper.load_model(args.model_dir)
    model = model_info['model']

    logging.info('Load data ...')
    params = {
        'data_dir': args.data_dir,
        'param': model_info['config']['data'],
        'test_date': args.test_date,
        **model_info['dataloader_data'],
    }
    dataloader = model_info['DataLoader'](**params)

    logging.info('Test ...')
    if args.item == 'auc':
        params = {
            'data': dataloader.load_test_auc_data(),
        }
    elif args.item == 'recall':
        params = {
            'data': dataloader.load_test_data(),
            'candidates': dataloader.get_candidates(),
            'top_ns': args.top_ns,
            'candidates_filter': dataloader.get_candidates_filter(),
        }
    params['n_processes'] = args.n_processes
    params['dataloader'] = dataloader
    if args.is_pass_data_info:
        params['data_info'] = {
            'data_dir': args.data_dir,
            'test_date': args.test_date,
            'data_config': model_info['config']['data'],
        }
    result = getattr(model, 'evaluate_' + args.item)(**params)

    CommFunc.save_data(result, args.output, save_type='json')
    logging.info(f"Save result to '{args.output}'")

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Evaluate model.')
    arg_parser.add_argument('--item', choices=['auc', 'recall'], required=True)
    arg_parser.add_argument('--test_date',
                            type=argparse_helper.date,
                            required=True,
                            help='Test date.')
    arg_parser.add_argument('--data_dir',
                            type=argparse_helper.dirs,
                            required=True,
                            help='Data directory.')
    arg_parser.add_argument('--is_pass_data_info',
                            default=False,
                            action='store_true',
                            help='Whether pass data information or not.')
    arg_parser.add_argument('--model_dir',
                            type=argparse_helper.dirs,
                            required=True,
                            help='Model directory.')
    arg_parser.add_argument('--top_ns',
                            help="Top Ns, separated by ','.")
    arg_parser.add_argument('--output', required=True, help='Output.')
    arg_parser.add_argument('--n_processes',
                            default=1,
                            type=argparse_helper.positive_integer,
                            help='The number of processes.')
    args = arg_parser.parse_args()

    try:
        if args.item == 'recall':
            argparse_helper.check_miss_item(args, ('top_ns',))

            args.top_ns = args.top_ns.split(',')
            for i, value in enumerate(args.top_ns):
                argparse_helper.positive_integer(value)
                args.top_ns[i] = int(value)
            args.top_ns.sort()

        main(args)
    except argparse_helper.MissingOption as err:
        print(err)
