#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train model."""
import argparse
import importlib
import logging
import os
import shutil
import time

from comm_func import CommFunc
import argparse_helper


__author__ = 'haochun.fu'
__date__ = '2019-10-15'


def main(args: argparse.Namespace) -> None:
    """Execution.

    Args:
        args (argparse.Namespace): Arguments.
    """
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    start_time = time.time()

    logging.info('Load config ...')
    param = CommFunc.load_json(args.config)

    model_param = param['model']
    ModelClass = getattr(importlib.import_module(model_param['module']),
                         model_param['class'])
    model = ModelClass()

    logging.info('Load data ...')
    DataLoader = model.get_dataloader_class()
    params = {
        'data_dir': args.data_dir,
        'param': param['data'],
        'test_date': args.test_date,
    }
    params['is_test' if args.is_train_for_test else 'is_validation'] = True
    dataloader = DataLoader(**params)
    train_data = dataloader.load_train_data()

    logging.info('Train ...')
    fit_args = {
        'data': train_data,
        'config': param,
        'is_gen_default_prediction': args.is_gen_default_prediction,
    }
    if args.ensembler_model_dir:
        fit_args['ensembler_model_dir'] = args.ensembler_model_dir
    model.fit(**fit_args)

    logging.info('Save result ...')
    model.save(args.output_dir)
    logging.info(f"Save model to '{args.output_dir}'")

    rows = [args.test_date]
    file = os.path.join(args.output_dir, 'test_date.json')
    CommFunc.save_data(rows, file, save_type='json')
    logging.info(f"Save test date to '{file}'")

    file = os.path.join(args.output_dir, 'config.json')
    shutil.copyfile(args.config, file)
    logging.info(f"Save config to '{file}'")

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train.')
    arg_parser.add_argument('--test_date',
                            type=argparse_helper.date,
                            required=True,
                            help='Test date.')
    arg_parser.add_argument('--config',
                            type=argparse_helper.files,
                            required=True,
                            help='Configuration file.')
    arg_parser.add_argument('--data_dir',
                            type=argparse_helper.dirs,
                            required=True,
                            help='Data directory.')
    arg_parser.add_argument('--output_dir',
                            required=True,
                            help='Output directory.')
    arg_parser.add_argument('--is_gen_default_prediction',
                            default=False,
                            action='store_true',
                            help='Whether generate default prediction or not.')
    arg_parser.add_argument('--is_train_for_test',
                            default=False,
                            action='store_true',
                            help='Whether train for test or not.')
    arg_parser.add_argument(
        '--ensembler_model_dir',
        type=argparse_helper.dirs,
        help='Ensembler model directory for training interaction difference'
             ' merger model with its split models. If given, it will be used.')
    args = arg_parser.parse_args()

    main(args)
