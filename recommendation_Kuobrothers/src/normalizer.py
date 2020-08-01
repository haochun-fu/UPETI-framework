#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalizer class and execution."""
import argparse
import logging
import os
import time
from typing import (
    Tuple,
    Callable,
)

import numpy as np

import argparse_helper
from comm_func import CommFunc

__author__ = 'haochun.fu'
__date__ = '2019-12-27'


class Normalizer(object):
    """Normalizer."""

    def mean(self, data: dict) -> Tuple[dict, dict]:
        """Mean normalization.
    
        Args:
            data (dict): Data.
    
        Returns:
            tuple: Result and information.
        """
        info = {
            'mean': data.mean(axis=0),
            'min': data.min(axis=0),
            'max': data.max(axis=0),
        }

        # x' = (x - mean) / (max - min)
        result = (data - info['mean']) / (info['max'] - info['min'])
    
        return result, info

    def min_max(self, data: np.ndarray) -> Tuple[dict, dict]:
        """Min-max normalization.
    
        Args:
            data (numpy.ndarray): Data.
    
        Returns:
            tuple: Result and information.
        """
        info = {
            'min': data.min(axis=0),
            'max': data.max(axis=0),
        }
    
        # x' = (x - min) / (max - min)
        result = (data - info['min']) / (info['max'] - info['min'])
    
        return result, info
    
    def standardization(self, data: dict) -> Tuple[dict, dict]:
        """Standardization normalization.
    
        Args:
            data (dict): Data.
    
        Returns:
            tuple: Result and information.
        """
        info = {
            'mean': data.mean(axis=0),
            'std': data.std(axis=0),
        }
    
        # x' = (x - mean) / std
        result = (data - info['mean']) / info['std']
    
        return result, info


def _normalize(
    data: dict,
    process: Callable[[np.ndarray], Tuple[np.ndarray, dict]]
) -> Tuple[dict, dict]:
    """Normalize.

    Args:
        data (dict): Data.
        process (class 'method'): Process method.

    Returns:
        tuple: Result and information.
    """
    contract_ids = list(data)
    
    result = {id_: {} for id_ in contract_ids}
    info = {}

    for item in ('name', 'desc_short'):
        idx_to_id = []
        vectors = []
        for id_ in contract_ids:
            vector = data[id_][item]
            if vector is None:
                continue
            vectors.append(vector)
            idx_to_id.append(id_)
        idx_to_id_set = set(idx_to_id)
        vectors = np.asarray(vectors, dtype=np.float64)
    
        vectors, info[item] = process(vectors)
    
        for id_ in contract_ids:
            result[id_][item] = vectors[idx_to_id.index(id_)].tolist()\
                if id_ in idx_to_id_set else None

    for info_ in info.values():
        for key, val in info_.items():
            info_[key] = val.tolist()
    
    return result, info


def main(args: dict) -> None:
    """Execution.

    Args:
        args (dict): Arguments.
    """
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    start_time = time.time()

    logging.info('Load Data ...')
    data = CommFunc.load_json(args['data'])

    logging.info(f"Execute {args['method']} ...")
    normalizer = Normalizer()
    if args['method'] == 'min_max':
        process = normalizer.min_max
    elif args['method'] == 'mean':
        process = normalizer.mean
    elif args['method'] == 'standardization':
        process = normalizer.standardization
    result, info = _normalize(data, process)

    CommFunc.save_data(result, args['output'], save_type='json')
    logging.info(f"Save result to {args['output']}")

    basename, ext = os.path.splitext(args['output'])
    output  = f'{basename}_info{ext}'
    CommFunc.save_data(info, output, save_type='json')
    logging.info(f'Save information to {output}')

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


def parse_args(args: argparse.Namespace) -> dict:
    """Parse arguments.

    Args:
        args (argparse.Namespace): Arguments.

    Returns:
        dict: Arguments.
    """
    items = (
        'method',
        'data',
        'output',
    )
    return {item: getattr(args, item) for item in items}


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Normalize Data.',
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        '--method',
        type=str,
        choices=['min_max', 'mean', 'standardization'],
        required=True,
        help='\n'.join([
            'mean: Mean normalization.',
            'min_max: Min-max normalization.',
            'standardization: Standardization.',
        ]))
    arg_parser.add_argument(
        '--data',
        type=argparse_helper.files,
        required=True,
        help='Product vector file in json format, e.q.'
             ' product_vector/doc2vec/size_64/product.json')
    arg_parser.add_argument('--output', type=str, required=True, help='output')
    args = arg_parser.parse_args()

    main(parse_args(args))
