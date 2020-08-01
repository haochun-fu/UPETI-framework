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
__date__ = '2020-07-04'


class Normalizer(object):
    """Normalizer."""


    """Name of mean."""
    METHOD_MEAN = 'mean'

    """Name of min-max."""
    METHOD_MIN_MAX = 'min_max'

    """Name of standardization."""
    METHOD_STANDARDIZATION = 'standardization'


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
    method: Callable[[np.ndarray], Tuple[np.ndarray, dict]]
) -> Tuple[dict, dict]:
    """Normalize.

    Args:
        data (dict): Data.
        method (class 'method'): Method.

    Returns:
        tuple: Result and information.
    """
    result = {id_: None for id_ in data}

    idx_to_id = {}
    vectors = []
    idx = 0
    for id_, vector in data.items():
        if vector is None:
            continue
    
        vectors.append(vector)
        idx_to_id[idx] = id_
        idx += 1
    
    vectors, info = method(np.asarray(vectors, dtype=np.float64))
    
    for idx, id_ in idx_to_id.items():
        result[id_] = vectors[idx].tolist()

    for key, val in info.items():
        info[key] = val.tolist()
    
    return result, info


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

    logging.info('Load Data ...')
    data = CommFunc.load_json(args.data)

    logging.info(f"Execute {args.method} ...")
    normalizer = Normalizer()
    method = {
        Normalizer.METHOD_MEAN: normalizer.mean,
        Normalizer.METHOD_MIN_MAX: normalizer.min_max,
        Normalizer.METHOD_STANDARDIZATION: normalizer.standardization,
    }[args.method]
    result, info = _normalize(data, method)

    CommFunc.save_data(result, args.output, save_type='json')
    logging.info(f"Save result to {args.output}")

    basename, ext = os.path.splitext(args.output)
    output  = f'{basename}_info{ext}'
    CommFunc.save_data(info, output, save_type='json')
    logging.info(f'Save information to {output}')

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Normalize Data.',
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        '--method',
        choices=[
            Normalizer.METHOD_MEAN,
            Normalizer.METHOD_MIN_MAX,
            Normalizer.METHOD_STANDARDIZATION
        ],
        required=True,
        help='\n'.join([
            f'{Normalizer.METHOD_MEAN}: Mean normalization.',
            f'{Normalizer.METHOD_MIN_MAX}: Min-max normalization.',
            f'{Normalizer.METHOD_STANDARDIZATION}: Standardization.',
        ]))
    arg_parser.add_argument(
        '--data',
        type=argparse_helper.files,
        required=True,
        help='Item vector JSON file, e.q. vector/item/doc2vec/exp1/item.json')
    arg_parser.add_argument('--output', required=True, help='Output.')
    args = arg_parser.parse_args()

    main(args)
