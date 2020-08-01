#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WordCutter class and execution."""
import argparse
import logging
import os
import time
from typing import Union

from ckiptagger import (
    data_utils,
    WS,
)
import pandas as pd
from zhon.hanzi import punctuation as hanzi_punctuation

import argparse_helper
from comm_func import CommFunc
import pandas_helper


__author__ = 'haochun.fu'
__date__ = '2019-12-22'

class WordCutter(object):
    """WordCutter.
    
    Cut sentences into words.
    """

    """Base directory of data."""
    BASE_DATA_DIR = 'ckiptagger_data'

    """Directory of data."""
    DATA_DIR = 'data'


    def __init__(self, data_dir: str = None) -> None:
        """Constructor.
        
        Args:
            data_dir: Directory of required data of ckiptagger. Default is
                ckiptagger_data in script directory.

        Raises:
            FileNotFoundError: If data directory is not found.
        """
        if data_dir is not None:
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"'{data_dir}' is not found.")
        else:
            data_dir = os.path.join(
                           os.path.dirname(os.path.realpath(__file__)),
                           WordCutter.BASE_DATA_DIR
                       )
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                data_utils.download_data(data_dir)
            data_dir = os.path.join(data_dir, WordCutter.DATA_DIR)
            if not os.path.isdir(data_dir):
                raise RuntimeError('Failed to download ckiptagger data.')
                
        self.__ws = WS(data_dir)

    def cut(self, data: Union[list, str]) -> list:
        """Cut data into words.

        Args:
            data (list|str): Data.

        Returns:
            list: List of words.
        """
        if isinstance(data, str):
            data = [data]
        return self.__ws(data)


def main(args: dict) -> None:
    """Execution.

    Args:
        args (dict): Arguments.
    """
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    start_time = time.time()

    data = pd.read_csv(args['data'], dtype=str)

    items = ('name', 'desc_short')
    products = []
    for row in pandas_helper.iterate_with_dict(data):
        product = {item: row[item] for item in items}
        product['contract_id'] = row['contractid']
        products.append(product)

    sentences = []
    for row in products:
        for item in items:
            sentences.append(row[item] if not pd.isna(row[item]) else '')
    sentences = WordCutter().cut(sentences)

    result = {}
    for i, row in enumerate(products):
        product = result.setdefault(row['contract_id'], {})
        for j, item in enumerate(items):
            val = sentences[2 * i + j]
            product[item] = val if val else None

    output = args['output'] + '.json'
    CommFunc.save_data(result, output, save_type='json')
    logging.info(f"Save result to {output}")

    result = []
    for row in sentences:
        if not row:
            continue
        result.append(row)

    output = args['output'] + '_corpus.json'
    CommFunc.save_data(result, output, save_type='json')
    logging.info(f"Save result to {output}")

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
        'data',
        'output',
    )
    return {item: getattr(args, item) for item in items}


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Cut sentences into words.')
    arg_parser.add_argument('--data',
                            type=argparse_helper.files,
                            required=True,
                            help='Data in CSV format.')
    arg_parser.add_argument('--output',
                            required=True,
                            help='Output file.')
    args = arg_parser.parse_args()

    main(parse_args(args))
