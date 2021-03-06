#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vectorizer classes and execution."""
import argparse
import logging
import os
import re
import shutil
import string
import time
from typing import Union

from gensim.corpora import Dictionary
from gensim.models import (
    FastText,
    Word2Vec,
    TfidfModel,
)
from gensim.models.doc2vec import (
    Doc2Vec,
    TaggedDocument,
)
import numpy as np
from zhon.hanzi import punctuation as hanzi_punctuation

import argparse_helper
from comm_func import CommFunc


__author__ = 'haochun.fu'
__date__ = '2020-07-04'


class VectorizerBase(object):
    """VectorizerBase class."""


    """Model file name."""
    MODEL_NAME = 'model'

    """Symbols to be replaced."""
    REPLACED_SYMBOLS = ('\n', '◆', '■', '★', '➤', '🔴', '→', '←', '🎁''♛', '☆',
        '※', '◎', '✔', '▼', '●', '✦', '．', '✿', '▓', '\r', '✡', '⇪', '☞ ', '👍',
        '🔹', '➡', '💪', '↘', '◇', '•', '👏', '🚛', '💨', '🚵', '♀', '👨', '🔧',
        '👦', '➨', '\t', '►', '❤', '♥', '✒', '♦', '▤', '①', '②', '③', '✨',
        '👉')


    def __init__(self) -> None:
        """Constructor."""
        self._model = None
        self.__punctuation =\
            set(''.join((hanzi_punctuation, string.punctuation)))
        self.__filter = re.compile(
            '|'.join(VectorizerBase.REPLACED_SYMBOLS),
            re.MULTILINE)

    def _gen_corpus(self, data: list) -> list:
        """Generate corpus.

        Args:
            data (list): Data.

        Returns:
            list: Corpus.
        """
        ret = []

        for row in data:
            row = self._normalize(row)
            if not row:
                continue

            ret.append(row)

        return ret

    def _get_model_file(self, model_dir: str) -> str:
        """Get model file.

        Args:
            model_dir (str): Model directory.

        Returns:
            str: Model file.
        """
        return os.path.join(model_dir, VectorizerBase.MODEL_NAME)

    def _normalize(self, data: list) -> list:
        """Normalize data.

        Filter out unnecessary characters.

        Uncapitalize characters.

        Args:
            data(list): List of words.

        Returns:
            list: List of filtered words.
        """
        ret = []

        for word in data:
            word = self.__filter.sub('', word.strip())
            for c in self.__punctuation:
                if c in word:
                    word = word.replace(c, '')

            if word == '':
                continue

            ret.append(word.lower())

        return ret

    def save(self, output_dir: str) -> None:
        """Save model.

        Args:
            output: Output directory.
        """
        CommFunc.create_dir(output_dir)
        self._model.save(self._get_model_file(output_dir))


class Doc2VecVectorizer(VectorizerBase):
    """Doc2VecVectorizer class."""


    def load(self, model_dir: str) -> None:
        """Load model.
 
        Args:
            model_dir (str): Model directory.
        """
        self._model = Doc2Vec.load(self._get_model_file(model_dir))

    def train(self, data: list, **param: dict) -> None:
        """Train.

        Args:
            data (list): List of list of words of sentences.
            param (dict): Parameters.
        """
        docs = [
            TaggedDocument(doc, [i])\
                for i, doc in enumerate(self._gen_corpus(data))
        ]

        items = (
            'dm',
            'vector_size',
            'window',
            'alpha',
            'min_alpha',
            'seed',
            'min_count',
            'max_vocab_size',
            'sample',
            'workers',
            'epochs',
            'hs',
            'negative',
            'ns_exponent',
            'dm_mean',
            'dm_concat',
            'dm_tag_count',
            'dbow_words',
        )
        params = {item: param[item] for item in items}
        self._model = Doc2Vec(docs, **params)

    def vectorize(self, data: str) -> np.ndarray:
        """Vectorize data.

        Args:
            data (str): data.

        Returns:
            numpy.ndarray: Vector or None if failed to vectorize data.
        """
        ret = None

        try:
            words = self._normalize(data.split(' '))
            ret = self._model.infer_vector(words)
        except Exception as e:
            print(f'[ERROR] vectorize({repr(data)}): {e}')

        return ret


class FastTextVectorizer(VectorizerBase):
    """FastTextVectorizer class."""


    def load(self, model_dir: str) -> None:
        """Load model.
 
        Args:
            model_dir (str): Model directory.
        """
        self._model = FastText.load(self._get_model_file(model_dir))

    def train(self, data: list, **param: dict) -> None:
        """Train.

        Args:
            data (list): List of list of words of sentences.
            param (dict): Parameters.
        """
        items = (
            'min_count',
            'size',
            'window',
            'workers',
            'alpha',
            'min_alpha',
            'sg',
            'hs',
            'seed',
            'max_vocab_size',
            'sample',
            'negative',
            'ns_exponent',
            'cbow_mean',
            'iter',
            'sorted_vocab',
            'batch_words',
            'min_n',
            'max_n',
            'word_ngrams',
            'bucket',
            'compatible_hash',
        )
        params = {item: param[item] for item in items}
        self._model = FastText(self._gen_corpus(data), **params)

    def vectorize(self, data: str) -> np.ndarray:
        """Vectorize data.

        Args:
            data (str): Data.

        Returns:
            numpy.ndarray: Vector or None if failed to vectorize data.
        """
        ret = None

        try:
            words = self._normalize(data.split(' '))
            ret = self._model.wv.word_vec(' '.join(words))
        except Exception as e:
            print(f'[ERROR] vectorize({repr(data)}): {e}')

        return ret


class Word2VecVectorizer(VectorizerBase):
    """Word2VecVectorizer class."""


    def load(self, model_dir: str) -> None:
        """Load model.
 
        Args:
            model_dir (str): Model directory.
        """
        self._model = Word2Vec.load(self._get_model_file(model_dir))

    def train(self, data: list, **param: dict) -> None:
        """Train.

        Args:
            data (list): List of list of words of sentences.
            param (dict): Parameters.
        """
        items = (
            'size',
            'window',
            'min_count',
            'workers',
            'sg',
            'hs',
            'negative',
            'ns_exponent',
            'cbow_mean',
            'alpha',
            'min_alpha',
            'seed',
            'max_vocab_size',
            'max_final_vocab',
            'sample',
            'iter',
            'sorted_vocab',
            'batch_words',
        )
        params = {item: param[item] for item in items}
        self._model = Word2Vec(self._gen_corpus(data), **params)

    def vectorize(self, data: str) -> np.ndarray:
        """Vectorize data.

        Args:
            data (str): data.

        Returns:
            numpy.ndarray: Vector or None if failed to vectorize data.
        """
        ret = None

        try:
            words = self._normalize(data.split(' '))

            counter = 0
            for word in words:
                if word not in self._model.wv.vocab:
                    continue

                word_vec = self._model.wv[word]
                if not ret:
                    ret = word_vec
                else:
                    ret += word_vec

                counter += 1

            if ret is not None:
                ret = ret / counter
        except Exception as e:
            print(f'[ERROR] vectorize({repr(data)}): {e}')

        return ret


class TfIdfWord2VecVectorizer(VectorizerBase):
    """TfIdfWord2VecVectorizer class."""


    """Postfix of tf-idf model file."""
    TFIDF_FILE_POSTFIX = '.tfidf'

    """Postfix of dictionary file."""
    DICT_FILE_POSTFIX = '.dict'

    
    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

        self.__tfidf = None
        self.__tfidf_dict = None
        self.__vectorizer = Word2VecVectorizer()

    def __get_tfidf_dict_file(self, file: str) -> str:
        """Get tf-idf dictionary file.

        Args:
            file (str): Vectorizer model file.

        Returns:
            str: tf-idf dictionary file.
        """
        return self.__get_tfidf_model_file(file) \
               + TfIdfWord2VecVectorizer.DICT_FILE_POSTFIX

    def __get_tfidf_model_file(self, file: str) -> str:
        """Get tf-idf model file.

        Args:
            file (str): Vectorizer model file.

        Returns:
            str: tf-idf model file.
        """
        return file + TfIdfWord2VecVectorizer.TFIDF_FILE_POSTFIX

    def load(self, model_dir: str) -> None:
        """Load model.
 
        Args:
            model_dir (str): Model directory.
        """
        file = self._get_model_file(model_dir)
        self.__vectorizer.load(file)
        self.__tfidf = TfidfModel.load(self.__get_tfidf_model_file(file))
        self.__tfidf_dict = Dictionary.load(self.__get_tfidf_dict_file(file))

    def save(self, output_dir: str) -> None:
        """Save model.

        Args:
            output_dir (str): Output directory.
        """
        CommFunc.create_dir(output_dir)

        file = self._get_model_file(output_dir)
        self.__vectorizer.save(file)
        self.__tfidf.save(self.__get_tfidf_model_file(file))
        self.__tfidf_dict.save(self.__get_tfidf_dict_file(file))

    def train(self, data: list, **param: dict) -> None:
        """Train.

        Args:
            data (list): List of list of words of sentences.
            param (dict): Parameters.
        """
        self.__vectorizer.train(data, **param)

        corpus = self._gen_corpus(data)
        self.__tfidf_dict = Dictionary(corpus)

        tfidf_corpus = [self.__tfidf_dict.doc2bow(row) for row in corpus]
        self.__tfidf = TfidfModel(tfidf_corpus)

    def vectorize(self, data: str) -> np.ndarray:
        """Vectorize data.

        Args:
            data (str): data.

        Returns:
            numpy.ndarray: Vector or None if failed to vectorize data.
        """
        ret = None

        try:
            words = self._normalize(data.split(' '))

            counter = 0

            tfidf_dict_idxs = self.__tfidf_dict.doc2idx(words)
            words_bow = self.__tfidf_dict.doc2bow(words)
            words_dict_id_to_tfidf =\
                {dict_id: score for dict_id, score in self.__tfidf[words_bow]}

            for i, word in enumerate(words):
                word_vec = self.__vectorizer.vectorize(word)
                if word_vec is None:
                    continue

                weight = words_dict_id_to_tfidf[tfidf_dict_idxs[i]]\
                    if tfidf_dict_idxs[i] != -1 else 1
                if ret is None:
                    ret = weight * word_vec
                else:
                    ret += weight * word_vec

                counter += 1

            if ret is not None:
                ret = ret / counter
        except Exception as e:
            print(f'[ERROR] vectorize({repr(data)}): {e}')

        return ret


def _model_to_class(
    model_name: str
) -> Union[Doc2VecVectorizer, FastTextVectorizer, TfIdfWord2VecVectorizer]:
    """Get vectorizer class by model name.

    Args:
        model_name (str): Model name.

    Returns:
        Doc2VecVectorizer|FastTextVectorizer|TfIdfWord2VecVectorizer: Vectorizer
            class.

    Raises:
        ValueError: If model name is invalid.
    """
    ret = {
        'doc2vec': Doc2VecVectorizer,
        'fasttext': FastTextVectorizer,
        'tfidfWord2vec': TfIdfWord2VecVectorizer,
    }.get(model_name, None)

    if ret is None:
        raise ValueError(f"Model name '{model_name}' is invalid")

    return ret


def _train(args: argparse.Namespace) -> None:
    """Train.

    Args:
        args (argparse.Namespace): Arguments.
    """
    vectorizer = _model_to_class(args.model)()

    data = CommFunc.load_json(args.data)
    config = CommFunc.load_json(args.config)

    vectorizer.train(data, **config)
    vectorizer.save(args.output_dir)
    logging.info(f"Save model to '{args.output_dir}'")

    file = os.path.join(args.output_dir, 'config.json')
    shutil.copyfile(args.config, file)
    logging.info(f"Save config to '{file}'")


def _vectorize(args: argparse.Namespace) -> None:
    """Vectorize.

    Args:
        args (argparse.Namespace): Arguments.
    """
    vectorizer = _model_to_class(args.model)()
    vectorizer.load(args.model_dir)

    rs = {}
    for key, val in CommFunc.load_json(args.data).items():
        vector = vectorizer.vectorize(' '.join(val))
        rs[key] = vector.tolist() if vector is not None else None

    CommFunc.save_data(rs, args.output, save_type='json')
    logging.info(f"Save result to '{args.output}'")


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

    if args.item == 'train':
        _train(args)
    elif args.item == 'vectorize':
        _vectorize(args)

    elapsed_time = time.time() - start_time
    logging.info(f'Elapsed time: {CommFunc.second_to_time(elapsed_time)}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
         description='Vectorization.',
         formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        '--model',
        choices=[
            'doc2vec',
            'fasttext',
            'tfidfWord2vec',
        ],
        required=True,
        help='\n'.join([
            'doc2vec: doc2vec.',
            'fasttext: fasttext.',
            'tfidfWord2vec: word2vec with tf-idf.',
        ]))
    arg_parser.add_argument(
        '--item',
        choices=[
            'train',
            'vectorize',
        ],
        required=True,
        help='\n'.join([
            'train: Train model.',
            'vectorize: Vectorize data.',
        ]))
    arg_parser.add_argument(
        '--data',
        type=argparse_helper.files,
        required=True,
        help='\n'.join([
            'train: Corpus JSON file.',
            'vectorize: Data JSON file.',
        ]))
    arg_parser.add_argument('--output', help='Output file.')
    arg_parser.add_argument('--output_dir', help='Output directory.')
    arg_parser.add_argument(
        '--config',
        type=argparse_helper.files,
        help='Configuration file.')
    arg_parser.add_argument(
        '--model_dir',
        type=argparse_helper.dirs,
        help='Model direcdtory.')
    args = arg_parser.parse_args()

    try:
        if args.item == 'train':
            argparse_helper.check_miss_item(args, ('config', 'output_dir'))
        elif args.item == 'vectorize':
            argparse_helper.check_miss_item(args, ('model_dir', 'output'))

        main(args)
    except argparse_helper.MissingOption as err:
        print(err)
