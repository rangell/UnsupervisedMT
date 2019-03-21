# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import torch

from ..utils import create_word_masks, create_st_word_masks
from .dataset import MonolingualDataset, ParallelDataset, StyleDataset
from .dictionary import EOS_WORD, PAD_WORD, UNK_WORD, SPECIAL_WORD, SPECIAL_WORDS

from IPython import embed
import json
import pandas as pd
from functools import reduce
from operator import mul

logger = getLogger()


loaded_data = {}  # store binarized datasets in memory in case of multiple reloadings


def load_binarized(path, params):
    """
    Load a binarized dataset and log main statistics.
    """
    if path in loaded_data:
        logger.info("Reloading data loaded from %s ..." % path)
        return loaded_data[path]
    assert os.path.isfile(path), path
    logger.info("Loading data from %s ..." % path)
    data = torch.load(path)
    data['positions'] = data['positions'].numpy()
    logger.info("%i words (%i unique) in %i sentences. %i unknown words (%i unique)." % (
        len(data['sentences']) - len(data['positions']),
        len(data['dico']), len(data['positions']),
        sum(data['unk_words'].values()), len(data['unk_words'])
    ))
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info("Selecting %i most frequent words ..." % params.max_vocab)
        data['dico'].prune(params.max_vocab)
        data['sentences'].masked_fill_((data['sentences'] >= params.max_vocab), data['dico'].index(UNK_WORD))
        unk_count = (data['sentences'] == data['dico'].index(UNK_WORD)).sum()
        logger.info("Now %i unknown words covering %.2f%% of the data." % (
            unk_count, 100. * unk_count / (len(data['sentences']) - len(data['positions']))
        ))
    loaded_data[path] = data
    return data


def load_vocab(params, data):
    """
    Load vocabulary files.
    """
    if not hasattr(params, 'vocab') or len(params.vocab) == 0:
        assert getattr(params, 'vocab_min_count', 0) == 0
        return

    data['vocab'] = {}

    for lang, path in params.vocab.items():

        assert lang in params.langs
        logger.info('============ Vocabulary (%s)' % lang)

        def read_vocab(path, min_count):
            assert os.path.isfile(path)
            vocab = set()
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.rstrip().split()
                    if not len(line) == 2:
                        assert len(line) == 1
                        logger.warning("Incorrect vocabulary word in line %i!" % i)
                        continue
                    count = int(line[1])
                    assert count > 0
                    if count < min_count:
                        break
                    vocab.add(line[0])
            return vocab

        data['vocab'][lang] = read_vocab(path, params.vocab_min_count)

        # check vocabulary words
        for w in data['vocab'][lang]:
            if w not in data['dico'][lang]:
                logger.warning("\"%s\" not found in the vocabulary!" % w)

        logger.info("Loaded %i words from the \"%s\" vocabulary."
                    % (len(data['vocab'][lang]), lang))

    logger.info('')


def load_st_vocab(params, data):
    """
    Load vocabulary files.
    """
    if not hasattr(params, 'vocab_filename'):
        assert getattr(params, 'vocab_min_count', 0) == 0
        return

    def read_vocab(path, min_count):
        assert os.path.isfile(path)
        vocab = set()
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.rstrip().split()
                if not len(line) == 2:
                    assert len(line) == 1
                    logger.warning("Incorrect vocabulary word in line %i!" % i)
                    continue
                count = int(line[1])
                assert count > 0
                if count < min_count:
                    break
                vocab.add(line[0])
        return vocab

    data['vocab'] = read_vocab(params.vocab_filename, params.vocab_min_count)

    # check vocabulary words
    for w in data['vocab']:
        if w not in data['dico']:
            logger.warning("\"%s\" not found in the vocabulary!" % w)

    logger.info("Loaded %i words from the vocabulary."
                % (len(data['vocab'])))

    logger.info('')


def set_parameters(params, dico):
    """
    Define parameters / check dictionaries.
    """
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    blank_index = dico.index(SPECIAL_WORD % 0)
    bos_index = [dico.index(SPECIAL_WORD % (i + 1)) for i in range(params.n_langs)]
    if hasattr(params, 'eos_index'):
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.blank_index == blank_index
        assert params.bos_index == bos_index
    else:
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.blank_index = blank_index
        params.bos_index = bos_index


def check_dictionaries(params, data):
    """
    Check dictionaries.
    """
    assert set(data['dico'].keys()) == set(params.langs)
    params.n_words = [len(data['dico'][lang]) for lang in params.langs]
    # for lang, dico in data['dico'].items():
    #     dico.lang = lang

    dico_0 = data['dico'][params.langs[0]]

    # check dictionaries indexes
    _SPECIAL_WORDS = ([EOS_WORD, PAD_WORD, UNK_WORD] +
                      [SPECIAL_WORD % i for i in range(SPECIAL_WORDS)])
    for i in range(1, params.n_langs):
        dico_i = data['dico'][params.langs[i]]
        assert all(dico_0.index(x) == dico_i.index(x) for x in _SPECIAL_WORDS)

    assert (not getattr(params, 'share_lang_emb', False) or
            all(data['dico'][params.langs[0]] == data['dico'][params.langs[i]]
                for i in range(1, params.n_langs)))


def check_st_dictionary(params, data):
    pass


def load_para_data(params, data):
    """
    Load parallel data.
    """
    assert len(params.para_dataset) > 0

    for (lang1, lang2), paths in params.para_dataset.items():

        assert lang1 in params.langs and lang2 in params.langs
        logger.info('============ Parallel data (%s - %s)' % (lang1, lang2))

        datasets = []

        for name, path in zip(['train', 'valid', 'test'], paths):
            if path == '':
                assert name == 'train'
                datasets.append((name, None))
                continue
            assert name != 'train' or params.n_para != 0

            # load data
            data1 = load_binarized(path.replace('XX', lang1), params)
            data2 = load_binarized(path.replace('XX', lang2), params)
            set_parameters(params, data1['dico'])
            set_parameters(params, data2['dico'])

            # set / check dictionaries
            if lang1 not in data['dico']:
                data['dico'][lang1] = data1['dico']
            else:
                assert data['dico'][lang1] == data1['dico']
            if lang2 not in data['dico']:
                data['dico'][lang2] = data2['dico']
            else:
                assert data['dico'][lang2] == data2['dico']

            # parallel data
            para_data = ParallelDataset(
                data1['sentences'], data1['positions'], data['dico'][lang1], params.lang2id[lang1],
                data2['sentences'], data2['positions'], data['dico'][lang2], params.lang2id[lang2],
                params
            )

            # remove too long sentences (train / valid only, test must remain unchanged)
            if name != 'test':
                para_data.remove_long_sentences(params.max_len)

            # select a subset of sentences
            if name == 'train' and params.n_para != -1:
                para_data.select_data(0, params.n_para)
            # if name == 'valid':
            #     para_data.select_data(0, 100)
            # if name == 'test':
            #     para_data.select_data(0, 167)

            datasets.append((name, para_data))

        assert (lang1, lang2) not in data['para']
        data['para'][(lang1, lang2)] = {k: v for k, v in datasets}

    logger.info('')


def load_back_data(params, data):
    """
    Load back-parallel data.
    """
    assert not (len(params.back_dataset) == 0) ^ (params.n_back == 0)

    for (lang1, lang2), (src_path, tgt_path) in params.back_dataset.items():

        assert lang1 in params.langs and lang2 in params.langs
        assert os.path.isfile(src_path)
        assert os.path.isfile(tgt_path)

        logger.info('============ Back-parallel data (%s - %s)' % (lang1, lang2))

        # load data
        data1 = load_binarized(src_path, params)
        data2 = load_binarized(tgt_path, params)
        set_parameters(params, data1['dico'])
        set_parameters(params, data2['dico'])

        # set / check dictionaries
        if lang1 not in data['dico']:
            data['dico'][lang1] = data1['dico']
        else:
            assert data['dico'][lang1] == data1['dico']
        if lang2 not in data['dico']:
            data['dico'][lang2] = data2['dico']
        else:
            assert data['dico'][lang2] == data2['dico']

        # parallel data
        para_data = ParallelDataset(
            data1['sentences'], data1['positions'], data['dico'][lang1], params.lang2id[lang1],
            data2['sentences'], data2['positions'], data['dico'][lang2], params.lang2id[lang2],
            params
        )

        # remove too long sentences
        para_data.remove_long_sentences(params.max_len)

        # select a subset of sentences
        if params.n_back != -1:
            para_data.select_data(0, params.n_back)

        assert (lang1, lang2) not in data['back']
        data['back'][(lang1, lang2)] = para_data

    logger.info('')


def load_mono_data(params, data):
    """
    Load monolingual data.
    """
    assert not (len(params.mono_dataset) == 0) ^ (params.n_mono == 0)
    if len(params.mono_dataset) == 0:
        return

    for lang, paths in params.mono_dataset.items():

        assert lang in params.langs
        logger.info('============ Monolingual data (%s)' % lang)

        datasets = []

        for name, path in zip(['train', 'valid', 'test'], paths):
            if path == '':
                assert name != 'train'
                datasets.append((name, None))
                continue

            # load data
            mono_data = load_binarized(path, params)
            set_parameters(params, mono_data['dico'])

            # set / check dictionary
            if lang not in data['dico']:
                data['dico'][lang] = mono_data['dico']
            else:
                assert data['dico'][lang] == mono_data['dico']

            # monolingual data
            mono_data = MonolingualDataset(mono_data['sentences'], mono_data['positions'],
                                           data['dico'][lang], params.lang2id[lang], params)

            # remove too long sentences (train / valid only, test must remain unchanged)
            if name != 'test':
                mono_data.remove_long_sentences(params.max_len)

            # select a subset of sentences
            if name == 'train' and params.n_mono != -1:
                mono_data.select_data(0, params.n_mono)

            datasets.append((name, mono_data))

        assert lang not in data['mono']
        data['mono'][lang] = {k: v for k, v in datasets}

    logger.info('')


def set_st_parameters(params, dico):
    """
    Define parameters / check dictionaries.
    """
    eos_index = dico.index(EOS_WORD)
    pad_index = dico.index(PAD_WORD)
    unk_index = dico.index(UNK_WORD)
    blank_index = dico.index(SPECIAL_WORD % 0)
    bos_index = dico.index(SPECIAL_WORD % 1)
    if hasattr(params, 'eos_index'):
        assert params.eos_index == eos_index
        assert params.pad_index == pad_index
        assert params.unk_index == unk_index
        assert params.blank_index == blank_index
        assert params.bos_index == bos_index
    else:
        params.eos_index = eos_index
        params.pad_index = pad_index
        params.unk_index = unk_index
        params.blank_index = blank_index
        params.bos_index = bos_index


def load_attributes(path, params):
    attr_df = pd.read_csv(path, header=None)
    attr_df = attr_df.applymap(lambda x : params.style2id[x])
    return attr_df.values


def load_st_data(params):
    """
    Load style transfer data.
    The returned dictionary contains:
        - dico (dictionary)
        - vocab (vocabulary)
        - splits (dictionary of datasets (train, valid, test))
    """
    data = {'dico': None, 'vocab': None, 'splits': {}}

    _prefixes = [params.train_prefix, params.dev_prefix, params.test_prefix]

    for name, prefix in zip(['train', 'dev', 'test'], _prefixes):
        text_path = ".".join([prefix, params.text_suffix])
        text_data = load_binarized(text_path, params)

        if not data['dico']:
            data['dico'] = text_data['dico']
        else:
            assert data['dico'] == text_data['dico']

        set_st_parameters(params, text_data['dico'])

        attribute_path = ".".join([prefix, params.attribute_suffix])
        attribute_data = load_attributes(attribute_path, params)

        dataset = StyleDataset(text_data['sentences'], text_data['positions'],
                               text_data['dico'], attribute_data, params)

        # remove too long sentences (train / valid only, test must remain unchanged)
        if name != 'test':
            dataset.remove_long_sentences(params.max_len)

        data['splits'][name] = dataset

    # update parameters
    params.n_words = len(data['dico'])

    # vocabulary
    load_st_vocab(params, data)

    return data


def check_all_data_params(params):
    """
    Check datasets parameters.
    """

    # make sure dataset files exist
    for suffix in [params.text_suffix, params.attribute_suffix]:
        for prefix in [params.train_prefix, params.dev_prefix, params.test_prefix]:
            assert os.path.isfile(".".join([prefix, suffix]))

    # make sure vocab file exists
    assert os.path.isfile(params.vocab_filename)

    # check params related to training 
    assert params.max_len > 0
    assert params.max_vocab == -1 or params.max_vocab > 0
    assert params.word_shuffle == 0 or params.word_shuffle > 1
    assert 0 <= params.word_dropout < 1
    assert 0 <= params.word_blank < 1

    # check styles
    assert os.path.isfile(params.metadata_filename)
    params.style_metadata = json.load(open(params.metadata_filename, 'r'))
    params.styles = []
    for attr_dict in params.style_metadata['attributes']:
        params.styles += attr_dict[list(attr_dict.keys())[0]]
    params.n_styles = len(params.styles)
    params.id2style = {k : v for k, v in enumerate(params.styles)}
    params.style2id = {k : v for v, k in params.id2style.items()}

    # style sampling variables
    params.modulo_constants = []
    for attribute_dict in params.style_metadata['attributes']:
        key = list(attribute_dict.keys())[0]
        params.modulo_constants.append(len(attribute_dict[key]))
    params.n_attributes = len(params.modulo_constants)
    params.n_uniq_styles = reduce(mul, params.modulo_constants, 1)
    params.start_indices = [sum(params.modulo_constants[:i]) 
                                for i in range(params.n_attributes)]
    params.integer_divisors = [reduce(mul, params.modulo_constants[:i], 1)
                                    for i in range(params.n_attributes)]

    params.modulo_constants = torch.LongTensor(params.modulo_constants)
    params.start_indices = torch.LongTensor(params.start_indices)
    params.integer_divisors = torch.LongTensor(params.integer_divisors)

    # definitely want to share embeddings between enc & dec
    assert params.share_encdec_emb == True

    # max length / max vocab / sentence noise
    assert params.max_len > 0
    assert params.max_vocab == -1 or params.max_vocab > 0
    assert params.word_shuffle == 0 or params.word_shuffle > 1
    assert 0 <= params.word_dropout < 1
    assert 0 <= params.word_blank < 1


def load_data(params, mono_only=False):
    """
    Load parallel / monolingual data.
    We start with the parallel test set, which defines the dictionaries.
    Each other dataset has to match the same dictionaries.
    The returned dictionary contains:
        - dico (dictionary of dictionaries)
        - vocab (dictionary of vocabularies)
        - mono (dictionary of monolingual datasets (train, valid, test))
        - para (dictionary of parallel datasets (train, valid, test))
        - back (dictionary of parallel datasets (train only))
    """
    data = {'dico': {}, 'mono': {}, 'para': {}, 'back': {}}

    if not mono_only:

        # parallel datasets
        load_para_data(params, data)

        # back-parallel datasets
        load_back_data(params, data)

    # monolingual datasets
    load_mono_data(params, data)

    # update parameters
    check_dictionaries(params, data)

    # vocabulary
    load_vocab(params, data)
    create_word_masks(params, data)

    # data summary
    logger.info('============ Data summary')
    for (lang1, lang2), v in data['para'].items():
        for data_type in ['train', 'valid', 'test']:
            if v[data_type] is None:
                continue
            logger.info('{: <18} - {: >5} - {: >4} -> {: >4}:{: >10}'.format('Parallel data', data_type, lang1, lang2, len(v[data_type])))

    for (lang1, lang2), v in data['back'].items():
        logger.info('{: <18} - {: >5} - {: >4} -> {: >4}:{: >10}'.format('Back-parallel data', 'train', lang1, lang2, len(v)))

    for lang, v in data['mono'].items():
        for data_type in ['train', 'valid', 'test']:
            logger.info('{: <18} - {: >5} - {: >12}:{: >10}'.format('Monolingual data', data_type, lang, len(v[data_type]) if v[data_type] is not None else 0))

    if hasattr(params, 'vocab') and len(params.vocab) > 0:
        for lang in params.langs:
            logger.info("Vocabulary - {: >4}):{: >7} words".format(lang, len(data['vocab'][lang])))

    logger.info('')
    return data
