# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch

from IPython import embed

logger = getLogger()


class Dataset(object):

    def __init__(self, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.unk_index = params.unk_index
        self.bos_index = params.bos_index
        self.batch_size = params.batch_size

    def batch_sentences(self, sentences, lang_id):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (s_len, n) where s_len is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        assert type(lang_id) is int
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max(), lengths.size(0)).fill_(self.pad_index)

        sent[0] = self.bos_index[lang_id]
        for i, s in enumerate(sentences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths


class MonolingualDataset(Dataset):

    def __init__(self, sent, pos, dico, lang_id, params):
        super(MonolingualDataset, self).__init__(params)
        assert type(lang_id) is int
        self.sent = sent
        self.pos = pos
        self.dico = dico
        self.lang_id = lang_id
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.is_parallel = False

        # check number of sentences
        assert len(self.pos) == (self.sent == -1).sum()

        self.remove_empty_sentences()

        assert len(pos) == (sent[torch.from_numpy(pos[:, 1])] == -1).sum()  # check sentences indices
        assert -1 <= sent.min() < sent.max() < len(dico)                    # check dictionary indices
        assert self.lengths.min() > 0                                       # check empty sentences

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos)
        if a < b:
            self.pos = self.pos[a:b]
            self.lengths = self.pos[:, 1] - self.pos[:, 0]
        else:
            self.pos = torch.LongTensor()
            self.lengths = torch.LongTensor()

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos = self.pos[sentence_ids]
                sent = [self.sent[a:b] for a, b in pos]
                yield self.batch_sentences(sent, self.lang_id)
        return iterator

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths[indices], kind='mergesort')]

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)


class ParallelDataset(Dataset):

    def __init__(self, sent1, pos1, dico1, lang1_id, sent2, pos2, dico2, lang2_id, params):
        super(ParallelDataset, self).__init__(params)
        assert type(lang1_id) is int
        assert type(lang2_id) is int
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.dico1 = dico1
        self.dico2 = dico2
        self.lang1_id = lang1_id
        self.lang2_id = lang2_id
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.is_parallel = True

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == -1).sum()
        assert len(self.pos2) == (self.sent2 == -1).sum()

        self.remove_empty_sentences()

        assert len(pos1) == len(pos2) > 0                                      # check number of sentences
        assert len(pos1) == (sent1[torch.from_numpy(pos1[:, 1])] == -1).sum()  # check sentences indices
        assert len(pos2) == (sent2[torch.from_numpy(pos2[:, 1])] == -1).sum()  # check sentences indices
        assert -1 <= sent1.min() < sent1.max() < len(dico1)                    # check dictionary indices
        assert -1 <= sent2.min() < sent2.max() < len(dico2)                    # check dictionary indices
        assert self.lengths1.min() > 0                                         # check empty sentences
        assert self.lengths2.min() > 0                                         # check empty sentences

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos1)
        if a < b:
            self.pos1 = self.pos1[a:b]
            self.pos2 = self.pos2[a:b]
            self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
            self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        else:
            self.pos1 = torch.LongTensor()
            self.pos2 = torch.LongTensor()
            self.lengths1 = torch.LongTensor()
            self.lengths2 = torch.LongTensor()

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos1 = self.pos1[sentence_ids]
                pos2 = self.pos2[sentence_ids]
                sent1 = [self.sent1[a:b] for a, b in pos1]
                sent2 = [self.sent2[a:b] for a, b in pos2]
                yield self.batch_sentences(sent1, self.lang1_id), self.batch_sentences(sent2, self.lang2_id)
        return iterator

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths2[indices], kind='mergesort')]
            indices = indices[np.argsort(self.lengths1[indices], kind='mergesort')]

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)


############################################################################### 


class StyleDataset(object):

    def __init__(self, params):
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.unk_index = params.unk_index
        self.bos_index = params.bos_index
        self.batch_size = params.batch_size

    def batch_sentences(self, sentences, attributes):
        """
        Take as input a list of n sentences (torch.LongTensor vectors) and return
        a tensor of size (s_len, n) where s_len is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sentences])
        sent = torch.LongTensor(lengths.max(), lengths.size(0)).fill_(self.pad_index)
        attr = torch.LongTensor(attributes)

        sent[0] = self.bos_index    # BOS token should be replaced by style embeds
        for i, s in enumerate(sentences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths, attr

class UnpairedStyleDataset(StyleDataset):

    def __init__(self, sent, pos, dico, attr, params):
        super(UnpairedStyleDataset, self).__init__(params)

        self.sent = sent
        self.pos = pos
        self.dico = dico
        self.attr = attr
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        self.is_parallel = False

        # check number of sentences
        assert len(self.pos) == (self.sent == -1).sum()
        assert len(self.pos) == len(self.attr)

        self.remove_empty_sentences()

        assert len(pos) == (sent[torch.from_numpy(pos[:, 1])] == -1).sum()  # check sentences indices
        assert -1 <= sent.min() < sent.max() < len(dico)                    # check dictionary indices
        assert self.lengths.min() > 0                                       # check empty sentences

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] > 0]
        self.pos = self.pos[indices]
        self.attr = self.attr[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos)
        indices = np.arange(len(self.pos))
        indices = indices[self.lengths[indices] <= max_len]
        self.pos = self.pos[indices]
        self.attr = self.attr[indices]
        self.lengths = self.pos[:, 1] - self.pos[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos)
        if a < b:
            self.pos = self.pos[a:b]
            self.attr = self.attr[a:b]
            self.lengths = self.pos[:, 1] - self.pos[:, 0]
        else:
            self.pos = torch.LongTensor()
            self.attr = torch.LongTensor()
            self.lengths = torch.LongTensor()

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos = self.pos[sentence_ids]
                attr = self.attr[sentence_ids]
                sent = [self.sent[a:b] for a, b in pos]
                yield self.batch_sentences(sent, attr)
        return iterator

    def balance_dataset(self, indices):
        assert self.attr.shape[1] == 1 # Only one attribute we're controlling

        styles = np.unique(self.attr)
        style_counts = [np.sum(self.attr == s) for s in styles]
        max_style_count = max(style_counts)
        num_to_sample = [max_style_count - x for x in style_counts]

        for s, to_sample in zip(styles, num_to_sample):
            if to_sample == 0:
                continue
            s_indices = indices[np.squeeze(self.attr[indices] == s)]
            ind_of_ind = np.random.choice(s_indices.size, to_sample,
                                          replace=False)
            indices = np.append(indices, s_indices[ind_of_ind])

        return indices

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        indices = np.arange(n_sentences)
        if shuffle:
            np.random.shuffle(indices)

        if self.attr.shape[1] == 1:
            indices = self.balance_dataset(indices)
        else:
            raise NotImplementedError("Iterator not implemented for multiple attributes")

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths[indices], kind='mergesort')]

        if self.attr.shape[1] == 1:
            styles = np.unique(self.attr)
            n_styles = styles.size
            group_indices = np.zeros(indices.shape, dtype='int64')
            for i, s, in zip(range(n_styles), styles):
                s_indices = np.squeeze(self.attr[indices] == s)
                group_indices[i::n_styles] = indices[s_indices]
            indices = group_indices
        else:
            raise NotImplementedError("Iterator not implemented for multiple attributes")

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)


class PairedStyleDataset(StyleDataset):

    def __init__(self, sent1, pos1, dico1, attr1, sent2, pos2, dico2, attr2, params):
        super(PairedStyleDataset, self).__init__(params)
        self.sent1 = sent1
        self.sent2 = sent2
        self.pos1 = pos1
        self.pos2 = pos2
        self.dico1 = dico1
        self.dico2 = dico2
        self.attr1 = attr1
        self.attr2 = attr2
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        self.is_parallel = True

        # check number of sentences
        assert len(self.pos1) == (self.sent1 == -1).sum()
        assert len(self.pos2) == (self.sent2 == -1).sum()

        self.remove_empty_sentences()

        assert len(pos1) == len(pos2) > 0                                      # check number of sentences
        assert len(pos1) == (sent1[torch.from_numpy(pos1[:, 1])] == -1).sum()  # check sentences indices
        assert len(pos2) == (sent2[torch.from_numpy(pos2[:, 1])] == -1).sum()  # check sentences indices
        assert -1 <= sent1.min() < sent1.max() < len(dico1)                    # check dictionary indices
        assert -1 <= sent2.min() < sent2.max() < len(dico2)                    # check dictionary indices
        assert self.lengths1.min() > 0                                         # check empty sentences
        assert self.lengths2.min() > 0                                         # check empty sentences

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return len(self.pos1)

    def remove_empty_sentences(self):
        """
        Remove empty sentences.
        """
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] > 0]
        indices = indices[self.lengths2[indices] > 0]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i empty sentences." % (init_size - len(indices)))

    def remove_long_sentences(self, max_len):
        """
        Remove sentences exceeding a certain length.
        """
        assert max_len > 0
        init_size = len(self.pos1)
        indices = np.arange(len(self.pos1))
        indices = indices[self.lengths1[indices] <= max_len]
        indices = indices[self.lengths2[indices] <= max_len]
        self.pos1 = self.pos1[indices]
        self.pos2 = self.pos2[indices]
        self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
        self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        logger.info("Removed %i too long sentences." % (init_size - len(indices)))

    def select_data(self, a, b):
        """
        Only retain a subset of the dataset.
        """
        assert 0 <= a <= b <= len(self.pos1)
        if a < b:
            self.pos1 = self.pos1[a:b]
            self.pos2 = self.pos2[a:b]
            self.lengths1 = self.pos1[:, 1] - self.pos1[:, 0]
            self.lengths2 = self.pos2[:, 1] - self.pos2[:, 0]
        else:
            self.pos1 = torch.LongTensor()
            self.pos2 = torch.LongTensor()
            self.lengths1 = torch.LongTensor()
            self.lengths2 = torch.LongTensor()

    def get_batches_iterator(self, batches):
        """
        Return a sentences iterator, given the associated sentence batches.
        """
        def iterator():
            for sentence_ids in batches:
                pos1 = self.pos1[sentence_ids]
                pos2 = self.pos2[sentence_ids]
                attr1 = self.attr1[sentence_ids]
                attr2 = self.attr2[sentence_ids]
                sent1 = [self.sent1[a:b] for a, b in pos1]
                sent2 = [self.sent2[a:b] for a, b in pos2]
                yield self.batch_sentences(sent1, attr1), self.batch_sentences(sent2, attr2)
        return iterator

    def get_iterator(self, shuffle, group_by_size=False, n_sentences=-1):
        """
        Return a sentences iterator.
        """
        n_sentences = len(self.pos1) if n_sentences == -1 else n_sentences
        assert 0 < n_sentences <= len(self.pos1)
        assert type(shuffle) is bool and type(group_by_size) is bool

        # select sentences to iterate over
        if shuffle:
            indices = np.random.permutation(len(self.pos1))[:n_sentences]
        else:
            indices = np.arange(n_sentences)

        # group sentences by lengths
        if group_by_size:
            indices = indices[np.argsort(self.lengths2[indices], kind='mergesort')]
            indices = indices[np.argsort(self.lengths1[indices], kind='mergesort')]

        # create batches / optionally shuffle them
        batches = np.array_split(indices, math.ceil(len(indices) * 1. / self.batch_size))
        if shuffle:
            np.random.shuffle(batches)

        # return the iterator
        return self.get_batches_iterator(batches)
