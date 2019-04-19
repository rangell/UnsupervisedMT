# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
from collections import OrderedDict
from logging import getLogger
import numpy as np
import torch
from torch import nn

from .utils import restore_segmentation, sample_style
import fastText
import random

from IPython import embed

logger = getLogger()


TOOLS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
BLEU_SCRIPT_PATH = os.path.join(TOOLS_PATH, 'mosesdecoder/scripts/generic/multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH), "Moses not found. Please be sure you downloaded Moses in %s" % TOOLS_PATH


class EvaluatorMT(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.discriminator = trainer.discriminator
        self.data = data
        self.dico = data['dico']
        self.params = params

        # create reference files for BLEU evaluation
        self.create_reference_files()

        # load classfication models for computing style transfer accuracy
        self.load_classification_models()

    def get_pair_for_mono(self, lang):
        """
        Find a language pair for monolingual data.
        """
        candidates = [(l1, l2) for (l1, l2) in self.data['para'].keys() if l1 == lang or l2 == lang]
        assert len(candidates) > 0
        return sorted(candidates)[0]

    def mono_iterator(self, data_type, lang):
        """
        If we do not have monolingual validation / test sets, we take one from parallel data.
        """
        dataset = self.data['mono'][lang][data_type]
        if dataset is None:
            pair = self.get_pair_for_mono(lang)
            dataset = self.data['para'][pair][data_type]
            i = 0 if pair[0] == lang else 1
        else:
            i = None
        dataset.batch_size = 32
        for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
            yield batch if i is None else batch[i]

    def get_iterator(self, data_type):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['dev', 'test']
        dataset = self.data['splits'][data_type]
        dataset.batch_size = 32
        for batch in dataset.get_iterator(shuffle=False, group_by_size=True)():
            yield batch

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}
        params.attr_paths = {}

        for data_type in ['dev', 'test']:

            if data_type == 'test' and params.test_para:
                data_type == 'test_para'

            ref_path = os.path.join(params.dump_path, 'ref.{0}.txt'.format(data_type))
            attr_path = os.path.join(params.dump_path, 'ref.{0}.attr'.format(data_type))

            ref_txt = []
            ref_attr = []

            # convert to text
            for sent_, len_, attr_ in self.get_iterator(data_type):
                ref_txt.extend(convert_to_text(sent_, len_, self.dico, params))
                ref_attr.extend(convert_to_attr(attr_, params))

            # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
            ref_txt = [x.replace('<unk>', '<<unk>>') for x in ref_txt]

            # export hypothesis
            with open(ref_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ref_txt) + '\n')
            with open(attr_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(ref_attr) + '\n')

            # restore original segmentation
            restore_segmentation(ref_path)

            # store data paths
            params.ref_paths[data_type] = ref_path
            params.attr_paths[data_type] = attr_path

    def load_classification_models(self):
        params = self.params
        self.style_clfs = {}
        for attr_name in params.attr_names:
            clf_path = params.clf_paths[attr_name]
            self.style_clfs[attr_name] = fastText.load_model(clf_path)

    def eval_para(self, lang1, lang2, data_type, scores):
        """
        Evaluate lang1 -> lang2 perplexity and BLEU scores.
        """
        logger.info("Evaluating %s -> %s (%s) ..." % (lang1, lang2, data_type))
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        # hypothesis
        txt = []

        # for perplexity
        loss_fn2 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang2_id].weight, size_average=False)
        n_words2 = self.params.n_words[lang2_id]
        count = 0
        xe_loss = 0

        for batch in self.get_iterator(data_type, lang1, lang2):

            # batch
            (sent1, len1), (sent2, len2) = batch
            sent1, sent2 = sent1.cuda(), sent2.cuda()

            # encode / decode / generate
            encoded = self.encoder(sent1, len1, lang1_id)
            decoded = self.decoder(encoded, sent2[:-1], lang2_id)
            sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

            # cross-entropy loss
            xe_loss += loss_fn2(decoded.view(-1, n_words2), sent2[1:].view(-1)).item()
            count += (len2 - 1).sum().item()  # skip bos word

            # convert to text
            txt.extend(convert_to_text(sent2_, len2_, self.dico[lang2], lang2_id, self.params))

        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)
        ref_path = params.ref_paths[data_type]

        # export sentences to hypothesis file / restore bpe segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt) + '\n')
        restore_segmentation(hyp_path)

        # evaluate bleu score
        bleu = eval_moses_bleu(ref_path, hyp_path)
        assert False

        # update scores
        scores['self-bleu_%s' % (data_type)] = bleu

    def eval_content(self, data_type, scores):
        """
        Evaluate self-BLEU scores.
        """
        logger.info("Evaluating %s ..." % (data_type))
        assert data_type in ['dev', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params

        # hypothesis
        txt_list1 = [] # Coming from txt
        txt_list2 = [] # Going to txt
        attr_list1 = [] # Coming from attr
        attr_list2 = [] # Going to attr

        # for perplexity
        loss_fn2 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[0].weight, reduction='sum')
        n_words2 = self.params.n_words
        count = 0

        for batch in self.get_iterator(data_type):

            # batch
            sent1, len1, attr1 = batch
            attr2_ = sample_style(self.params, attr1)
            sent1, attr1, attr2_ = sent1.cuda(), attr1.cuda(), attr2_.cuda()

            # encode & generate
            encoded = self.encoder(sent1, len1, attr1)
            max_len = int(len1.max() + 10)
            sent2_, len2_, _ = self.decoder.generate(encoded, attr2_, max_len=max_len)

            # convert to text
            txt_list1.extend(convert_to_text(sent1, len1, self.dico, self.params))
            txt_list2.extend(convert_to_text(sent2_, len2_, self.dico, self.params))

            # convert attr id labels to text versions
            attr_list1.extend([list(x) for x in attr1.cpu().numpy()])
            attr_list2.extend([list(x) for x in attr2_.cpu().numpy()])

        # obtain predictions
        pred_attr = []
        for txt_sent in txt_list2:
            preds = []
            for attr_name in params.attr_names:
                pred_label = self.style_clfs[attr_name].predict(txt_sent)[0][0]
                pred_id = params.style2id[pred_label.replace('__label__', '')]
                preds.append(pred_id)
            pred_attr.append(preds)

        # Print out some examples here
        num_samples = 10
        sample_indices = random.sample(range(len(txt_list1)), num_samples)
        logger.info("Sampling random examples...\n")
        for i in sample_indices:
            logger.info("Example {}:".format(i) + "\noriginal attributes: " 
                        + " ".join([params.id2style[x] for x in attr_list1[i]])
                        + "\noriginal text: " + txt_list1[i]
                        + "\n\nintended attributes: "
                        + " ".join([params.id2style[x] for x in attr_list2[i]])
                        + "\npredicted attributes: "
                        + " ".join([params.id2style[x] for x in pred_attr[i]])
                        + "\ntransferred text: " + txt_list2[i] + "\n")

        orig_attr = np.asarray(attr_list1)
        tgt_attr = np.asarray(attr_list2)
        pred_attr = np.asarray(pred_attr)

        tsf_acc = list(np.mean(tgt_attr == pred_attr, axis=0))
        pred_orig_style = np.mean(np.all(orig_attr == pred_attr, axis=1))

        for attr_name, acc in zip(params.attr_names, tsf_acc):
            key = 'tsf_accuracy_%s_%s' % (attr_name, data_type)
            logger.info('%s: %f' % (key, acc))
            scores[key] = acc
        logger.info('pred_orig_style_%s: %f' % (data_type, pred_orig_style))
        scores['pred_orig_style_%s' % data_type] = pred_orig_style
                
        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}.txt'.format(scores['epoch'], data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)
        ref_path = params.ref_paths[data_type]

        # export sentences to hypothesis file / restore bpe segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_list2) + '\n')
        restore_segmentation(hyp_path)

        # evaluate bleu score
        bleu = eval_moses_bleu(ref_path, hyp_path)

        if data_type == 'test' and params.test_para:
            logger.info("bleu %s : %f - ref_path: %s - ref_path: %s " % (data_type, bleu, ref_path, hyp_path))
            # update scores
            scores['bleu_%s' % (data_type)] = bleu
        else:
            logger.info("self-bleu %s : %f - ref_path: %s - ref_path: %s " % (data_type, bleu, ref_path, hyp_path))
            # update scores
            scores['self-bleu_%s' % (data_type)] = bleu


    def eval_back(self, lang1, lang2, lang3, data_type, scores):
        """
        Compute lang1 -> lang2 -> lang3 perplexity and BLEU scores.
        """
        logger.info("Evaluating %s -> %s -> %s (%s) ..." % (lang1, lang2, lang3, data_type))
        assert data_type in ['valid', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        lang3_id = params.lang2id[lang3]

        # hypothesis
        txt = []

        # for perplexity
        loss_fn3 = nn.CrossEntropyLoss(weight=self.decoder.loss_fn[lang3_id].weight, size_average=False)
        n_words3 = self.params.n_words[lang3_id]
        count = 0
        xe_loss = 0

        for batch in self.get_iterator(data_type, lang1, lang3):

            # batch
            (sent1, len1), (sent3, len3) = batch
            sent1, sent3 = sent1.cuda(), sent3.cuda()

            # encode / generate lang1 -> lang2
            encoded = self.encoder(sent1, len1, lang1_id)
            sent2_, len2_, _ = self.decoder.generate(encoded, lang2_id)

            # encode / decode / generate lang2 -> lang3
            encoded = self.encoder(sent2_.cuda(), len2_, lang2_id)
            decoded = self.decoder(encoded, sent3[:-1], lang3_id)
            sent3_, len3_, _ = self.decoder.generate(encoded, lang3_id)

            # cross-entropy loss
            xe_loss += loss_fn3(decoded.view(-1, n_words3), sent3[1:].view(-1)).item()
            count += (len3 - 1).sum().item()  # skip BOS word

            # convert to text
            txt.extend(convert_to_text(sent3_, len3_, self.dico[lang3], lang3_id, self.params))

        # hypothesis / reference paths
        hyp_name = 'hyp{0}.{1}-{2}-{3}.{4}.txt'.format(scores['epoch'], lang1, lang2, lang3, data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)
        if lang1 == lang3:
            _lang1, _lang3 = self.get_pair_for_mono(lang1)
            if lang3 != _lang3:
                _lang1, _lang3 = _lang3, _lang1
            ref_path = params.ref_paths[(_lang1, _lang3, data_type)]
        else:
            ref_path = params.ref_paths[(lang1, lang3, data_type)]

        # export sentences to hypothesis file / restore BPE segmentation
        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt) + '\n')
        restore_segmentation(hyp_path)

        # evaluate BLEU score
        bleu = eval_moses_bleu(ref_path, hyp_path)
        logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))

        # update scores
        scores['ppl_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = np.exp(xe_loss / count)
        scores['bleu_%s_%s_%s_%s' % (lang1, lang2, lang3, data_type)] = bleu

    def run_all_evals(self, epoch):
        """
        Run all evaluations.
        """
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():

            for data_type in ['dev', 'test']:
                self.eval_content(data_type, scores)
                # if 'test ground truth'/'test para'

        return scores


def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(ref) and os.path.isfile(hyp)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)
    result = p.communicate()[0].decode("utf-8")
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1


def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()
    bos_index = params.bos_index

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == bos_index).sum() == bs
    assert (batch == params.eos_index).sum() == bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences

def convert_to_attr(attr, params):
    """
    Convert batch of attributes to a list of attribute names.
    """
    attr_list = [list(l) for l in attr]
    return [", ".join([params.id2style[x.item()] for x in l]) for l in attr_list]
