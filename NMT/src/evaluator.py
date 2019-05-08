# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import random
import pickle
from collections import OrderedDict
from logging import getLogger
import math
import numpy as np
import spacy
import torch
from torch import nn

import sys
from .utils import restore_segmentation, sample_style
from tqdm import tqdm

import fastText
from scipy.spatial.distance import cosine
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu

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

        # load necessary objects for evaluation
        self._load_eval_params()

    def _load_eval_params(self):
        params = self.params

        # token embedding idf weighted vectors
        self.token_idf_vecs = pickle.load(open(params.idf_vecs_filename, 'rb'))

        # pretrained language model
        sys.path.insert(0, './src/pretrain_lm')
        with open(params.lang_model_filename, 'rb') as f:
            self.lang_model = pickle.load(f)
            self.lang_model.model.rnn.flatten_parameters() # speeds up forward pass

        # pretrained style classifier
        self.style_clfs = {}
        for attr_name in params.attr_names:
            clf_path = params.clf_paths[attr_name]
            self.style_clfs[attr_name] = fastText.load_model(clf_path)

    def get_iterator(self, data_type):
        """
        Create a new iterator for a dataset.
        """
        assert data_type in ['dev', 'test', 'test_para']
        dataset = self.data['splits'][data_type]
        dataset.batch_size = 32
        for batch in dataset.get_iterator(shuffle=False, group_by_size=False)():
            yield batch

    def generate_ref_and_hyp(self, data_type, scores):
        logger.info("Generating for %s ..." % (data_type))

        assert data_type in ['dev', 'test']
        self.encoder.eval()
        self.decoder.eval()
        params = self.params

        # hypothesis
        ref_txt = []      # Coming from txt
        ref_attr = []     # Going to txt
        hyp_txt = []      # Coming from attr
        hyp_attr = []     # Going to attr

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
            ref_txt.extend(convert_to_text(sent1, len1, self.dico, params))
            hyp_txt.extend(convert_to_text(sent2_, len2_, self.dico, params))

            # convert attr id labels to text versions
            ref_attr.extend([list(x) for x in attr1.cpu().numpy()])
            hyp_attr.extend([list(x) for x in attr2_.cpu().numpy()])

        # restore from bpe
        ref_txt = [s.replace('@@ ', '') for s in ref_txt]
        hyp_txt = [s.replace('@@ ', '') for s in hyp_txt]

        # export sentences to hypothesis file / restore bpe segmentation (shouldn't be one)
        hyp_name = 'hyp{0}.{1}.txt'.format(scores['epoch'], data_type)
        hyp_path = os.path.join(params.dump_path, hyp_name)

        with open(hyp_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hyp_txt) + '\n')

        return ref_txt, ref_attr, hyp_txt, hyp_attr

    def eval_style_transfer(self,
                            data_type,
                            ref_txt,
                            ref_attr,
                            hyp_txt,
                            hyp_attr,
                            scores):

        logger.info("Evaluating %s ..." % (data_type))
        params = self.params
        
        # obtain predictions
        pred_attr = []
        for txt_sent in hyp_txt:
            preds = []
            for attr_name in params.attr_names:
                pred_label = self.style_clfs[attr_name].predict(txt_sent)[0][0]
                pred_id = params.style2id[pred_label.replace('__label__', '')]
                preds.append(pred_id)
            pred_attr.append(preds)

        # Print out some examples here
        num_samples = 10
        sample_indices = random.sample(range(len(ref_txt)), num_samples)
        logger.info("Sampling random examples...\n")
        for i in sample_indices:
            logger.info("Example {}:".format(i) + "\noriginal attributes: " 
                        + " ".join([params.id2style[x] for x in ref_attr[i]])
                        + "\noriginal text: " + ref_txt[i]
                        + "\n\nintended attributes: "
                        + " ".join([params.id2style[x] for x in hyp_attr[i]])
                        + "\npredicted attributes: "
                        + " ".join([params.id2style[x] for x in pred_attr[i]])
                        + "\ntransferred text: " + hyp_txt[i] + "\n")

        orig_attr = np.asarray(ref_attr)
        tgt_attr = np.asarray(hyp_attr)
        pred_attr = np.asarray(pred_attr)

        tsf_acc = list(np.mean(tgt_attr == pred_attr, axis=0))
        pred_orig_style = np.mean(np.all(orig_attr == pred_attr, axis=1))

        for attr_name, acc in zip(params.attr_names, tsf_acc):
            key = 'tsf_accuracy_%s_%s' % (attr_name, data_type)
            logger.info('%s: %f' % (key, acc))
            scores[key] = acc
        logger.info('pred_orig_style_%s: %f' % (data_type, pred_orig_style))
        scores['pred_orig_style_%s' % data_type] = pred_orig_style

        # semantic similarity
        sim = self.compute_sim(ref_txt, hyp_txt)
        logger.info("sim_%s : %f " % (data_type, sim))
        scores['sim_%s' % (data_type)] = sim

        ## meteor score
        #met = self.compute_met(ref_txt, hyp_txt)
        #logger.info("met_%s : %f " % (data_type, met))
        #scores['met_%s' % (data_type)] = met

        # bleu score
        bleu = corpus_bleu([[s.split()] for s in ref_txt], [s.split() for s in hyp_txt])
        bleu *= 100
        logger.info("bleu_%s : %f " % (data_type, bleu))
        scores['bleu_%s' % (data_type)] = bleu

        ## pre-trained perplexity
        #ppl = math.exp(self.lang_model.evaluate(hyp_txt))
        spacy_lm = spacy.load('en_core_web_lg')
        perplexities = []
        for sent in hyp_txt:
            tokens = spacy_lm(sent)
            perplexities.append(math.pow(2, -np.mean([t.prob for t in tokens])))
        ppl = np.mean(perplexities)
        logger.info("ppl_%s : %f " % (data_type, ppl))
        scores['ppl_%s' % (data_type)] = ppl

    def compute_sim(self, original, transferred):
        token_idf_vecs = self.token_idf_vecs

        def get_vec(token):
            try:
                return token_idf_vecs[token]
            except:
                return np.zeros(token_idf_vecs[list(token_idf_vecs.keys())[0]].shape)

        cos_sims = []
        for org_sent, tsf_sent in zip(original, transferred):
            org_toks = org_sent.split(' ')
            tsf_toks = tsf_sent.split(' ')
            org_vecs = np.asarray([get_vec(token) for token in org_toks])
            tsf_vecs = np.asarray([get_vec(token) for token in tsf_toks])
            org_vec = np.sum(org_vecs, axis=0)
            tsf_vec = np.sum(tsf_vecs, axis=0)
            cos_sim = 1 - cosine(org_vec, tsf_vec)
            cos_sims.append(cos_sim)

        return np.mean(cos_sims)

    def compute_met(self, original, transferred):
        met_scores = []
        for org_sent, tsf_sent in zip(original, transferred):
            met_scores.append(meteor_score(org_sent, tsf_sent))
        return np.mean(met_scores)

    def load_file_into_list(self, filename):
        lines = []
        with open(filename, 'r') as f:
            for line in f:
                lines.append(line.strip())
        return lines

    def load_txt_and_attr_from_iterator(self, data_type):
        params = self.params

        txt, attr = [], []
        for batch in self.get_iterator(data_type):
            # batch
            sent_, len_, attr_ = batch

            # convert to text
            txt.extend(convert_to_text(sent_, len_, self.dico, params))

            # convert attr id labels to text versions
            attr.extend([list(x) for x in attr_.cpu().numpy()])

        return txt, attr

    def run_external_eval(self, ref_txt_filename, ref_attr_filename,
                          hyp_txt_filename, hyp_attr_filename):
        params = self.params
        scores = OrderedDict({'epoch': -1})
        data_type = 'test'


        ref_txt = self.load_file_into_list(ref_txt_filename)
        ref_attr = [[params.style2id[x]]
                        for x in self.load_file_into_list(ref_attr_filename)]
        hyp_txt = self.load_file_into_list(hyp_txt_filename)
        hyp_attr = [[params.style2id[x]]
                        for x in self.load_file_into_list(hyp_attr_filename)]

        self.eval_style_transfer(data_type, ref_txt, ref_attr,
                                 hyp_txt, hyp_attr, scores)

        #bleu = eval_moses_bleu(ref_txt_filename, hyp_txt_filename)
        #logger.info("BLEU : %f" % (bleu))

    def run_all_evals(self, epoch):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': epoch})

        with torch.no_grad():

            for data_type in ['dev', 'test']:
                ref_txt, ref_attr, hyp_txt, hyp_attr = \
                        self.generate_ref_and_hyp(data_type, scores)
                self.eval_style_transfer(data_type, ref_txt, ref_attr,
                                         hyp_txt, hyp_attr, scores)

                if data_type == 'test' and params.test_para:
                    data_type = 'test_para'
                    ref_txt, ref_attr = self.load_txt_and_attr_from_iterator(data_type)
                    self.eval_style_transfer(data_type, ref_txt, ref_attr,
                                             hyp_txt, hyp_attr, scores)

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
