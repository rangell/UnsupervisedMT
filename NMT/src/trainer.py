# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
from logging import getLogger
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .utils import reverse_sentences, clip_parameters, sample_style
from .utils import get_optimizer, parse_lambda_config, update_lambdas
from .model import build_mt_model
from .model.feature_extractor import IPOT
from .multiprocessing_event_loop import MultiprocessingEventLoop
from .test import test_sharing


from IPython import embed


logger = getLogger()


class TrainerMT(MultiprocessingEventLoop):

    VALIDATION_METRICS = []

    def __init__(self, encoder, decoder, discriminator, feat_extr, lm, data, params):
        """
        Initialize trainer.
        """
        super().__init__(device_ids=tuple(range(params.otf_num_processes)))
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.feat_extr = feat_extr
        self.lm = lm
        self.data = data
        self.params = params

        # initialization for on-the-fly generation/training
        self.otf_start_multiprocessing()
        self.subprocess_ids = list(range(self.num_replicas))
        self.ranks = {}

        # define encoder parameters (the ones shared with the
        # decoder are optimized by the decoder optimizer)
        enc_params = list(encoder.parameters())
        feat_extr_params = list(feat_extr.parameters())
        assert enc_params[0].size() == (params.n_words, params.emb_dim)
        assert enc_params[1].size() == (params.n_styles+1, params.emb_dim)
        assert feat_extr_params[0].size() == (params.n_words, params.emb_dim)
        assert feat_extr_params[1].size() == (params.n_styles+1, params.emb_dim)

        if self.params.share_encdec_emb:
            to_ignore = 2   # ignore word embeddings and style embeddings
            enc_params = enc_params[to_ignore:]
            feat_extr_params = feat_extr_params[to_ignore:]

        # optimizers
        if params.dec_optimizer == 'enc_optimizer':
            params.dec_optimizer = params.enc_optimizer

        self.enc_optimizer = None
        self.dec_optimizer = None
        self.dis_optimizer = None
        self.feat_extr_optimizer = None
        self.lm_optimizer = None
        
        if len(enc_params) > 0:
            self.enc_optimizer = get_optimizer(enc_params,
                                               params.enc_optimizer)
        self.dec_optimizer = get_optimizer(decoder.parameters(),
                                           params.dec_optimizer)
        if discriminator is not None:
            self.dis_optimizer = get_optimizer(discriminator.parameters(),
                                               params.dis_optimizer)
        if feat_extr is not None:
            self.feat_extr_optimizer = get_optimizer(feat_extr_params,
                                                     params.feat_extr_optimizer)
        if lm is not None:
            self.lm_optimizer = get_optimizer(lm.parameters(),
                                              params.enc_optimizer)

        # models / optimizers
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'feat_extr': (self.feat_extr, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }

        # define validation metrics / stopping criterion used for early stopping
        logger.info("Stopping criterion: %s" % params.stopping_criterion)
        if params.stopping_criterion == '':
            for data_type in ['valid', 'test']:
                self.VALIDATION_METRICS.append('self_bleu_%s' % (data_type))
            self.stopping_criterion = None
            self.best_stopping_criterion = None
        else:
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            self.stopping_criterion = split[0]
            self.best_stopping_criterion = -1e12
            assert len(self.VALIDATION_METRICS) == 0
            self.VALIDATION_METRICS.append(self.stopping_criterion)

        # training variables
        self.best_metrics = {metric: -1e12 for metric in self.VALIDATION_METRICS}
        self.epoch = 0
        self.n_total_iter = 0
        self.freeze_enc_emb = self.params.freeze_enc_emb
        self.freeze_dec_emb = self.params.freeze_dec_emb

        # training statistics
        self.n_iter = 0
        self.n_sentences = 0
        self.stats = {
            'dis_costs': [],
            'processed_s': 0,
            'processed_w': 0,
            'xe_ae_costs': [],
            'xe_bt_costs': [],
            'ppl_ae_costs': [],
            'ppl_bt_costs': [],
            'ipot_fe_costs': [],
            'ipot_adv_costs': [],
            'lme_costs': [],
            'lmd_costs': [],
            'lmer_costs': [],
            'enc_norms': []
        }

        self.last_time = time.time()
        self.gen_time = 0

        # data iterators
        self.iterators = {}

        # initialize BPE subwords
        self.init_bpe()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params, 'lambda_xe_ae')
        parse_lambda_config(params, 'lambda_ipot_ae')
        parse_lambda_config(params, 'lambda_xe_otf_bt')
        parse_lambda_config(params, 'lambda_ipot_otf_bt')
        parse_lambda_config(params, 'lambda_lm')
        parse_lambda_config(params, 'lambda_dis')
        parse_lambda_config(params, 'lambda_feat_extr')
        parse_lambda_config(params, 'lambda_adv')

    def init_bpe(self):
        """
        Index BPE words.
        """
        self.bpe_end = []
        dico = self.data['dico']
        self.bpe_end.append(np.array([not dico[i].endswith('@@') for i in range(len(dico))]))

    def get_iterator(self, iter_name):
        """
        Create a new iterator for a dataset.
        """
        dataset = self.data['splits']['train']
        iterator = dataset.get_iterator(shuffle=True,
                group_by_size=self.params.group_by_size)()
        self.iterators[iter_name] = iterator
        return iterator

    def get_batch(self, iter_name):
        """
        Return a batch of sentences from a dataset.
        """
        iterator = self.iterators.get(iter_name, None)
        if iterator is None:
            iterator = self.get_iterator(iter_name)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(iter_name)
            batch = next(iterator)
        return batch

    def word_shuffle(self, x, l, lang_id):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.size(0) - 1, x.size(1)))
        noise[0] = -1  # do not move start sentence symbol

        # be sure to shuffle entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        assert self.params.word_shuffle > 1
        x2 = x.clone()
        for i in range(l.size(0)):
            # generate a random permutation
            scores = word_idx[:l[i] - 1, i] + noise[word_idx[:l[i] - 1, i], i]
            scores += 1e-6 * np.arange(l[i] - 1)  # ensure no reordering inside a word
            permutation = scores.argsort()
            # shuffle words
            x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][torch.from_numpy(permutation)])
        return x2, l

    def word_dropout(self, x, l, lang_id):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        # be sure to drop entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        lengths = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[word_idx[j, i], i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(self.params.eos_index)
            assert len(new_s) >= 3 and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input
        l2 = torch.LongTensor(lengths)
        x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        for i in range(l2.size(0)):
            x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l2

    def word_blank(self, x, l, lang_id):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        bos_index = self.params.bos_index[lang_id]
        assert (x[0] == bos_index).sum() == l.size(0)
        keep = np.random.rand(x.size(0) - 1, x.size(1)) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        # be sure to blank entire words
        bpe_end = self.bpe_end[lang_id][x]
        word_idx = bpe_end[::-1].cumsum(0)[::-1]
        word_idx = word_idx.max(0)[None, :] - word_idx

        sentences = []
        for i in range(l.size(0)):
            assert x[l[i] - 1, i] == self.params.eos_index
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[word_idx[j, i], i] else self.params.blank_index for j, w in enumerate(words)]
            new_s.append(self.params.eos_index)
            assert len(new_s) == l[i] and new_s[0] == bos_index and new_s[-1] == self.params.eos_index
            sentences.append(new_s)
        # re-construct input
        x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        for i in range(l.size(0)):
            x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
        return x2, l

    def add_noise(self, words, lengths, lang_id):
        """
        Add noise to the encoder input.
        """
        words, lengths = self.word_shuffle(words, lengths, lang_id)
        words, lengths = self.word_dropout(words, lengths, lang_id)
        words, lengths = self.word_blank(words, lengths, lang_id)
        return words, lengths

    def zero_grad(self, models):
        """
        Zero gradients.
        """
        if type(models) is not list:
            models = [models]
        models = [self.model_opt[name] for name in models]
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.zero_grad()

    def update_params(self, models):
        """
        Update parameters.
        """
        if type(models) is not list:
            models = [models]
        # don't update encoder when it's frozen
        models = [self.model_opt[name] for name in models]
        # clip gradients
        for model, _ in models:
            clip_grad_norm_(model.parameters(), self.params.clip_grad_norm)

        # optimizer
        for _, optimizer in models:
            if optimizer is not None:
                optimizer.step()

    def get_lrs(self, models):
        """
        Get current optimizer learning rates.
        """
        if type(models) is not list:
            models = [models]
        lrs = {}
        for name in models:
            optimizer = self.model_opt[name][1]
            if optimizer is not None:
                lrs[name] = optimizer.param_groups[0]['lr']
        return lrs

    def discriminator_step(self):
        """
        Train the discriminator on the latent space.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()

        # train on monolingual data only
        if self.params.n_mono == 0:
            raise Exception("No data to train the discriminator!")

        # batch / encode
        encoded = []
        for lang_id, lang in enumerate(self.params.langs):
            sent1, len1 = self.get_batch('dis', lang, None)
            with torch.no_grad():
                encoded.append(self.encoder(sent1.cuda(), len1, lang_id))

        # discriminator
        dis_inputs = [x.dis_input.view(-1, x.dis_input.size(-1)) for x in encoded]
        ntokens = [dis_input.size(0) for dis_input in dis_inputs]
        encoded = torch.cat(dis_inputs, 0)
        predictions = self.discriminator(encoded.data)

        # loss
        self.dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)])
        self.dis_target = self.dis_target.contiguous().long().cuda()
        y = self.dis_target

        loss = F.cross_entropy(predictions, y)
        self.stats['dis_costs'].append(loss.item())

        # optimizer
        self.zero_grad('dis')
        loss.backward()
        self.update_params('dis')
        clip_parameters(self.discriminator, self.params.dis_clip)

    def lm_step(self, lang):
        """
        Language model training.
        """
        assert self.params.lambda_lm > 0
        assert lang in self.params.langs
        assert self.lm.use_lm_enc or self.lm.use_lm_dec
        lang_id = self.params.lang2id[lang]
        self.lm.train()

        loss_fn = self.decoder.loss_fn[lang_id]
        n_words = self.params.n_words[lang_id]

        # batch
        sent1, len1 = self.get_batch('lm', lang, None)
        sent1 = sent1.cuda()
        if self.lm.use_lm_enc_rev:
            sent1_rev = reverse_sentences(sent1, len1)

        # forward
        if self.lm.use_lm_enc:
            scores_enc = self.lm(sent1[:-1], len1 - 1, lang_id, True, False)
        if self.lm.use_lm_dec:
            scores_dec = self.lm(sent1[:-1], len1 - 1, lang_id, False, False)
        if self.lm.use_lm_enc_rev:
            scores_enc_rev = self.lm(sent1_rev[:-1], len1 - 1, lang_id, True, True)

        # loss
        loss = 0
        if self.lm.use_lm_enc:
            loss_enc = loss_fn(scores_enc.view(-1, n_words), sent1[1:].view(-1))
            self.stats['lme_costs_%s' % lang].append(loss_enc.item())
            loss += loss_enc
        if self.lm.use_lm_dec:
            loss_dec = loss_fn(scores_dec.view(-1, n_words), sent1[1:].view(-1))
            self.stats['lmd_costs_%s' % lang].append(loss_dec.item())
            loss += loss_dec
        if self.lm.use_lm_enc_rev:
            loss_enc_rev = loss_fn(scores_enc_rev.view(-1, n_words), sent1_rev[1:].view(-1))
            self.stats['lmer_costs_%s' % lang].append(loss_enc_rev.item())
            loss += loss_enc_rev
        loss = self.params.lambda_lm * loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['lm'])
        loss.backward()
        self.update_params(['lm'])

        # number of processed sentences / words
        self.stats['processed_s'] += len1.size(0)
        self.stats['processed_w'] += len1.sum()

    def gen_st_batch(self):
        params = self.params
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            sent1, len1, attr1 = self.get_batch('gen_st')

            encoded = self.encoder(sent1.cuda(), len1, attr1.cuda())
            max_len = int(1.5 * len1.max() + 10)

            attr2 = sample_style(params, attr1)
            attr2 = attr2.cuda()

            if params.otf_sample == -1:
                sent2, len2, _ = self.decoder.generate(encoded, attr2, max_len=max_len)
            else:
                sent2, len2, _ = self.decoder.generate(encoded, attr2, max_len=max_len,
                                                       sample=True, temperature=params.otf_sample)

        return dict([
                 ('sent1', sent1), ('len1', len1), ('attr1', attr1),
                 ('sent2', sent2), ('len2', len2), ('attr2', attr2),
            ])
    
    def enc_dec_ae_step(self, lambda_xe):
        # essentially enc_dec_step(...) called in AE mode
        params = self.params
        loss_fn = self.decoder.loss_fn[0]
        n_words = params.n_words
        self.encoder.train()
        self.decoder.train()
        if self.discriminator is not None:
            self.discriminator.eval()

        # get batch
        sent_, len_, attr_ = self.get_batch('encdec_ae')
        sent_ = sent_.cuda()
        attr_ = attr_.cuda()

        # encode
        encoded = self.encoder(sent_, len_, attr_)
        self.stats['enc_norms'].append(encoded.dis_input.data.norm(2, 1).mean().item())

        # decode
        scores = self.decoder(encoded, sent_[:-1], attr_)
        xe_loss = loss_fn(scores.view(-1, n_words), sent_[1:].view(-1))

        self.stats['xe_ae_costs'].append(xe_loss.item())
        self.stats['ppl_ae_costs'].append(torch.exp(xe_loss).item())

        # discriminator feedback loss
        if params.lambda_dis:
            predictions = self.discriminator(encoded.dis_input.view(-1, encoded.dis_input.size(-1)))
            fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = (fake_y + lang1_id) % params.n_langs
            fake_y = fake_y.cuda()
            dis_loss = F.cross_entropy(predictions, fake_y)

        # total loss
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss
        if params.lambda_dis:
            loss = loss + params.lambda_dis * dis_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['enc', 'dec'])
        loss.backward()
        self.update_params(['enc', 'dec'])

        # number of processed sentences / words
        self.stats['processed_s'] += len_.size(0)
        self.stats['processed_w'] += len_.sum()

    def enc_dec_bt_step(self, batch, lambda_xe):
        # essentially otf_bt(...) called in AE mode (yes AE mode, not a typo)
        params = self.params
        loss_fn = self.decoder.loss_fn[0]
        self.encoder.train()
        self.decoder.train()

        sent1 = batch['sent1'].cuda()
        len1 = batch['len1']
        attr1 = batch['attr1'].cuda()

        sent2 = batch['sent2'].cuda()
        len2 = batch['len2']
        attr2 = batch['attr2'].cuda()

        # encode previously generated sentence
        encoded = self.encoder(sent2, len2, attr2)

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent1[:-1], attr1)
        xe_loss = loss_fn(scores.view(-1, params.n_words), sent1[1:].view(-1))
        self.stats['xe_bt_costs'].append(xe_loss.item())
        self.stats['ppl_bt_costs'].append(torch.exp(xe_loss).item())
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        assert params.otf_update_enc or params.otf_update_dec
        to_update = []
        if params.otf_update_enc:
            to_update.append('enc')
        if params.otf_update_dec:
            to_update.append('dec')
        self.zero_grad(to_update)
        loss.backward()
        self.update_params(to_update)

        # number of processed sentences / words
        self.stats['processed_s'] += batch['len1'].size(0)
        self.stats['processed_w'] += batch['len1'].sum()

    def feat_extr_step(self, batch, lambda_feat_extr):
        assert lambda_feat_extr > 0

        if batch['len2'].min() < 3:
            logger.warning("Missed feat_extr step")
            return

        real_reps = self.feat_extr(batch['sent1'].cuda(), batch['len1'],
                                   batch['attr1'].cuda())
        fake_reps = self.feat_extr(batch['sent2'].cuda(), batch['len2'],
                                   batch['attr2'].cuda())
        # TODO: concat negative examples (i.e. (sent2, len2, attr1), (sent1, len1, attr2), etc.)

        loss = -IPOT(real_reps, fake_reps)
        self.stats['ipot_fe_costs'].append(loss.item())
        loss *= lambda_feat_extr

        self.zero_grad(['feat_extr'])
        loss.backward()
        self.update_params(['feat_extr'])

    def enc_dec_adv_step(self, batch, lambda_adv):
        assert lambda_adv > 0

        if batch['len2'].min() < 3:
            logger.warning("Missed adv step")
            return

        params = self.params
        self.encoder.train()
        self.decoder.train()
        self.feat_extr.eval()

        sent1 = batch['sent1'].cuda()
        len1 = batch['len1']
        attr1 = batch['attr1'].cuda()

        sent2 = batch['sent2'].cuda()
        len2 = batch['len2']
        attr2 = batch['attr2'].cuda()

        # encode previously generated sentence
        encoded = self.encoder(sent1, len1, attr1)

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent2[:-1], attr2)
        smooth_scores = self.decoder.smooth(scores)
        soft_sent2 = torch.matmul(smooth_scores, self.decoder.embeddings.weight)
        soft_sent2 = torch.cat([torch.zeros(1, soft_sent2.size(1),
                                            soft_sent2.size(2)).cuda(),
                                soft_sent2])

        real_reps = self.feat_extr(sent1, len1, attr1)
        fake_reps = self.feat_extr(soft_sent2, len2, attr2, soft=True)
        
        loss = IPOT(real_reps, fake_reps)
        self.stats['ipot_adv_costs'].append(loss.item())
        loss *= lambda_adv

        # optimizer
        self.zero_grad(['enc', 'dec'])
        loss.backward()
        self.update_params(['enc', 'dec'])

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()

    def enc_dec_step(self, lang1, lang2, lambda_xe, back=False):
        """
        Source / target autoencoder training (parallel data):
            - encoders / decoders training on cross-entropy
            - encoders training on discriminator feedback
            - encoders training on L2 loss (seq2seq only, not for attention)
        """
        params = self.params
        assert lang1 in params.langs and lang2 in params.langs
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        loss_fn = self.decoder.loss_fn[lang2_id]
        n_words = params.n_words[lang2_id]
        self.encoder.train()
        self.decoder.train()
        if self.discriminator is not None:
            self.discriminator.eval()

        # batch
        if back:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2, back=True)
        elif lang1 == lang2:
            sent1, len1 = self.get_batch('encdec', lang1, None)
            sent2, len2 = sent1, len1
        else:
            (sent1, len1), (sent2, len2) = self.get_batch('encdec', lang1, lang2)

        # prepare the encoder / decoder inputs
        if lang1 == lang2:
            sent1, len1 = self.add_noise(sent1, len1, lang1_id)
        sent1, sent2 = sent1.cuda(), sent2.cuda()

        # encoded states
        encoded = self.encoder(sent1, len1, lang1_id)
        self.stats['enc_norms_%s' % lang1].append(encoded.dis_input.data.norm(2, 1).mean().item())

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent2[:-1], lang2_id)
        xe_loss = loss_fn(scores.view(-1, n_words), sent2[1:].view(-1))
        if back:
            self.stats['xe_costs_bt_%s_%s' % (lang1, lang2)].append(xe_loss.item())
        else:
            self.stats['xe_costs_%s_%s' % (lang1, lang2)].append(xe_loss.item())

        # discriminator feedback loss
        if params.lambda_dis:
            predictions = self.discriminator(encoded.dis_input.view(-1, encoded.dis_input.size(-1)))
            fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = (fake_y + lang1_id) % params.n_langs
            fake_y = fake_y.cuda()
            dis_loss = F.cross_entropy(predictions, fake_y)

        # total loss
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss
        if params.lambda_dis:
            loss = loss + params.lambda_dis * dis_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        self.zero_grad(['enc', 'dec'])
        loss.backward()
        self.update_params(['enc', 'dec'])

        # number of processed sentences / words
        self.stats['processed_s'] += len2.size(0)
        self.stats['processed_w'] += len2.sum()

    def otf_start_multiprocessing(self):
        logger.info("Starting subprocesses for OTF generation ...")

        # initialize subprocesses
        for rank in range(self.num_replicas):
            self.call_async(rank, '_async_otf_init', params=self.params)

    def _async_otf_init(self, rank, device_id, params):
        # build model on subprocess

        from copy import deepcopy
        params = deepcopy(params)
        self.params = params
        self.params.cpu_thread = True
        self.data = None  # do not load data in the CPU threads
        self.iterators = {}
        self.encoder, self.decoder, _, _, _ = build_mt_model(self.params, self.data, cuda=False)

    def otf_sync_params(self, iter_name):
        logger.info("Syncing encoder and decoder params for {} iterator ...".format(iter_name))

        def get_flat_params(module):
            return torch._utils._flatten_dense_tensors(
                [p.data for p in module.parameters()])

        encoder_params = get_flat_params(self.encoder).cpu().share_memory_()
        decoder_params = get_flat_params(self.decoder).cpu().share_memory_()

        for rank in self.ranks[iter_name]:
            self.call_async(rank, '_async_otf_sync_params', encoder_params=encoder_params,
                            decoder_params=decoder_params)

    def _async_otf_sync_params(self, rank, device_id, encoder_params, decoder_params):

        def set_flat_params(module, flat):
            params = [p.data for p in module.parameters()]
            for p, f in zip(params, torch._utils._unflatten_dense_tensors(flat, params)):
                p.copy_(f)

        # copy parameters back into modules
        set_flat_params(self.encoder, encoder_params)
        set_flat_params(self.decoder, decoder_params)

    def otf_before_gen(self, iter_name=None, num_proc=None):
        num_proc_left = len(self.subprocess_ids)
        assert iter_name is not None
        assert iter_name not in self.ranks.keys()
        assert num_proc is not None
        assert num_proc_left > 0

        if num_proc == -1:
            self.ranks[iter_name] = self.subprocess_ids[:]
            self.subprocess_ids = []
        else:
            self.ranks[iter_name] = self.subprocess_ids[:num_proc]
            self.subprocess_ids = self.subprocess_ids[num_proc:]

        if num_proc > num_proc_left:
            logger.warning("Can only provide {}/{} requested processes".format(num_proc_left, num_proc))
        else:
            logger.info("Allocating {} processes to {} iterator".format(num_proc, iter_name))

    def otf_gen_async(self, iter_name=None):
        logger.info("Populating initial {} iterator cache ...".format(iter_name))
        cache = [
            self.call_async(rank=i, action='_async_otf_gen',
                            result_type='otf_gen', fetch_all=True,
                            batch=self.get_batch(iter_name))
            for i in self.ranks[iter_name]
        ]

        while True:
            results = cache[0].gen()
            for rank, _ in results:
                cache.pop(0)  # keep the cache a fixed size
                cache.append(
                    self.call_async(rank=rank, action='_async_otf_gen',
                                    result_type='otf_gen', fetch_all=True,
                                    batch=self.get_batch(iter_name))
                )
            for _, result in results:
                yield result[0]

    def _async_otf_gen(self, rank, device_id, batch):
        """
        On the fly back-translation (generation step).
        """
        params = self.params
        self.encoder.eval()
        self.decoder.eval()

        results = []

        with torch.no_grad():

            sent1, len1, attr1 = batch

            encoded = self.encoder(sent1, len1, attr1)
            max_len = int(1.5 * len1.max() + 10)

            attr2 = sample_style(params, attr1)

            if params.otf_sample == -1:
                sent2, len2, _ = self.decoder.generate(encoded, attr2, max_len=max_len)
            else:
                sent2, len2, _ = self.decoder.generate(encoded, attr2, max_len=max_len,
                                                       sample=True, temperature=params.otf_sample)

            assert not any(x.is_cuda for x in [sent1, sent2])
            results.append(dict([
                 ('sent1', sent1), ('len1', len1), ('attr1', attr1),
                 ('sent2', sent2), ('len2', len2), ('attr2', attr2),
            ]))

        return (rank, results)

    def otf_bt(self, batch, lambda_xe, backprop_temperature):
        """
        On the fly back-translation.
        """
        params = self.params
        lang1, sent1, len1 = batch['lang1'], batch['sent1'], batch['len1']
        lang2, sent2, len2 = batch['lang2'], batch['sent2'], batch['len2']
        lang3, sent3, len3 = batch['lang3'], batch['sent3'], batch['len3']
        if lambda_xe == 0:
            logger.warning("Unused generated CPU batch for direction %s-%s-%s!" % (lang1, lang2, lang3))
            return
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        lang3_id = params.lang2id[lang3]
        direction = (lang1, lang2, lang3)
        assert direction in params.pivo_directions
        loss_fn = self.decoder.loss_fn[lang3_id]
        n_words2 = params.n_words[lang2_id]
        n_words3 = params.n_words[lang3_id]
        self.encoder.train()
        self.decoder.train()

        # prepare batch
        sent1, sent2, sent3 = sent1.cuda(), sent2.cuda(), sent3.cuda()
        bs = sent1.size(1)

        if backprop_temperature == -1:
            # lang2 -> lang3
            encoded = self.encoder(sent2, len2, lang_id=lang2_id)
        else:
            # lang1 -> lang2
            encoded = self.encoder(sent1, len1, lang_id=lang1_id)
            scores = self.decoder(encoded, sent2[:-1], lang_id=lang2_id)
            assert scores.size() == (len2.max() - 1, bs, n_words2)

            # lang2 -> lang3
            bos = torch.cuda.FloatTensor(1, bs, n_words2).zero_()
            bos[0, :, params.bos_index[lang2_id]] = 1
            sent2_input = torch.cat([bos, F.softmax(scores / backprop_temperature, -1)], 0)
            encoded = self.encoder(sent2_input, len2, lang_id=lang2_id)

        # cross-entropy scores / loss
        scores = self.decoder(encoded, sent3[:-1], lang_id=lang3_id)
        xe_loss = loss_fn(scores.view(-1, n_words3), sent3[1:].view(-1))
        self.stats['xe_costs_%s_%s_%s' % direction].append(xe_loss.item())
        assert lambda_xe > 0
        loss = lambda_xe * xe_loss

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # optimizer
        assert params.otf_update_enc or params.otf_update_dec
        to_update = []
        if params.otf_update_enc:
            to_update.append('enc')
        if params.otf_update_dec:
            to_update.append('dec')
        self.zero_grad(to_update)
        loss.backward()
        self.update_params(to_update)

        # number of processed sentences / words
        self.stats['processed_s'] += len3.size(0)
        self.stats['processed_w'] += len3.sum()

    def iter(self, n_batches):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.n_sentences += n_batches * self.params.batch_size
        self.print_stats()
        update_lambdas(self.params, self.n_total_iter)

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # average loss / statistics
        if self.n_iter % 50 == 0:
            mean_loss = [
                ('DIS', 'dis_costs'),
                ('XE-AE', 'xe_ae_costs'),
                ('XE-BT', 'xe_bt_costs'),
                ('PPL-AE', 'ppl_ae_costs'),
                ('PPL-BT', 'ppl_bt_costs'),
                ('IPOT-FE', 'ipot_fe_costs'),
                ('IPOT-ADV', 'ipot_adv_costs'),
                ('LME', 'lme_costs'),
                ('LMD', 'lmd_costs'),
                ('LMER', 'lmer_costs'),
                ('ENC-L2', 'enc_norms')
            ]

            s_iter = "%7i - " % self.n_iter
            s_stat = ' || '.join(['{}: {:7.4f}'.format(k, np.mean(self.stats[l]))
                                 for k, l in mean_loss if len(self.stats[l]) > 0])
            for _, l in mean_loss:
                del self.stats[l][:]

            # processing speed
            new_time = time.time()
            diff = new_time - self.last_time
            s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(self.stats['processed_s'] * 1.0 / diff,
                                                                   self.stats['processed_w'] * 1.0 / diff)
            self.stats['processed_s'] = 0
            self.stats['processed_w'] = 0
            self.last_time = new_time

            lrs = self.get_lrs(['enc', 'dec'])
            s_lr = " - LR " + ",".join("{}={:.4e}".format(k, lr) for k, lr in lrs.items())

            # generation time
            s_time = " - Sentences generation time: % .2fs (%.2f%%)" % (self.gen_time, 100. * self.gen_time / diff)
            self.gen_time = 0

            # log speed + stats
            logger.info(s_iter + s_speed + s_stat + s_lr + s_time)

    def save_model(self, name):
        """
        Save the model.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info('Saving model to %s ...' % path)
        torch.save({
            'enc': self.encoder,
            'dec': self.decoder,
            'dis': self.discriminator,
            'lm': self.lm,
        }, path)

    def save_checkpoint(self):
        """
        Checkpoint the experiment.
        """
        checkpoint_data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'discriminator': self.discriminator,
            'lm': self.lm,
            'enc_optimizer': self.enc_optimizer,
            'dec_optimizer': self.dec_optimizer,
            'dis_optimizer': self.dis_optimizer,
            'lm_optimizer': self.lm_optimizer,
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        logger.info("Saving checkpoint to %s ..." % checkpoint_path)
        torch.save(checkpoint_data, checkpoint_path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        # reload checkpoint
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint.pth')
        if not os.path.isfile(checkpoint_path):
            return
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        checkpoint_data = torch.load(checkpoint_path)
        self.encoder = checkpoint_data['encoder']
        self.decoder = checkpoint_data['decoder']
        self.discriminator = checkpoint_data['discriminator']
        self.lm = checkpoint_data['lm']
        self.enc_optimizer = checkpoint_data['enc_optimizer']
        self.dec_optimizer = checkpoint_data['dec_optimizer']
        self.dis_optimizer = checkpoint_data['dis_optimizer']
        self.lm_optimizer = checkpoint_data['lm_optimizer']
        self.epoch = checkpoint_data['epoch']
        self.n_total_iter = checkpoint_data['n_total_iter']
        self.best_metrics = checkpoint_data['best_metrics']
        self.best_stopping_criterion = checkpoint_data['best_stopping_criterion']
        self.model_opt = {
            'enc': (self.encoder, self.enc_optimizer),
            'dec': (self.decoder, self.dec_optimizer),
            'dis': (self.discriminator, self.dis_optimizer),
            'lm': (self.lm, self.lm_optimizer),
        }
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)

    def test_sharing(self):
        """
        Test to check that parameters are shared correctly.
        """
        test_sharing(self.encoder, self.decoder, self.lm, self.params)
        logger.info("Test: Parameters are shared correctly.")

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        for metric in self.VALIDATION_METRICS:
            if scores[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_model('best-%s' % metric)

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if self.params.save_periodic and self.epoch % 20 == 0 and self.epoch > 0:
            self.save_model('periodic-%i' % self.epoch)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None:
            assert self.stopping_criterion in scores
            if scores[self.stopping_criterion] > self.best_stopping_criterion:
                self.best_stopping_criterion = scores[self.stopping_criterion]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            if scores[self.stopping_criterion] < self.best_stopping_criterion:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                exit()
        self.epoch += 1
        self.save_checkpoint()
