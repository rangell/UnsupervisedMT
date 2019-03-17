# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from .model import LSTM_PARAMS, BILSTM_PARAMS


hashs = {}


def assert_equal(x, y):
    assert x.size() == y.size()
    assert (x.data - y.data).abs().sum() == 0


def hash_data(x):
    """
    Compute a hash on tensor data.
    """
    # TODO: make a better hash function (although this is good enough for embeddings)
    return (x.data.sum(), x.data.abs().sum())


def test_sharing(encoder, decoder, lm, params):
    """
    Test parameters sharing between the encoder,
    the decoder, and the language model.
    Test that frozen parameters are not being updated.
    """
    if not params.attention:  # TODO: implement this for seq2seq model
        return
    assert params.attention is True

    # frozen parameters
    if params.freeze_enc_emb:
        k = 'enc_emb'
        if k in hashs:
            assert hash_data(encoder.embeddings.weight) == hashs[k]
        else:
            hashs[k] = hash_data(encoder.embeddings.weight)
    if params.freeze_dec_emb:
        k = 'dec_emb'
        if k in hashs:
            assert hash_data(decoder.embeddings.weight) == hashs[k]
        else:
            hashs[k] = hash_data(decoder.embeddings.weight)

    #
    # encoder
    #
    # Nothing here

    #
    # decoder
    #
    # embedding layers
    if params.share_encdec_emb:
        assert_equal(encoder.embeddings.weight, decoder.embeddings.weight)

    #
    # language model
    #
    assert (not (lm is None) ^ (params.lm_after == params.lm_before == 0 and
                                params.lm_share_enc == params.lm_share_dec == 0 and
                                params.lm_share_emb is False and params.lm_share_proj is False))
    if lm is not None:
        assert lm.use_lm_enc or lm.use_lm_dec

        # encoder
        if lm.use_lm_enc:
            # embedding layers
            if params.lm_share_emb:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_enc.embeddings[i].weight, encoder.embeddings[i].weight)
            # LSTM layers
            for k in range(params.lm_share_enc):
                for i in range(params.n_langs):
                    for name in LSTM_PARAMS:
                        assert_equal(getattr(lm.lm_enc.lstm[i], name % k), getattr(encoder.lstm[i], name % k))

        # encoder - reverse direction
        if lm.use_lm_enc_rev:
            # embedding layers
            if params.lm_share_emb:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_enc_rev.embeddings[i].weight, encoder.embeddings[i].weight)
            # LSTM layers
            for k in range(params.lm_share_enc):
                for i in range(params.n_langs):
                    for name in LSTM_PARAMS:
                        _name = '%s_reverse' % name
                        assert_equal(getattr(lm.lm_enc_rev.lstm[i], name % k), getattr(encoder.lstm[i], _name % k))

        # decoder
        if lm.use_lm_dec:
            # embedding layers
            if params.lm_share_emb:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_dec.embeddings[i].weight, decoder.embeddings[i].weight)
            # LSTM layers
            for k in range(params.lm_share_dec):
                for i in range(params.n_langs):
                    for name in LSTM_PARAMS:
                        if k == 0:
                            assert_equal(getattr(lm.lm_dec.lstm[i], name % k), getattr(decoder.lstm1[i], name % k))
                        else:
                            assert_equal(getattr(lm.lm_dec.lstm[i], name % k), getattr(decoder.lstm2[i], name % (k - 1)))
            # projection layers
            if params.lm_share_proj:
                for i in range(params.n_langs):
                    assert_equal(lm.lm_dec.proj[i].weight, decoder.proj[i].weight)
                    assert_equal(lm.lm_dec.proj[i].bias, decoder.proj[i].bias)
