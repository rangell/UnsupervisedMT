# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import json
import argparse

from src.data.loader import check_all_data_params, load_data, load_st_data
from src.utils import bool_flag, initialize_exp, sample_style
from src.model import check_mt_model_params, build_mt_model
from src.trainer import TrainerMT
from src.evaluator import EvaluatorMT

from src.model.feature_extractor import ConvFeatureExtractor, cost_matrix, IPOT
 
from IPython import embed


def get_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description='Text style transfer')
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--save_periodic", type=bool_flag, default=False,
                        help="Save the model periodically")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random generator seed (-1 for random)")
    # autoencoder parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of layers in the encoders")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Number of layers in the decoders")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer size")
    parser.add_argument("--window_width", type=int, default=5,
                        help="Max-pool stride and window width on top of encoder outputs")
    parser.add_argument("--lstm_proj", type=bool_flag, default=False,
                        help="Projection layer between decoder LSTM and output layer")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--label-smoothing", type=float, default=0,
                        help="Label smoothing")
    parser.add_argument("--attention", type=bool_flag, default=True,
                        help="Use an attention mechanism")
    if not parser.parse_known_args()[0].attention:
        parser.add_argument("--enc_dim", type=int, default=512,
                            help="Latent space dimension")
        parser.add_argument("--proj_mode", type=str, default="last",
                            help="Projection mode (proj / pool / last)")
        parser.add_argument("--init_encoded", type=bool_flag, default=False,
                            help="Initialize the decoder with the encoded state. Append it to each input embedding otherwise.")
    else:
        parser.add_argument("--transformer", type=bool_flag, default=True,
                            help="Use transformer architecture + attention mechanism")
        if parser.parse_known_args()[0].transformer:
            parser.add_argument("--transformer_ffn_emb_dim", type=int, default=2048,
                                help="Transformer fully-connected hidden dim size")
            parser.add_argument("--attention_dropout", type=float, default=0,
                                help="attention_dropout")
            parser.add_argument("--relu_dropout", type=float, default=0,
                                help="relu_dropout")
            parser.add_argument("--encoder_attention_heads", type=int, default=8,
                                help="encoder_attention_heads")
            parser.add_argument("--decoder_attention_heads", type=int, default=8,
                                help="decoder_attention_heads")
            parser.add_argument("--encoder_normalize_before", type=bool_flag, default=False,
                                help="encoder_normalize_before")
            parser.add_argument("--decoder_normalize_before", type=bool_flag, default=False,
                                help="decoder_normalize_before")
        else:
            parser.add_argument("--input_feeding", type=bool_flag, default=False,
                                help="Input feeding")
            parser.add_argument("--share_att_proj", type=bool_flag, default=False,
                                help="Share attention projetion layer")
    parser.add_argument("--share_encdec_emb", type=bool_flag, default=False,
                        help="Share encoder embeddings / decoder embeddings")
    parser.add_argument("--share_decpro_emb", type=bool_flag, default=False,
                        help="Share decoder embeddings / decoder output projection")
    parser.add_argument("--share_output_emb", type=bool_flag, default=False,
                        help="Share decoder output embeddings")
    parser.add_argument("--share_lstm_proj", type=bool_flag, default=False,
                        help="Share projection layer between decoder LSTM and output layer)")
    parser.add_argument("--share_enc", type=int, default=0,
                        help="Number of layers to share in the encoders")
    parser.add_argument("--share_dec", type=int, default=0,
                        help="Number of layers to share in the decoders")
    # encoder input perturbation
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")
    # discriminator parameters
    parser.add_argument("--dis_layers", type=int, default=3,
                        help="Number of hidden layers in the discriminator")
    parser.add_argument("--dis_hidden_dim", type=int, default=128,
                        help="Discriminator hidden layers dimension")
    parser.add_argument("--dis_dropout", type=float, default=0,
                        help="Discriminator dropout")
    parser.add_argument("--dis_clip", type=float, default=0,
                        help="Clip discriminator weights (0 to disable)")
    parser.add_argument("--dis_smooth", type=float, default=0,
                        help="GAN smooth predictions")
    parser.add_argument("--dis_input_proj", type=bool_flag, default=True,
                        help="Feed the discriminator with the projected output (attention only)")
    # feature extractor parameters
    parser.add_argument("--transformer_feat_extr", type=bool_flag, default=False,
                        help="Use transformer architecture for feature extractor")
    parser.add_argument("--cnn_feat_extr", type=bool_flag, default=False,
                        help="Use cnn architecture for feature extractor")
    if parser.parse_known_args()[0].transformer_feat_extr:
        parser.add_argument("--feat_extr_layers", type=int, default=2,
                            help="Number of layers in transformer feature extractor")
    elif parser.parse_known_args()[0].cnn_feat_extr:
        parser.add_argument("--feat_extr_n_filters", type=int, default=128,
                            help="Number of filters in cnn feature extractor")
        parser.add_argument("--feat_extr_filter_sizes", type=str, default="2,3,4",
                            help="Filter sizes used in cnn feature extractor (e.g. '2,3,4')")

    # dataset
    parser.add_argument("--data_dir", type=str, default="",
                        help="Directory containing all dataset and classifier files")
    parser.add_argument("--text_suffix", type=str, default="",
                        help="Text file suffix")
    parser.add_argument("--attribute_suffix", type=str, default="",
                        help="Attribute (styles) file suffix")
    parser.add_argument("--train_prefix", type=str, default="",
                        help="Train prefix, expected with test and attribute suffixes.")
    parser.add_argument("--dev_prefix", type=str, default="",
                        help="Dev prefix, expected with test and attribute suffixes.")
    parser.add_argument("--test_prefix", type=str, default="",
                        help="Test prefix, expected with test and attribute suffixes.")
    parser.add_argument("--test_para_prefix", type=str, default="",
                        help="Test parallel ground truth prefix, expected with test and attribute suffixes.")
    parser.add_argument("--vocab_filename", type=str, default="",
                        help="Vocabulary filename")
    parser.add_argument("--vocab_min_count", type=int, default=0,
                        help="Vocabulary minimum word count")
    parser.add_argument("--max_len", type=int, default=175,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--metadata_filename", type=str, default="",
                        help="Metadata filename")
    parser.add_argument("--idf_vecs_filename", type=str, default="",
                        help="For evaluating semantic similarity")

    # training steps
    parser.add_argument("--n_dis", type=int, default=0,
                        help="Number of discriminator training iterations")
    parser.add_argument("--fe_smooth_temp", type=str, default="1",
                        help="Temperature for soft argmax")
    parser.add_argument("--otf_sample", type=str, default="0",
                        help="Temperature for sampling back-translations (-1 for greedy decoding)")
    parser.add_argument("--otf_backprop_temperature", type=float, default=-1,
                        help="Back-propagate through the encoder (-1 to disable, temperature otherwise)")
    parser.add_argument("--otf_bt_sync_params_every", type=int, default=1000, metavar="N",
                        help="Number of updates between synchronizing params")
    parser.add_argument("--otf_fe_sync_params_every", type=int, default=50, metavar="N",
                        help="Number of updates between synchronizing params")
    parser.add_argument("--otf_num_processes", type=int, default=30, metavar="N",
                        help="Number of processes to use for OTF generation")
    parser.add_argument("--otf_update_enc", type=bool_flag, default=True,
                        help="Update the encoder during back-translation training")
    parser.add_argument("--otf_update_dec", type=bool_flag, default=True,
                        help="Update the decoder during back-translation training")
    # language model training
    parser.add_argument("--lm_before", type=int, default=0,
                        help="Training steps with language model pretraining (0 to disable)")
    parser.add_argument("--lm_after", type=int, default=0,
                        help="Keep training the language model during MT training (0 to disable)")
    parser.add_argument("--lm_share_enc", type=int, default=0,
                        help="Number of shared LSTM layers in the encoder")
    parser.add_argument("--lm_share_dec", type=int, default=0,
                        help="Number of shared LSTM layers in the decoder")
    parser.add_argument("--lm_share_emb", type=bool_flag, default=False,
                        help="Share language model lookup tables")
    parser.add_argument("--lm_share_proj", type=bool_flag, default=False,
                        help="Share language model projection layers")
    # training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--lambda_xe_ae", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (autoencoding)")
    parser.add_argument("--lambda_ipot_ae", type=str, default="0",
                        help="IPOT reconstruction coefficient (autoencoding)")
    parser.add_argument("--lambda_xe_otf_bt", type=str, default="0",
                        help="Cross-entropy reconstruction coefficient (on-the-fly back-translation autoencoding data)")
    parser.add_argument("--lambda_ipot_otf_bt", type=str, default="0",
                        help="IPOT reconstruction coefficient (on-the-fly back-translation autoencoding data)")
    parser.add_argument("--lambda_dis", type=str, default="0",
                        help="Discriminator loss coefficient")
    parser.add_argument("--lambda_feat_extr", type=str, default="0",
                        help="Feature extractor loss coefficient")
    parser.add_argument("--lambda_adv", type=str, default="0",
                        help="Feature extractor loss coefficient")
    parser.add_argument("--lambda_lm", type=str, default="0",
                        help="Language model loss coefficient")
    parser.add_argument("--enc_optimizer", type=str, default="adam,lr=0.0003",
                        help="Encoder optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--dec_optimizer", type=str, default="enc_optimizer",
                        help="Decoder optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--dis_optimizer", type=str, default="rmsprop,lr=0.0005",
                        help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--feat_extr_optimizer", type=str, default="rmsprop,lr=0.0005",
                        help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    # reload models
    parser.add_argument("--pretrained_emb", type=str, default="",
                        help="Reload pre-trained source and target word embeddings")
    parser.add_argument("--pretrained_out", type=bool_flag, default=False,
                        help="Pretrain the decoder output projection matrix")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pre-trained model")
    parser.add_argument("--reload_enc", type=bool_flag, default=False,
                        help="Reload a pre-trained encoder")
    parser.add_argument("--reload_dec", type=bool_flag, default=False,
                        help="Reload a pre-trained decoder")
    parser.add_argument("--reload_dis", type=bool_flag, default=False,
                        help="Reload a pre-trained discriminator")
    # freeze network parameters
    parser.add_argument("--freeze_enc_emb", type=bool_flag, default=False,
                        help="Freeze encoder embeddings")
    parser.add_argument("--freeze_dec_emb", type=bool_flag, default=False,
                        help="Freeze decoder embeddings")
    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--beam_size", type=int, default=0,
                        help="Beam width (<= 0 means greedy)")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Length penalty: <1.0 favors shorter, >1.0 favors longer sentences")
    return parser


def main(params):
    # check parameters
    assert params.exp_name
    check_all_data_params(params)
    check_mt_model_params(params)

    # initialize experiment / load data / build model
    logger = initialize_exp(params)
    data = load_st_data(params)

    encoder, decoder, discriminator, feat_extr, lm = build_mt_model(params, data)

    # initialize trainer / reload checkpoint / initialize evaluator
    trainer = TrainerMT(encoder, decoder, discriminator, feat_extr, lm, data, params)

    trainer.reload_checkpoint()
    trainer.test_sharing()  # check parameters sharing

    evaluator = EvaluatorMT(trainer, data, params)
    evaluator.run_all_evals(0)
    exit()

    # evaluation mode
    if params.eval_only:
        evaluator.run_all_evals(0)
        exit()

    # language model pretraining
    if params.lm_before > 0:
        logger.info("Pretraining language model for %i iterations ..." % params.lm_before)
        trainer.n_sentences = 0
        for _ in range(params.lm_before):
            for lang in params.langs:
                trainer.lm_step(lang)
            trainer.iter()

    # define epoch size
    if params.epoch_size == -1:
        params.epoch_size = params.n_para
    assert params.epoch_size > 0

    # start training
    for _ in range(trainer.epoch, params.max_epoch):

        logger.info("====================== Starting epoch %i ... ======================" % trainer.epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < params.epoch_size:
            n_batches = 0

            # discriminator training
            if params.n_dis > 0:
                raise NotImplementedError()
            for _ in range(params.n_dis):
                trainer.discriminator_step()

            # language model training
            if params.lambda_lm > 0:
                raise NotImplementedError()
                for _ in range(params.lm_after):
                    for lang in params.langs:
                        trainer.lm_step(lang)

            # generate on-the-fly batch
            before_gen = time.time()
            st_batch = trainer.gen_st_batch()
            trainer.gen_time += time.time() - before_gen

            # auto-encoder training
            if params.lambda_xe_ae > 0:
                n_batches += 1
                trainer.enc_dec_ae_step(params.lambda_xe_ae,
                                        params.lambda_ipot_ae)

            # back-translation training
            if params.lambda_xe_otf_bt > 0:
                n_batches += 1
                trainer.enc_dec_bt_step(st_batch,
                                        params.lambda_xe_otf_bt,
                                        params.lambda_ipot_otf_bt)

            # adversarial optimization of feature extractor
            if params.lambda_feat_extr > 0:
                n_batches += 1
                trainer.feat_extr_step(st_batch, params.lambda_feat_extr)

            # adversarial optimization of generator
            if params.lambda_adv > 0:
                n_batches += 1
                trainer.enc_dec_adv_step(st_batch, params.lambda_adv)

            trainer.iter(n_batches)

        # end of epoch
        logger.info("====================== End of epoch %i ======================" % trainer.epoch)

        # evaluate self-BLEU
        scores = evaluator.run_all_evals(trainer.epoch)

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # save best / save periodic / end epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)
        trainer.test_sharing()


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)
