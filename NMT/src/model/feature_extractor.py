from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import math
from IPython import embed
from .transformer import TransformerEncoderLayer, PositionalEmbedding


logger = getLogger()


class ConvFeatureExtractor(nn.Module):

    def __init__(self, params, encoder):
        super().__init__()
        self.embeddings = encoder.embeddings
        self.style_embeddings = encoder.style_embeddings

        num_filters = params.feat_extr_n_filters
        filter_sizes = params.filter_sizes
        self.conv_layers = []
        for filter_size in filter_sizes:
            self.conv_layers.append(nn.Conv2d(1, num_filters,
                    (filter_size, params.emb_dim)))
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, sentence, length, attributes, src_tokens=None, soft=False):
        # embed style
        style_embed = self.style_embeddings(attributes)
        style_embed = torch.mean(torch.transpose(style_embed, 0, 1), 0)
        # embed src tokens and replace <BOS> w/ style_embed
        if soft: # soft src tokens
            src_embed = sentence
        else:
            src_embed = self.embeddings(sentence)
        src_embed[0] = style_embed

        # reshape for conv net
        src_embed = src_embed.unsqueeze(1).permute(2, 1, 0, 3)

        maxes_over_time = []
        for conv_layer in self.conv_layers:
            x = torch.tanh(conv_layer(src_embed))
            x = torch.max(x.squeeze(dim=3), 2)[0]
            maxes_over_time.append(x)

        rep = torch.cat(maxes_over_time, 1)

        return rep


class RecurrentFeatureExtractor(nn.Module):
    pass


class TransformerFeatureExtractor(nn.Module):

    def __init__(self, params, encoder):
        super().__init__()
        self.embeddings = encoder.embeddings
        self.style_embeddings = encoder.style_embeddings

        self.dropout = params.dropout

        embed_dim = params.encoder_embed_dim

        self.padding_idx = params.pad_index
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=params.left_pad_source,
        )

        self.freeze_enc_emb = params.freeze_enc_emb

        self.num_layers = params.feat_extr_layers

        self.layers = nn.ModuleList()
        for k in range(self.num_layers):
            self.layers.append(TransformerEncoderLayer(params))

    def forward(self, sentence, length, attributes, src_tokens=None, soft=False):

        # NOTE: `src_attributes` are unused for now, will be used for style-specific word embeddings

        # embed style
        style_embed = self.style_embeddings(attributes)
        style_embed = torch.mean(torch.transpose(style_embed, 0, 1), 0)
        # embed src tokens and replace <BOS> w/ style_embed
        if soft: # soft src tokens
            src_embed = sentence
        else:
            src_tokens = sentence
            src_embed = self.embeddings(sentence)
        src_embed[0] = style_embed

        # embed positions
        x = self.embed_scale * src_embed
        x = x.detach() if self.freeze_enc_emb else x
        x = x + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        encoder_padding_mask = src_tokens.t().eq(self.padding_idx)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        # max-over-time
        x = torch.max(x, 0)[0]

        return x

###############################################################################

def batched_IPOT(feature_vecs1, feature_vecs2, iterations=50, beta=1):
    feature_vecs1 = torch.transpose(feature_vecs1, 0, 1)
    feature_vecs2 = torch.transpose(feature_vecs2, 0, 1)
    batch_size = feature_vecs1.shape[0]
    n1, n2 = feature_vecs1.shape[1], feature_vecs2.shape[1]
    assert feature_vecs1.shape[0] == feature_vecs2.shape[0]
    sigma = (torch.ones((batch_size, n2, 1)) / n2).cuda()
    T = torch.ones((batch_size, n1, n2)).cuda()
    C = cost_matrix(feature_vecs1, feature_vecs2)
    A = torch.exp(-C / beta)
    
    for _ in range(iterations):
        Q = A * T
        delta = 1 / (n1 * torch.matmul(Q, sigma))
        sigma = 1 / (n2 * torch.matmul(torch.transpose(Q, 1, 2), delta))
        T = torch.matmul(torch.diag_embed(delta.squeeze(), dim1=1, dim2=2), 
                         torch.matmul(Q, torch.diag_embed(sigma.squeeze(),
                                                          dim1=1, dim2=2)))
    T = T.detach()

    weighted_dists = torch.matmul(torch.transpose(C, 1, 2), T)
    _ones = torch.diag_embed(torch.ones(batch_size, n1), dim1=1, dim2=2).cuda()
    return torch.sum(weighted_dists * _ones)

def IPOT(feature_vecs1, feature_vecs2, iterations=50, beta=1):
    n1, n2 = feature_vecs1.shape[0], feature_vecs2.shape[0]
    sigma = (torch.ones((n2, 1)) / n2).cuda()
    T = torch.ones((n1, n2)).cuda()
    C = cost_matrix(feature_vecs1, feature_vecs2)
    A = torch.exp(-C / beta)
    
    for _ in range(iterations):
        Q = A * T
        delta = 1 / (n1 * torch.matmul(Q, sigma))
        sigma = 1 / (n2 * torch.matmul(torch.transpose(Q, 0, 1), delta))
        T = torch.matmul(torch.diag(delta.squeeze()), 
                         torch.matmul(Q, torch.diag(sigma.squeeze())))
    T = T.detach()

    return torch.trace(torch.matmul(torch.transpose(C, 0, 1), T))


def cost_matrix(feature_vecs1, feature_vecs2):
    feature_vecs1 = F.normalize(feature_vecs1, p=2, dim=-1)
    feature_vecs2 = F.normalize(feature_vecs2, p=2, dim=-1)
    
    cosine_similarity = torch.matmul(feature_vecs1, torch.transpose(feature_vecs2, -1, -2))
    
    return (1 - cosine_similarity)
