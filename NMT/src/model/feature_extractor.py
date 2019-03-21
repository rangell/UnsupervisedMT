from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from IPython import embed


logger = getLogger()


class ConvFeatureExtractor(nn.Module):

    def __init__(self, params, encoder):
        super().__init__()
        self.embeddings = encoder.embeddings
        self.style_embeddings = encoder.style_embeddings

        num_filters = 128
        window_sizes = [2, 3, 5, 7]
        self.conv_layers = []
        for window_size in window_sizes:
            self.conv_layers.append(nn.Conv2d(1, num_filters,
                    (window_size, params.emb_dim)))
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, sentence, length, attributes, soft=False):
        # embed style
        style_embed = self.style_embeddings(attributes)
        style_embed = torch.mean(torch.transpose(style_embed, 0, 1), 0)
        # embed src tokens and replace <BOS> w/ style_embed
        if soft: # soft src tokens
            src_embed = torch.matmul(sentence, self.embeddings.weight)
        else:
            src_embed = self.embeddings(sentence)
        src_embed[0] = style_embed

        # reshape for conv net
        src_embed = src_embed.unsqueeze(1).permute(2, 1, 0, 3)

        maxes_over_time = []
        for conv_layer in self.conv_layers:
            x = torch.tanh(conv_layer(src_embed))
            x = torch.max(x.squeeze(), 2)[0]
            maxes_over_time.append(x)

        rep = torch.cat(maxes_over_time, 1)

        return rep


class RecurrentFeatureExtractor(nn.Module):
    pass


class TransformerFeatureExtractor(nn.Module):
    pass


###############################################################################


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
    feature_vecs1 = F.normalize(feature_vecs1, p=2, dim=1)
    feature_vecs2 = F.normalize(feature_vecs2, p=2, dim=1)
    
    cosine_similarity = torch.matmul(feature_vecs1, torch.transpose(feature_vecs2, 0, 1))
    
    return (1 - cosine_similarity)
