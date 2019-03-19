from logging import getLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


logger = getLogger()


class ConvFeatureExtractor(nn.Module):

    def __init__(self, params, encoder):
        super().__init__()
        self.embeddings = nn.Embedding(encoder.embeddings.weight.shape[0],
                                       encoder.embeddings.weight.shape[1])
        self.embeddings.weight.data = encoder.embeddings.weight.detach()
        self.style_embeddings = nn.Embedding(encoder.style_embeddings.weight.shape[0],
                                             encoder.style_embeddings.weight.shape[1])
        self.style_embeddings.weight.data = encoder.style_embeddings.weight.detach()

        num_filters = 128
        window_sizes = [2, 3, 5, 7]
        self.conv_layers = []
        for window_size in window_sizes:
            self.conv_layers.append(nn.Conv2d(1, num_filters,
                    (window_size, params.emb_dim)).cuda())

    def forward(self, sentence, length, attributes):
        # embed style
        style_embed = self.style_embeddings(attributes)
        style_embed = torch.mean(torch.transpose(style_embed, 0, 1), 0)

        # embed src tokens and replace <BOS> w/ style_embed
        src_embed = self.embeddings(sentence)
        src_embed[0] = style_embed
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
