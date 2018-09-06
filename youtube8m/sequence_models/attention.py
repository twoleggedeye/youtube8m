import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadMultiQueryAttention(nn.Module):
    """
    Multi-head multi-query attention.
    Allows to attend each sequence to each query and produce multiple attended variants.
    """

    def __init__(self, in_query_size, in_key_size, work_query_size, in_value_size, out_value_size, heads_number,
                 temperature=None, projection_init=nn.init.xavier_normal_):
        """
        :param query_size: integer, size of query
        :param temperature:
        """
        super().__init__()
        self._temperature = (in_query_size ** 0.5) if temperature is None else temperature

        self.query_projections = nn.Parameter(torch.FloatTensor(heads_number, in_query_size, work_query_size))
        projection_init(self.query_projections)
        self.key_projections = nn.Parameter(torch.FloatTensor(heads_number, in_key_size, work_query_size))
        projection_init(self.key_projections)
        self.value_projections = nn.Parameter(torch.FloatTensor(heads_number, in_value_size, out_value_size))
        projection_init(self.value_projections)

    def forward(self, queries, keys, values, mask=None):
        """
        :param queries: BatchSize x QueriesN x InQuerySize
        :param keys: BatchSize x SequenceLen x InQuerySize
        :param values: BatchSize x SequenceLen x InValueSize
        :param mask: BatchSize x SequenceLen x QueriesN - 1 if this element is meaningful, 0 if it is absent
        :return: BatchSize x QueriesN x HeadsN x OutValueSize
        """
        # Definitions for Einstein summations:
        # b - BatchSize
        # q - QueriesN
        # i - InQuerySize, InValueSize (they are different, but that's does not matter, they never occur together)
        # k - WorkQuerySize
        # l - SequenceLen
        # h - HeadsN
        projected_queries = torch.einsum("bqi,hik->bqhk", (queries, self.query_projections))
        projected_keys = torch.einsum("bli,hik->blhk", (keys, self.key_projections))
        projected_values = torch.einsum("bli,hiv->blhv", (values, self.value_projections))

        relevance = torch.einsum('bqhk,blhk->blqh', (projected_queries, projected_keys)) / self._temperature

        if mask is not None:
            assert mask.size() == relevance.size()[:3]
            mask = 1 - mask.unsqueeze(-1).expand_as(relevance)
            relevance.data.masked_fill_(mask, float('-inf'))

        normed_relevance = F.softmax(relevance, dim=1)

        # normed_relevance can be inf when both timestep and query are totally masked out
        # i.e. mask[sample, i, :] == 0 and mask[sample, :, j] == 0
        # so we have to zero these elements
        if mask is not None:
            normed_relevance.data.masked_fill_(mask, 0)

        return torch.einsum('blhv,blqh->bqhv', (projected_values, normed_relevance))


def sine_position_coding(dims=32, max_length=300, freq_factor=2):
    time = np.arange(0, max_length, 1)
    freq_factors = freq_factor * np.arange(1, dims + 1, 1)
    codes = np.sin(2 * np.pi * np.outer(time, freq_factors) / max_length).astype('float32')
    return torch.from_numpy(codes)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_size, out_size, heads_number, max_length=500, pos_code_ctor=sine_position_coding,
                 **base_att_kwargs):
        super().__init__()
        self._impl = MultiHeadMultiQueryAttention(in_size, in_size, out_size,
                                                  in_size, out_size,
                                                  heads_number, **base_att_kwargs)
        self._position_codes = nn.Parameter(pos_code_ctor(in_size, max_length=max_length))
        self._position_codes.requires_grad_(False)

    def forward(self, sequence, mask=None):
        """
        :param sequence: BatchSize x SequenceLen x InValueSize
        :param mask: BatchSize x SequenceLen
        :return: BatchSize x SequenceLen x HeadsN x OutValueSize
        """
        pos_coding = self._position_codes[:sequence.size(1), :].unsqueeze(0).expand(sequence.size(0), -1, -1)
        keys_and_queries = sequence  + pos_coding
        if mask is not None:
            assert mask.size() == sequence.size()[:2]
            mask = torch.einsum('bl,bq->blq', (mask, mask))
        return self._impl(keys_and_queries, keys_and_queries, sequence, mask=mask)


class TransformerEncoder(nn.Module):
    def __init__(self, in_size, out_size, heads_n=4, layers_n=2, dropout=0.3, activation=F.relu, **self_attention_kwargs):
        super().__init__()
        sizes = [in_size] + ([out_size] * layers_n)
        self._attentions = nn.ModuleList([MultiHeadSelfAttention(sizes[i], sizes[i+1], heads_n,
                                                                 **self_attention_kwargs)
                                          for i in range(layers_n)])
        self._projections = nn.ModuleList([nn.Linear(out_size * heads_n, out_size) for _ in range(layers_n)])
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

    def forward(self, sequence, mask=None):
        """
        :param sequence: BatchSize x SequenceLen x InSize
        :param mask: BatchSize x SequenceLen
        :return: BatchSize x SequenceLen x OutSize
        """
        batch_size, sequence_len = sequence.size()[:2]
        for att_layer, proj_layer in zip(self._attentions, self._projections):
            sequence = att_layer(sequence, mask=mask).view(batch_size, sequence_len, -1)
            sequence = proj_layer(sequence)
            sequence = self._dropout(sequence)
            sequence = self._activation(sequence)
        return sequence
