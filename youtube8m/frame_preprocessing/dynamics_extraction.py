import pywt
import math
import torch
import torch.nn.functional as F
from torch import nn


class AvgSubtractor(nn.Module):
    def __init__(self, window_size=10, device='cuda'):
        super().__init__()
        self.pool = nn.AvgPool1d(window_size, stride=1)
        self.window_size = window_size

    def forward(self, x):
        # input is [batch, seq, channels]
        # convert to [batch, channels, seq]
        x = x.permute([0, 2, 1])
        pad = _compute_pad_amount(x.shape[1], self.window_size, 1, 1)
        pooled = F.pad(x, [0, pad], 'replicate')
        pooled = self.pool(pooled)
        return (x - pooled).permute([0, 2, 1])


class HighpassFilter(nn.Module):
    def __init__(self, filter_type='db3', dilation=1, device='cuda'):
        super().__init__()
        self.device = device
        self.filter = torch.Tensor(pywt.Wavelet(filter_type).dec_hi[::-1]).to(self.device)
        self.dilation = dilation

    def forward(self, x):
        # input is [batch, seq, channels]
        # convert to [batch, channels, seq]
        x = x.permute([0, 2, 1])
        in_channels = x.shape[1]
        pad = _compute_pad_amount(x.shape[1], self.filter.shape[0], self.dilation, 1)
        x = F.pad(x, [0, pad], 'replicate')
        filter_for_conv = self.filter.view(1, 1, -1).repeat(in_channels, 1, 1)
        x = F.conv1d(x, filter_for_conv, groups=in_channels, dilation=self.dilation)
        return x.permute([0, 2, 1])


class PrincipalComponentPursuit(nn.Module):
    def __init__(self, max_iter=10, stopping=1e-4, device='cuda'):
        self.max_iter = max_iter
        self.stopping = stopping
        self.device = device
        super().__init__()

    def forward(self, batch):
        low_rank_batch = torch.zeros_like(batch).to(self.device)
        sparse_batch = torch.zeros_like(batch).to(self.device)
        for i in range(batch.shape[0]):
            matrix = batch[i]
            dnorm = matrix.norm()
            sparse = torch.zeros_like(matrix).to(self.device)
            lagrange = torch.zeros_like(matrix).to(self.device)
            mu = matrix.shape[0] * matrix.shape[1] / (4 * matrix.abs().sum())
            lam = 1 / max(matrix.shape) ** 0.5
            current = matrix.norm()
            j = 0
            while current / dnorm > self.stopping:
                tt = matrix - sparse + lagrange
                u, s, v = torch.svd(tt, some=True)
                s = self._shrink(s, 1 / mu)
                low_rank = (u * s.view(1, -1)).matmul(v.t())
                sparse = self._shrink(matrix - low_rank + lagrange, lam / mu)
                lagrange += (matrix - low_rank - sparse)
                j += 1
                current = (matrix - low_rank - sparse).norm()
                if j > self.max_iter:
                    break
            low_rank_batch[i] = low_rank
            sparse_batch[i] = sparse
        return low_rank_batch, sparse_batch

    def _shrink(self, matrix, threshold):
        sign = matrix.sign()
        matrix = matrix.abs() - threshold
        matrix[matrix < 0] = 0
        return matrix * sign


def _compute_pad_amount(input_sz, kernel_sz, dilation, stride):
    unpadded_out_sz = (input_sz - kernel_sz - (kernel_sz - 1) * (dilation - 1)) / stride + 1
    unpadded_out_sz = int(math.ceil(unpadded_out_sz))
    pad_amount = input_sz - unpadded_out_sz
    return pad_amount
