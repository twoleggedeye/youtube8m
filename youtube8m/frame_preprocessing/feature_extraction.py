import torch.nn as nn


class MeanExtractor(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)


class StdExtractor(nn.Module):
    def forward(self, x):
        return x.std(dim=1)


class MinExtractor(nn.Module):
    def forward(self, x):
        return x.min(dim=1)[0]


class MaxExtractor(nn.Module):
    def forward(self, x):
        return x.max(dim=1)[0]


class MedianExtractor(nn.Module):
    def forward(self, x):
        return x.median(dim=1)[0]


class ModeExtractor(nn.Module):
    def forward(self, x):
        return x.mode(dim=1)[0]
