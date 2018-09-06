import numpy as np
import torch.nn as nn
import torch


class RandomFrameSequenceExtractor(nn.Module):
    def __init__(self, min_length, max_length):
        super().__init__()
        self._min_length = min_length
        self._max_length = max_length

    def forward(self, video, audio):
        sequence_length = video.shape[1]
        min_length = min(self._min_length, sequence_length)

        range_start = np.random.randint(0, sequence_length - min_length + 1)
        length = np.random.randint(min_length, self._max_length + 1)
        length = min(length, sequence_length - range_start)

        return video[:, range_start:range_start + length], audio[:, range_start:range_start + length]


class RandomFramesExtractorWithBootstrap(nn.Module):
    def __init__(self, length):
        super().__init__()
        self._length = length

    def forward(self, video, audio):
        indices_arange = np.arange(0, video.shape[1])

        result_video = []
        result_audio = []
        for batch_index in range(video.shape[0]):
            indices = np.random.choice(indices_arange, size=min(self._length, video.shape[1]))
            indices = sorted(indices)
            result_video.append(video[batch_index][indices].unsqueeze(0))
            result_audio.append(audio[batch_index][indices].unsqueeze(0))

        return torch.cat(result_video), torch.cat(result_audio)


class FramesRangeExtractorWithRandomStep(nn.Module):
    def __init__(self, min_step, max_step):
        super().__init__()
        self._min_step = min_step
        self._max_step = max_step

    def forward(self, video, audio):
        step = np.random.randint(self._min_step, self._max_step + 1)
        indices = np.arange(0, video.shape[1], step=step)
        return video[:, indices], audio[:, indices]


class RandomFramesMeanCombiner(nn.Module):
    def __init__(self, max_output_length):
        super().__init__()
        self._max_output_length = max_output_length

    def forward(self, video, audio):
        output_length = min(self._max_output_length, video.shape[1]) - 1

        indices = np.arange(0, video.shape[1] - 1)
        border_indices = np.random.choice(indices, output_length, replace=False)
        border_indices = sorted(border_indices + 1)
        border_indices.append(video.shape[1])

        result_video = []
        result_audio = []
        left_border = 0
        for right_border in border_indices:
            result_video.append(video[:, left_border:right_border].mean(dim=1).unsqueeze(1))
            result_audio.append(audio[:, left_border:right_border].mean(dim=1).unsqueeze(1))
            left_border = right_border

        return torch.cat(result_video, dim=1), torch.cat(result_audio, dim=1)
