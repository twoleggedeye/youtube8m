import torch.nn as nn
import torch


EPS = 1e-5


class SequentialClusteringModule(nn.Module):
    def __init__(self, cluster_count):
        super().__init__()

        self._cluster_count = cluster_count

    def forward(self, video, audio):
        video_norm = torch.norm(video, p=2, dim=2) + EPS
        video_normalized = video / video_norm.unsqueeze(2).expand_as(video)
        similarities = video_normalized[:, 1:] * video_normalized[:, :-1]
        similarities = similarities.sum(dim=2).abs()

        _, indices = torch.topk(similarities, self._cluster_count, largest=False, dim=-1)
        indices += 1
        indices = torch.cat([torch.zeros(video.shape[0], 1, dtype=torch.long), indices], dim=1)

        result_video = []
        result_audio = []
        for i in range(video.shape[0]):
            result_video.append(video[i][indices[i]].unsqueeze(0))
            result_audio.append(audio[i][indices[i]].unsqueeze(0))

        return torch.cat(result_video), torch.cat(result_audio)


class GlobalClusteringModule(nn.Module):
    def __init__(self, cluster_count):
        super().__init__()

        self._cluster_count = cluster_count

    def forward(self, video, audio):
        video_norm = torch.norm(video, p=2, dim=2) + EPS
        video_normalized = video / video_norm.unsqueeze(2).expand_as(video)

        result_video = []
        result_audio = []
        for batch_index in range(video.shape[0]):
            video_batch_normalized = video_normalized[batch_index]
            batch_result_indices = [0]

            best_cosine_sims = None
            last_index = 0

            while len(batch_result_indices) != self._cluster_count:
                similarities = video_batch_normalized[last_index] * video_batch_normalized
                similarities = similarities.sum(dim=1).abs()

                if best_cosine_sims is None:
                    best_cosine_sims = similarities

                best_cosine_sims = torch.max(best_cosine_sims, similarities)

                last_index = torch.argmin(best_cosine_sims)
                batch_result_indices.append(int(last_index))

            batch_result_indices = sorted(batch_result_indices)
            result_video.append(video[batch_index][batch_result_indices].unsqueeze(0))
            result_audio.append(audio[batch_index][batch_result_indices].unsqueeze(0))

        return torch.cat(result_video), torch.cat(result_audio)


class SVDClusteringModule(nn.Module):
    def __init__(self, cluster_count):
        super().__init__()
        self._cluster_count = cluster_count

    def forward(self, x):
        result = []

        for batch_index in range(x.shape[0]):
            u, s, v = torch.svd(x[batch_index])
            result.append(v.t()[:self._cluster_count].unsqueeze(0))

        return torch.cat(result)
