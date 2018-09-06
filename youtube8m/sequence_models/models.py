import torch
from torch import nn
from torch.nn import functional as F

from youtube8m.sequence_models.attention import TransformerEncoder
from youtube8m.sequence_models.seq_utils import infer_lengths_from_mask, infer_mask_from_batch_data
from youtube8m.video_level_nn_models.defaults import AUDIO_FEATURES_SIZE, RGB_FEATURES_SIZE,\
    YOUTUBE8M_LABELS_N, MAX_SEQ_LEN
from youtube8m.video_level_nn_models.models import FCN, ResNetLike
from youtube8m.utils.io import load_class


class TransformerClassifier(nn.Module):
    def __init__(self, in_size, inner_size, out_size, out_act=F.sigmoid, **transf_kwargs):
        super().__init__()
        self._encoder = TransformerEncoder(in_size, inner_size, **transf_kwargs)
        self._out = nn.Linear(inner_size, out_size)
        self._out_act = out_act

    def forward(self, x, mask=None):
        """
        :param video: BatchSize x SequenceLen x InSize
        :return: BatchSize x OutSize
        """
        if mask is None:
            mask = infer_mask_from_batch_data(x)

        enc = self._encoder(x, mask=mask).mean(1)
        return self._out_act(self._out(enc))


class EarlyConcatTransformerClassifier(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=100, **transf_kwargs):
        super().__init__()
        self._impl = TransformerClassifier(rgb_size + audio_size, inner_size, output_size, **transf_kwargs)

    def forward(self, video, audio, mask=None):
        """
        :param video: BatchSize x SequenceLen x RGBSize
        :param audio: BatchSize x SequenceLen x AudioSize
        :return: BatchSize x OutputSize
        """
        if mask is None:
            mask = infer_mask_from_batch_data(video)

        return self._impl(torch.cat([video, audio], dim=2), mask=mask)


class EarlyConcatMean(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_sizes=(2048, 2048)):
        super().__init__()
        self._impl = FCN(rgb_size + audio_size, output_size, inner_sizes, dropout=0.2, out_activation=nn.Sigmoid)

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        batch = []
        for index in range(video.shape[0]):
            mean_video = video[index, :lengths[index]].mean(0)
            mean_audio = audio[index, :lengths[index]].mean(0)
            batch.append(torch.cat([mean_video, mean_audio]).unsqueeze(0))

        return self._impl(torch.cat(batch))


class EarlyConcatGRU(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=1024, layers_number=2):
        super().__init__()
        self._layers = nn.ModuleList([nn.GRU(input_size=rgb_size + audio_size if i == 0 else inner_size,
                                             hidden_size=inner_size,
                                             num_layers=1,
                                             batch_first=True,
                                             bidirectional=False) for i in range(layers_number)])
        self._out = FCN(inner_size * layers_number - 1024, output_size, (2048,), dropout=0.2)
        self._num_layers = layers_number
        self._inner_size = inner_size

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        inputs = torch.cat([video, audio], dim=2)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs[perm_idx], seq_lengths, batch_first=True)

        states = []
        for layer in self._layers:
            inputs, state = layer(inputs)
            state = state.permute(1, 2, 0).squeeze(2)
            states.append(state)

        representations = torch.cat(states, dim=1)[inverse_idx]
        return self._out(representations)


class EarlyConcatLSTM(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=1024, layers_number=2):
        super().__init__()
        self._layers = nn.ModuleList([nn.LSTM(input_size=rgb_size + audio_size if i == 0 else inner_size,
                                              hidden_size=inner_size,
                                              num_layers=1,
                                              batch_first=True,
                                              bidirectional=False) for i in range(layers_number)])
        self._out = FCN(inner_size * 2 * layers_number, output_size, (2048,), dropout=0.1)
        self._num_layers = layers_number
        self._inner_size = inner_size

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        inputs = torch.cat([video, audio], dim=2)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs[perm_idx], seq_lengths, batch_first=True)

        states = []
        for layer in self._layers:
            inputs, (state1, state2) = layer(inputs)
            state1 = state1.permute(1, 2, 0).squeeze(2)
            state2 = state2.permute(1, 2, 0).squeeze(2)
            state = torch.cat([state1, state2], dim=1)
            states.append(state)

        representations = torch.cat(states, dim=1)[inverse_idx]
        return self._out(representations)


class EarlyConcatLSTM2(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=512, layers_number=3):
        super().__init__()
        self._layers = nn.ModuleList([nn.LSTM(input_size=rgb_size + audio_size if i == 0 else inner_size,
                                              hidden_size=inner_size,
                                              num_layers=1,
                                              batch_first=True,
                                              bidirectional=False) for i in range(layers_number)])
        self._out = FCN(inner_size * 2 * layers_number, output_size, (1024, 1024))
        self._num_layers = layers_number
        self._inner_size = inner_size

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        inputs = torch.cat([video, audio], dim=2)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs[perm_idx], seq_lengths, batch_first=True)

        states = []
        for layer in self._layers:
            inputs, (state1, state2) = layer(inputs)
            state1 = state1.permute(1, 2, 0).squeeze(2)
            state2 = state2.permute(1, 2, 0).squeeze(2)
            state = torch.cat([state1, state2], dim=1)
            states.append(state)

        representations = torch.cat(states, dim=1)[inverse_idx]
        return self._out(representations)


class EarlyConcatGRU2(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=1024, layers_number=3):
        super().__init__()
        self._layers = nn.ModuleList([nn.GRU(input_size=rgb_size + audio_size if i == 0 else inner_size,
                                             hidden_size=inner_size,
                                             num_layers=1,
                                             batch_first=True,
                                             bidirectional=False) for i in range(layers_number)])
        self._out = FCN(inner_size * layers_number, output_size, (1024, 1024))
        self._num_layers = layers_number
        self._inner_size = inner_size

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        inputs = torch.cat([video, audio], dim=2)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs[perm_idx], seq_lengths, batch_first=True)

        states = []
        for layer in self._layers:
            inputs, state = layer(inputs)
            state = state.permute(1, 2, 0).squeeze(2)
            states.append(state)

        representations = torch.cat(states, dim=1)[inverse_idx]
        return self._out(representations)


class EarlyConcatGRU3(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=2048, layers_number=2):
        super().__init__()
        self._layers = nn.ModuleList([nn.GRU(input_size=rgb_size + audio_size if i == 0 else inner_size,
                                             hidden_size=inner_size,
                                             num_layers=1,
                                             batch_first=True,
                                             bidirectional=False) for i in range(layers_number)])
        self._out = FCN(inner_size * layers_number, output_size, (1524, 1524))
        self._num_layers = layers_number
        self._inner_size = inner_size

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        inputs = torch.cat([video, audio], dim=2)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs[perm_idx], seq_lengths, batch_first=True)

        states = []
        for layer in self._layers:
            inputs, state = layer(inputs)
            state = state.permute(1, 2, 0).squeeze(2)
            states.append(state)

        representations = torch.cat(states, dim=1)[inverse_idx]
        return self._out(representations)


class EarlyConcatLSTM3(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=512, layers_number=3):
        super().__init__()
        self._layers = nn.ModuleList([nn.LSTM(input_size=rgb_size + audio_size + inner_size * i,
                                              hidden_size=inner_size,
                                              num_layers=1,
                                              batch_first=True,
                                              bidirectional=False) for i in range(layers_number)])
        self._out = FCN(inner_size * 2 * layers_number, output_size, (1024, 1024))
        self._num_layers = layers_number
        self._inner_size = inner_size

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        inputs = torch.cat([video, audio], dim=2)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs[perm_idx], seq_lengths, batch_first=True)

        states = []
        for layer in self._layers:
            new_inputs, (state1, state2) = layer(inputs)
            inputs, _ = torch.nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
            new_inputs, _ = torch.nn.utils.rnn.pad_packed_sequence(new_inputs, batch_first=True)

            inputs = torch.cat([inputs, new_inputs], dim=2)
            inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lengths, batch_first=True)
            state1 = state1.permute(1, 2, 0).squeeze(2)
            state2 = state2.permute(1, 2, 0).squeeze(2)
            state = torch.cat([state1, state2], dim=1)
            states.append(state)

        representations = torch.cat(states, dim=1)[inverse_idx]
        return self._out(representations)


class EarlyConcatLSTM4(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, rgb_inner_size=1024, audio_inner_size=128, layers_number=1):
        super().__init__()
        self._rgb_layers = nn.ModuleList([nn.LSTM(input_size=rgb_size + 2 * rgb_inner_size * i,
                                                  hidden_size=rgb_inner_size,
                                                  num_layers=1,
                                                  batch_first=True,
                                                  bidirectional=True) for i in range(layers_number)])

        self._audio_layers = nn.ModuleList([nn.LSTM(input_size=audio_size + 2 * audio_inner_size * i,
                                                    hidden_size=audio_inner_size,
                                                    num_layers=1,
                                                    batch_first=True,
                                                    bidirectional=True) for i in range(layers_number)])

        self._first_linear_rgb = nn.Linear(rgb_size, rgb_size)
        self._first_linear_audio = nn.Linear(audio_size, audio_size)

        self._rgb_attention = nn.Linear(rgb_size + rgb_inner_size * 2 * layers_number, 1)
        self._audio_attention = nn.Linear(audio_size + audio_inner_size * 2 * layers_number, 1)

        self._bn = nn.BatchNorm1d(
            rgb_size + rgb_inner_size * 2 * layers_number + audio_size + audio_inner_size * 2 * layers_number)
        self._out = FCN(
            rgb_size + rgb_inner_size * 2 * layers_number + audio_size + audio_inner_size * 2 * layers_number,
            output_size, (4096, 4096))
        self._num_layers = layers_number

    def forward(self, video, audio):
        mask = infer_mask_from_batch_data(video)
        lengths = infer_lengths_from_mask(mask)

        video = self._first_linear_rgb(video)
        video = F.relu(video)

        audio = self._first_linear_audio(audio)
        audio = F.relu(audio)

        seq_lengths, perm_idx = lengths.sort(descending=True)
        _, inverse_idx = perm_idx.sort()
        video = torch.nn.utils.rnn.pack_padded_sequence(video[perm_idx], seq_lengths, batch_first=True)
        audio = torch.nn.utils.rnn.pack_padded_sequence(audio[perm_idx], seq_lengths, batch_first=True)

        for layer in self._rgb_layers:
            new_video, _ = layer(video)
            video, _ = torch.nn.utils.rnn.pad_packed_sequence(video, batch_first=True)
            new_video, _ = torch.nn.utils.rnn.pad_packed_sequence(new_video, batch_first=True)

            video = torch.cat([video, new_video], dim=2)
            video = torch.nn.utils.rnn.pack_padded_sequence(video, seq_lengths, batch_first=True)

        for layer in self._audio_layers:
            new_audio, _ = layer(audio)
            audio, _ = torch.nn.utils.rnn.pad_packed_sequence(audio, batch_first=True)
            new_audio, _ = torch.nn.utils.rnn.pad_packed_sequence(new_audio, batch_first=True)

            audio = torch.cat([audio, new_audio], dim=2)
            audio = torch.nn.utils.rnn.pack_padded_sequence(audio, seq_lengths, batch_first=True)

        video, _ = torch.nn.utils.rnn.pad_packed_sequence(video, batch_first=True)
        audio, _ = torch.nn.utils.rnn.pad_packed_sequence(audio, batch_first=True)

        rgb_attention_weights = F.softmax(self._rgb_attention(video), dim=1)
        video = (video * rgb_attention_weights).sum(dim=1)

        audio_attention_weights = F.softmax(self._audio_attention(audio), dim=1)
        audio = (audio * audio_attention_weights).sum(dim=1)

        representations = torch.cat([video, audio], dim=1)[inverse_idx]
        return self._out(self._bn(representations))


class AttentionModel(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, layers_number=3, hidden_size=1024):
        super().__init__()

        self._linear_layers = nn.ModuleList(
            [nn.Linear(rgb_size + audio_size + hidden_size * i, hidden_size) for i in range(layers_number)])
        self._attention_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layers_number)])

        self._bn = nn.BatchNorm1d(rgb_size + audio_size + hidden_size * layers_number)
        self._out = FCN(rgb_size + audio_size + hidden_size * layers_number, output_size, (4096, 4096))
        self._num_layers = layers_number

    def forward(self, video, audio):
        inputs = torch.cat([video, audio], dim=2)

        for i in range(self._num_layers):
            vectors = self._linear_layers[i](inputs)
            vectors = F.relu(vectors)
            weights = F.softmax(self._attention_layers[i](vectors), dim=1)
            vector = (vectors * weights).sum(dim=1)
            vector = vector.repeat(1, vectors.shape[1]).view(-1, vectors.shape[1], vectors.shape[2])
            inputs = torch.cat([inputs, vector], dim=2)

        inputs = inputs.mean(dim=1)
        return self._out(self._bn(inputs))


class AttentionModel2(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, layers_number=3, hidden_size=1024):
        super().__init__()

        self._linear_layers = nn.ModuleList(
            [nn.Linear(rgb_size + audio_size + hidden_size * i, hidden_size) for i in range(layers_number)])
        self._attention_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layers_number)])

        self._bn = nn.BatchNorm1d(rgb_size + audio_size + hidden_size * layers_number)
        self._out = FCN(rgb_size + audio_size + hidden_size * layers_number, output_size, (4096, 4096), dropout=0.3)
        self._num_layers = layers_number

    def forward(self, video, audio):
        inputs = torch.cat([video, audio], dim=2)

        for i in range(self._num_layers):
            vectors = self._linear_layers[i](inputs)
            vectors = F.relu(vectors)
            weights = F.softmax(self._attention_layers[i](vectors), dim=1)
            vector = (vectors * weights).sum(dim=1)
            vector = vector.repeat(1, vectors.shape[1]).view(-1, vectors.shape[1], vectors.shape[2])
            inputs = torch.cat([inputs, vector], dim=2)

        inputs = inputs.mean(dim=1)
        return self._out(self._bn(inputs))


class AttentionModel3(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, layers_number=5, hidden_size=512):
        super().__init__()

        self._linear_layers = nn.ModuleList(
            [nn.Linear(rgb_size + audio_size + hidden_size * i, hidden_size) for i in range(layers_number)])
        self._attention_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layers_number)])

        self._bn = nn.BatchNorm1d(rgb_size + audio_size + hidden_size * layers_number)
        self._out = FCN(rgb_size + audio_size + hidden_size * layers_number, output_size, (4096, 4096), dropout=0.3)
        self._num_layers = layers_number

    def forward(self, video, audio):
        inputs = torch.cat([video, audio], dim=2)

        for i in range(self._num_layers):
            vectors = self._linear_layers[i](inputs)
            vectors = F.relu(vectors)
            weights = F.softmax(self._attention_layers[i](vectors), dim=1)
            vector = (vectors * weights).sum(dim=1)
            vector = vector.repeat(1, vectors.shape[1]).view(-1, vectors.shape[1], vectors.shape[2])
            inputs = torch.cat([inputs, vector], dim=2)

        inputs = inputs.mean(dim=1)
        return self._out(self._bn(inputs))


class AttentionModel4(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, layers_number=4, hidden_size=768):
        super().__init__()

        self._linear_layers = nn.ModuleList(
            [nn.Linear(rgb_size + audio_size + hidden_size * i, hidden_size) for i in range(layers_number)])
        self._attention_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layers_number)])

        self._dropout_layers = nn.ModuleList([nn.Dropout(i / 10.) for i in range(layers_number)])

        self._bn = nn.BatchNorm1d(rgb_size + audio_size + hidden_size * layers_number)
        self._out = FCN(rgb_size + audio_size + hidden_size * layers_number, output_size, (4096, 4096), dropout=0.3)
        self._num_layers = layers_number

    def forward(self, video, audio):
        inputs = torch.cat([video, audio], dim=2)

        for i in range(self._num_layers):
            vectors = self._linear_layers[i](inputs)
            vectors = F.relu(vectors)
            weights = F.softmax(self._attention_layers[i](vectors), dim=1)
            vector = (vectors * weights).sum(dim=1)
            vector = vector.repeat(1, vectors.shape[1]).view(-1, vectors.shape[1], vectors.shape[2])
            inputs = torch.cat([inputs, vector], dim=2)

        inputs = inputs.mean(dim=1)
        return self._out(self._bn(inputs))


class AttentionModel5(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, layers_number=4, hidden_size=768):
        super().__init__()

        self._linear_layers = nn.ModuleList(
            [nn.Linear(rgb_size + audio_size + hidden_size * i, hidden_size) for i in range(layers_number)])
        self._attention_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(layers_number)])

        self._dropout_layers = nn.ModuleList([nn.Dropout((i + 1) / 10.) for i in range(layers_number)])
        self._out = FCN(rgb_size + audio_size + hidden_size * layers_number, output_size, (4096, 4096), dropout=0.3)
        self._num_layers = layers_number

    def forward(self, video, audio):
        inputs = torch.cat([video, audio], dim=2)

        for i in range(self._num_layers):
            vectors = self._linear_layers[i](inputs)
            vectors = F.relu(vectors)
            weights = F.softmax(self._attention_layers[i](vectors), dim=1)
            vector = (vectors * weights).sum(dim=1)
            vector = vector.repeat(1, vectors.shape[1]).view(-1, vectors.shape[1], vectors.shape[2])
            inputs = torch.cat([inputs, vector], dim=2)
            inputs = self._dropout_layers[i](inputs)

        inputs = inputs.mean(dim=1)
        return self._out(inputs)


# frame fcn model
class EarlyConcatAlternate(nn.Module):
    class AlternateFC(nn.Module):
        def __init__(self, in_features, in_frames, out_features):
            super().__init__()
            self.features_fc = nn.Linear(in_features, out_features)
            self.bn_feature = nn.BatchNorm1d(out_features)
            self.bn_frame = nn.BatchNorm1d(out_features)

        def forward(self, X):
            batch_sz, frame, n_features = X.shape
            X_featurewise = F.relu(self.bn_feature(self.features_fc(X).permute([0, 2, 1]).contiguous()))
            X_featurewise, _ = X_featurewise.max(dim=-1)
            return X_featurewise

    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, video_reduced_sz=1024, audio_reduced_sz=128,
                 inner_size=1024, concat_id_block_num=1, av_id_block_num=1, dropout=0.4, n_frames=303):
        super().__init__()
        self.dim_reduction_video = EarlyConcatAlternate.AlternateFC(rgb_size,
                                                                    n_frames,
                                                                    video_reduced_sz)

        self.dim_reduction_audio = EarlyConcatAlternate.AlternateFC(audio_size,
                                                                    n_frames,
                                                                    audio_reduced_sz)

        self.resnet = ResNetLike(inner_size=inner_size, video_features_sz=video_reduced_sz,
                                 audio_features_sz=audio_reduced_sz,
                                 av_id_block_num=av_id_block_num, concat_id_block_num=concat_id_block_num)

    def forward(self, video, audio, **unused_kwargs):
        batch_sz = video.shape[0]
        video_reduced = self.dim_reduction_video(video)
        audio_reduced = self.dim_reduction_audio(audio)
        video_reduced = video_reduced.view(batch_sz, -1)
        return self.resnet(video_reduced, audio_reduced)


class VLADBoW(nn.Module):
    """
    VLAD pooling as described in https://arxiv.org/pdf/1706.06905.pdf
    """
    def __init__(self, input_n_features, n_clusters, l2_normalize=False, power=None):
        super().__init__()
        self.assignment_fc = nn.Linear(input_n_features, n_clusters)
        self.cluster_centers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(n_clusters, input_n_features)))
        self.n_clusters = n_clusters
        self.l2_normalize = l2_normalize
        self.power = power

    def forward(self, X):
        batch_sz, n_frames, n_features = X.shape
        if self.power is not None:
            soft_assignment = F.softmax(self.assignment_fc(X).pow(self.power), dim=-1)
        else:
            soft_assignment = F.softmax(self.assignment_fc(X), dim=-1)
        return soft_assignment.sum(dim=1)


class BoWResnetLike(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_size=1024, n_clusters=1000, l2_normalize=False,
                 concat_id_block_num=1, av_id_block_num=1, dropout=0.4, bow_dropout=True,
                 softmax_power=None, learnable_power=False):
        super().__init__()
        if learnable_power:
            self.power_param = nn.Parameter(torch.Tensor([3]).float())
        else:
            self.power_param = softmax_power
        self.video_bow = VLADBoW(rgb_size, n_clusters, l2_normalize, self.power_param)
        self.audio_bow = VLADBoW(audio_size, n_clusters, l2_normalize, self.power_param)
        self.resnet = ResNetLike(inner_size=inner_size, video_features_sz=n_clusters,
                                 audio_features_sz=n_clusters, av_id_block_num=av_id_block_num,
                                 concat_id_block_num=concat_id_block_num)
        self.dropout = nn.Dropout(dropout)
        self.bow_dropout = bow_dropout

    def forward(self, video, audio, **unused_kwargs):
        video_bow_tensor = self.video_bow(video)
        audio_bow_tensor = self.audio_bow(audio)
        if self.bow_dropout:
            video_bow_tensor = self.dropout(video_bow_tensor)
            audio_bow_tensor = self.dropout(audio_bow_tensor)
        return self.resnet(video_bow_tensor, audio_bow_tensor)


class LinearCombinations(nn.Module):
    def __init__(self, head_size=3, dropout=0.5,
                 impl_ctor_name='youtube8m.video_level_nn_models.models.ResNetLike',
                 impl_kwargs=None):
        super().__init__()
        self._video_linear_combination = nn.Linear(MAX_SEQ_LEN, head_size)
        self._audio_linear_combination = nn.Linear(MAX_SEQ_LEN, head_size)
        self._dropout = nn.Dropout(p=dropout)
        if impl_kwargs is None:
            impl_kwargs = {'video_features_sz': RGB_FEATURES_SIZE * head_size,
                           'audio_features_sz': AUDIO_FEATURES_SIZE * head_size,
                           'dropout': dropout}
        self._impl = load_class(impl_ctor_name)(**impl_kwargs)

    def forward(self, video, audio):
        pad = nn.ConstantPad3d((0, 0, 0, MAX_SEQ_LEN - video.shape[1], 0, 0), 0)
        video = self._video_linear_combination(pad(video).permute(0, 2, 1))
        audio = self._audio_linear_combination(pad(audio).permute(0, 2, 1))
        video = self._dropout(video.reshape(video.shape[0], -1))
        audio = self._dropout(audio.reshape(audio.shape[0], -1))
        return self._impl(video, audio)


class ResNetLikeSort(nn.Module):
    class FCBlock(nn.Module):
        def __init__(self, input_size, output_size, dropout):
            super().__init__()
            self.fc1 = nn.Conv1d(input_size, output_size, 1)
            self.batch_norm = nn.BatchNorm1d(output_size, track_running_stats=False)
            self.leaky_relu = nn.LeakyReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.fc1(x)
            x = self.batch_norm(x)
            x = self.leaky_relu(x)
            out = self.dropout(x)
            return out

    class Identity(nn.Module):
        def __init__(self, outer_size, inner_size, dropout):
            super().__init__()
            self.fc1 = ResNetLikeSort.FCBlock(outer_size, inner_size, dropout)
            self.fc2 = nn.Conv1d(inner_size, outer_size, 1)

        def forward(self, x):
            x1 = self.fc1(x)
            x1 = self.fc2(x1)
            out = nn.functional.leaky_relu(x + x1)
            return out

    def __init__(self, inner_size=2 * 1024, av_id_block_num=1,
                 concat_id_block_num=1, dropout=0.5, frame_inner_size=1000):
        super().__init__()

        self.audio_blocks = nn.ModuleList(
                [self.FCBlock(AUDIO_FEATURES_SIZE, inner_size, dropout)] +
                [self.Identity(inner_size, inner_size, dropout)
                    for i in range(av_id_block_num)])

        self.video_blocks = nn.ModuleList(
                [self.FCBlock(RGB_FEATURES_SIZE, inner_size, dropout)] +
                [self.Identity(inner_size, inner_size, dropout)
                    for i in range(av_id_block_num)])

        self.concat_blocks = nn.ModuleList(
                [self.Identity(inner_size * 2, inner_size, dropout)
                    for i in range(concat_id_block_num)] +
                [self.FCBlock(inner_size * 2, inner_size, dropout),
                    nn.Conv1d(inner_size, YOUTUBE8M_LABELS_N, 1)])

        self.frame_blocks = nn.ModuleList(
                [self.FCBlock(MAX_SEQ_LEN, frame_inner_size, dropout),
                    nn.Conv1d(frame_inner_size, 1, 1)])

    def forward(self, video, audio):
        pad = nn.ConstantPad3d((0, 0, 0, MAX_SEQ_LEN - video.shape[1], 0, 0), 0)
        video = pad(video).permute(0, 2, 1)
        audio = pad(audio).permute(0, 2, 1)

        for block in self.audio_blocks:
            audio = block(audio)

        for block in self.video_blocks:
            video = block(video)

        concat = torch.cat([audio, video], dim=1)

        for block in self.concat_blocks:
            concat = block(concat)

        frame, _ = concat.sort(dim=2)
        frame = frame.permute(0, 2, 1)

        for block in self.frame_blocks:
            frame = block(frame)

        return frame.squeeze(1)


class FrameConv(nn.Module):
    def create_blocks(self, conv_sizes,  pool_sizes,  channels_num, pooling=None):
        blocks = []
        for i in range(len(pool_sizes)):
            blocks.append(nn.Conv1d(channels_num[i], channels_num[i + 1], conv_sizes[i]))
            blocks.append(nn.Dropout2d(self.dropout))
            self.activation()
            blocks.append(nn.BatchNorm1d(channels_num[i + 1]))
            pool = pooling or self.pooling
            blocks.append(pool(pool_sizes[i]))
        return nn.ModuleList(blocks)

    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N,
                 av_conv_kernel_sizes=[3, 3], av_pool_kernel_sizes=[3, 3],
                 video_channels_num=[2048, 2048], audio_channels_num=[512, 512],
                 concat_conv_kernel_sizes=[3, 3], concat_pool_kernel_sizes=[3],
                 concat_channels_num=[4096, 8192], dropout=0.1, activation=nn.LeakyReLU,
                 pooling=nn.MaxPool1d, end_pooling=nn.AdaptiveMaxPool1d):
        super().__init__()

        self.dropout = dropout
        self.pooling = pooling
        self.activation = activation
        self.audio_blocks = self.create_blocks(av_conv_kernel_sizes, av_pool_kernel_sizes,
                                          [audio_size] + audio_channels_num)

        self.video_blocks = self.create_blocks(av_conv_kernel_sizes, av_pool_kernel_sizes,
                                          [rgb_size] + video_channels_num)

        concat_channels_num = [video_channels_num[-1] + audio_channels_num[-1]] + concat_conv_kernel_sizes
        self.concat_blocks1 = self.create_blocks(concat_conv_kernel_sizes,
                                                 concat_pool_kernel_sizes, concat_channels_num)
        self.concat_blocks2 = self.create_blocks([concat_conv_kernel_sizes[-1]], [1],
                                            concat_channels_num[-2:], end_pooling)

        self.fc = nn.Linear(concat_channels_num[-1], output_size)

    def forward(self, video, audio):
        video = video.permute(0, 2, 1)
        audio = audio.permute(0, 2, 1)

        for block in self.audio_blocks:
            audio = block(audio)

        for block in self.video_blocks:
            video = block(video)

        concat = torch.cat([audio, video], dim=1)

        for block in self.concat_blocks1:
            concat = block(concat)

        for block in self.concat_blocks2:
            concat = block(concat)

        concat = concat.squeeze(2)
        concat = self.fc(concat)

        return concat
