import torch
from torch import nn
from torch.nn import functional

from .defaults import YOUTUBE8M_LABELS_N, RGB_FEATURES_SIZE, AUDIO_FEATURES_SIZE
from ..utils.activations import Identity
from ..utils.io import load_class


class FCN(nn.Module):
    def __init__(self, input_size, output_size, inner_sizes=(), dropout=0.0,
                 inner_activation=nn.ReLU, out_activation=Identity):
        super().__init__()
        sizes = [input_size] + list(inner_sizes) + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            is_last_layer = i == len(sizes) - 2
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if not is_last_layer:
                layers.append(nn.Dropout(dropout))
            layers.append(inner_activation() if not is_last_layer else out_activation())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EarlyConcatFCN(nn.Module):
    def __init__(self, rgb_size=RGB_FEATURES_SIZE, audio_size=AUDIO_FEATURES_SIZE,
                 output_size=YOUTUBE8M_LABELS_N, inner_sizes=(1024, 1024), dropout=0.0):
        super().__init__()
        self._impl = FCN(rgb_size + audio_size, output_size, inner_sizes, dropout=dropout)

    def forward(self, video, audio):
        return self._impl(torch.cat([video, audio], dim=1))


class ResNetLike(nn.Module):
    class FCBlock(nn.Module):
        def __init__(self, input_size, output_size, dropout):
            super().__init__()
            self.fc1 = nn.Linear(input_size, output_size)
            self.batch_norm = nn.BatchNorm1d(output_size)
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
            self.fc1 = ResNetLike.FCBlock(outer_size, inner_size, dropout)
            self.fc2 = nn.Linear(inner_size, outer_size)

        def forward(self, x):
            x1 = self.fc1(x)
            x1 = self.fc2(x1)
            out = functional.leaky_relu(x + x1)
            return out

    def __init__(self, inner_size=2 * 1024, av_id_block_num=1,
                 video_features_sz=RGB_FEATURES_SIZE, audio_features_sz=AUDIO_FEATURES_SIZE,
                 concat_id_block_num=1, dropout=0.5,
                 context_gating=False, output_size=YOUTUBE8M_LABELS_N,
                 video_part=1, audio_part=1):
        super().__init__()
        assert 2 * inner_size % (video_part + audio_part) == 0
        audio_size = audio_part * 2 * inner_size // (video_part + audio_part)
        self.context_gating = context_gating
        self.audio_blocks = nn.ModuleList(
                [self.FCBlock(audio_features_sz, audio_size, dropout)] +
                [self.Identity(audio_size, audio_size, dropout)
                    for i in range(av_id_block_num)])

        video_size = video_part * inner_size * 2 // (video_part + audio_part)
        self.video_blocks = nn.ModuleList(
                [self.FCBlock(video_features_sz, video_size, dropout)] +
                [self.Identity(video_size, video_size, dropout)
                    for i in range(av_id_block_num)])

        self.concat_blocks = nn.ModuleList(
                [self.Identity(inner_size * 2, inner_size, dropout)
                    for i in range(concat_id_block_num)] +
                [self.FCBlock(inner_size * 2, inner_size, dropout),
                    nn.Linear(inner_size, output_size)])
        if self.context_gating:
            self.context_weights = nn.Linear(YOUTUBE8M_LABELS_N, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        for block in self.audio_blocks:
            audio = block(audio)

        for block in self.video_blocks:
            video = block(video)

        concat = torch.cat([audio, video], dim=1)

        for block in self.concat_blocks:
            concat = block(concat)
        if self.context_gating:
            gate = nn.functional.sigmoid(self.context_weights(concat))
            concat = concat * gate
        return concat


class Noise(nn.Module):
    def __init__(self, var, type_, always_on):
        super().__init__()
        self.var = var
        self.always_on = always_on
        types = {'std': lambda x: x.std(),
                 'mean': lambda x: x.mean(),
                 'mean_abs': lambda x: x.abs().mean(),
                 'max_abs': lambda x: x.abs().max(),
                 'norm': lambda x: x.norm() / x.view(-1).shape[0],
                 'mean_square': lambda x: x * x / x.view(-1).shape[0]
                 }
        self.func = types[type_]

    def forward(self, x):
        if self.training or self.always_on:
            return x + torch.randn(x.shape).cuda() * self.var * self.func(x)
        else:
            return x


class ResNetLikeNoise(ResNetLike):
    class FCBlock(ResNetLike.FCBlock):
        def __init__(self, outer_size, inner_size, dropout, noise, no_bn, no_dropout, noise_place):
            super().__init__(outer_size, inner_size, dropout)
            self.noise = noise
            self.no_bn = no_bn
            self.no_dropout = no_dropout
            self.noise_place = noise_place

        def forward(self, x):
            x = self.fc1(x)

            if self.noise_place == 'before_bn':
                x = self.noise(x)

            if not self.no_bn:
                x = self.batch_norm(x)

            if self.noise_place == 'after_bn':
                x = self.noise(x)

            x = self.leaky_relu(x)

            if self.noise_place == 'before_dropout':
                x = self.noise(x)

            if not self.no_dropout:
                x = self.dropout(x)

            if self.noise_place == 'after_dropout':
                x = self.noise(x)

            return x

    class Identity(ResNetLike.Identity):
        def __init__(self, outer_size, inner_size, dropout, noise, no_bn, no_dropout, noise_place):
            super().__init__(outer_size, inner_size, dropout)
            self.fc1 = ResNetLikeNoise.FCBlock(outer_size, inner_size, dropout,
                                               noise, no_bn, no_dropout, noise_place)

    def __init__(self, noise_var, noise_type='std', noise_place='before_dropout', always_on=False,
                 no_bn=False, no_dropout=True, inner_size=2 * 1024, av_id_block_num=1,
                 video_features_sz=RGB_FEATURES_SIZE, audio_features_sz=AUDIO_FEATURES_SIZE,
                 concat_id_block_num=1, dropout=0.5, output_size=YOUTUBE8M_LABELS_N, context_gating=False):
        nn.Module.__init__(self)
        noise = Noise(noise_var, noise_type, always_on)

        self.context_gating = context_gating
        self.audio_blocks = nn.ModuleList(
                [self.FCBlock(audio_features_sz, inner_size, dropout, noise, no_bn, no_dropout, noise_place)] +
                [self.Identity(inner_size, inner_size, dropout, noise, no_bn, no_dropout, noise_place)
                    for i in range(av_id_block_num)])

        self.video_blocks = nn.ModuleList(
                [self.FCBlock(video_features_sz, inner_size, dropout, noise, no_bn, no_dropout, noise_place)] +
                [self.Identity(inner_size, inner_size, dropout, noise, no_bn, no_dropout, noise_place)
                    for i in range(av_id_block_num)])

        self.concat_blocks = nn.ModuleList(
                [self.Identity(inner_size * 2, inner_size, dropout, noise, no_bn, no_dropout, noise_place)
                    for i in range(concat_id_block_num)] +
                [self.FCBlock(inner_size * 2, inner_size, dropout, noise, no_bn, no_dropout, noise_place),
                    nn.Linear(inner_size, output_size)])
        if self.context_gating:
            self.context_weights = nn.Linear(YOUTUBE8M_LABELS_N, YOUTUBE8M_LABELS_N)


class AttentiveFCN(nn.Module):
    def __init__(self, inner_size=4096):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._activation = nn.ReLU()
        self._attention_layer = nn.Linear(inner_size, inner_size)
        self._attention_activation = nn.Sigmoid()
        self._layer2 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)

        weights = self._attention_activation(self._attention_layer(x))
        x = x * weights
        return self._layer2(x)


class AttentiveFCN2(nn.Module):
    def __init__(self, inner_size=4096, n_heads=5):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._activation = nn.ReLU()
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._layer3 = nn.Linear(inner_size, n_heads * YOUTUBE8M_LABELS_N)

        self._n_heads = n_heads

        self._attention_layer = nn.Linear(n_heads, n_heads)
        self._attention_activation = nn.Sigmoid()
        self._output = nn.Linear(n_heads, 1)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._layer2(x)
        x = self._activation(x)

        x = self._layer3(x)
        x = self._activation(x)
        x = x.view(-1, YOUTUBE8M_LABELS_N, self._n_heads)

        weights = self._attention_layer(x)
        # weights = weights.squeeze(2)
        weights = self._attention_activation(weights)
        # weights = weights.unsqueeze(2)
        x = x * weights

        x = self._output(x)
        return x.squeeze(2)


class AttentiveFCN3(nn.Module):
    def __init__(self, inner_size=4096, dropout=0.3):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout)
        self._attention_layer = nn.Linear(inner_size, inner_size)
        self._attention_activation = nn.Sigmoid()
        self._layer3 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._layer2(x)
        x = self._activation(x)

        weights = self._attention_activation(self._attention_layer(x))
        x = x * weights
        x = self._dropout(x)
        return self._layer3(x)


class BottleFCN1(nn.Module):
    def __init__(self, start_size=2048, layers_count=4):
        super().__init__()
        self._start_layer = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, start_size)
        layers = []
        for i in range(layers_count):
            layers.append(nn.Linear(start_size, start_size // 2))
            start_size = start_size // 2

        for i in range(layers_count):
            layers.append(nn.Linear(start_size, start_size * 2))
            start_size = start_size * 2

        self._layers = nn.ModuleList(layers)
        self._output = nn.Linear(start_size, YOUTUBE8M_LABELS_N)
        self._activation = nn.ReLU()

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._activation(self._start_layer(x))

        for layer in self._layers:
            x = self._activation(layer(x))

        return self._output(x)


class AttentiveFCN4(nn.Module):
    def __init__(self, inner_size=4096, dropout=0.3):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout)
        self._attention_layer = nn.Linear(inner_size, inner_size)
        self._attention_activation = nn.Softmax()
        self._layer3 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._layer2(x)
        x = self._activation(x)
        weights = self._attention_activation(self._attention_layer(x))
        x = x * weights
        x = self._dropout(x)
        return self._layer3(x)


class RankingFCN(nn.Module):
    def __init__(self, inner_size=2048):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._activation = nn.ReLU()
        self._ranking_layer = nn.Linear(1, 1)
        self._ranking_activation = nn.Softmax()
        self._layer3 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._layer2(x)
        x = self._activation(x)
        x = self._layer3(x)
        x = self._ranking_activation(x)
        x = x.unsqueeze(2)
        x = self._ranking_layer(x)
        x = x.squeeze(2)
        return x


class CyclicFCN(nn.Module):
    def __init__(self, inner_size=1024, length=3):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._activation = nn.LeakyReLU()
        self._length = length
        self._layer3 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)

        for _ in range(self._length):
            x = self._layer2(x)
            x = self._activation(x)
        x = self._layer3(x)
        return x


class AttentiveFCN5(nn.Module):
    def __init__(self, inner_size=4096, dropout=0.3):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout)
        self._attention_layer = nn.Linear(inner_size, inner_size)
        self._attention_activation = nn.Sigmoid()
        self._layer3 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x, _ = torch.sort(x, dim=1)
        x = self._layer2(x)
        x = self._activation(x)

        weights = self._attention_activation(self._attention_layer(x))
        x = x * weights
        x = self._dropout(x)
        return self._layer3(x)


class AttentiveFCN6(nn.Module):
    def __init__(self, inner_size=4096, dropout=0.3):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._layer4 = nn.Linear(inner_size, inner_size)

        self._bn1 = nn.BatchNorm1d(inner_size)
        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout)
        self._attention_layer = nn.Linear(inner_size, inner_size)
        self._attention_activation = nn.Sigmoid()
        self._layer3 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._layer2(x)
        x = self._bn1(x)
        x = self._activation(x)
        x = self._layer4(x) + x
        x = self._activation(x)

        weights = self._attention_activation(self._attention_layer(x))
        x = x * weights
        x = self._dropout(x)
        return self._layer3(x)


class AttentiveFCN7(nn.Module):
    def __init__(self, inner_size=4096, dropout=0.3):
        super().__init__()
        self._layer1 = nn.Linear(RGB_FEATURES_SIZE + AUDIO_FEATURES_SIZE, inner_size)
        self._layer2 = nn.Linear(inner_size, inner_size)
        self._layer4 = nn.Linear(inner_size, inner_size)
        self._layer5 = nn.Linear(inner_size, inner_size)

        self._bn1 = nn.BatchNorm1d(inner_size)
        self._bn2 = nn.BatchNorm1d(inner_size)
        self._bn3 = nn.BatchNorm1d(inner_size)

        self._activation = nn.ReLU()
        self._dropout = nn.Dropout(dropout)
        self._attention_layer = nn.Linear(inner_size, inner_size)
        self._attention_activation = nn.Sigmoid()
        self._layer3 = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = self._layer1(x)
        x = self._activation(x)
        x = self._layer2(x)
        x = self._bn1(x)
        x = self._activation(x)
        x = self._layer4(x) + x
        x = self._bn2(x)
        x = self._activation(x)
        x = self._layer5(x) + x
        x = self._bn3(x)
        x = self._activation(x)

        weights = self._attention_activation(self._attention_layer(x))
        x = x * weights
        x = self._dropout(x)
        return self._layer3(x)


class MondayFCN1(nn.Module):
    def __init__(self, inner_size=1024, bil_size=32):
        super().__init__()
        self._layer_1 = nn.Linear(512 + 128, bil_size)
        self._layer_2 = nn.Bilinear(512 + 128, bil_size, inner_size)

        self._activation = nn.Tanhshrink()
        self._output = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        video = video[:, :512]
        features = torch.cat([video, audio], dim=1)
        x = self._layer_1(features)
        x = self._activation(x)
        x = self._layer_2(features, x)
        x = self._activation(x)
        return self._output(x)


class MondayFCN2(nn.Module):
    def __init__(self, inner_size=1024, bil_size=32):
        super().__init__()
        self._layer_1 = nn.Linear(1024 + 128, bil_size)
        self._layer_2 = nn.Bilinear(1024 + 128, bil_size, inner_size)
        self._layer_3 = nn.Bilinear(inner_size, bil_size, inner_size)

        self._activation = nn.Tanhshrink()
        self._output = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        features = torch.cat([video, audio], dim=1)
        x1 = self._layer_1(features)
        x = self._activation(x1)
        x = self._layer_2(features, x)
        x = self._activation(x)
        x = self._layer_3(x, x1)
        x = self._activation(x)
        return self._output(x)


class MondayFCN3(nn.Module):
    def __init__(self, layers=2, hidden_size=1024, length=10):
        super().__init__()
        self._length = length
        self._gru_layer = nn.GRU(input_size=1024 + 128, num_layers=layers, hidden_size=hidden_size, batch_first=True)
        self._output = nn.Linear(hidden_size * layers, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, self._length)
        x = x.permute(0, 2, 1)
        _, state = self._gru_layer(x)
        state = state.permute(1, 0, 2).contiguous()
        state = state.view(-1, state.shape[1] * state.shape[2])
        return self._output(state)


class MondayFCN4(nn.Module):
    def __init__(self, layers=2, hidden_size=1024, length=10):
        super().__init__()
        self._length = length
        self._lstm_layer = nn.LSTM(input_size=1024 + 128, num_layers=layers, hidden_size=hidden_size, batch_first=True)
        self._output = nn.Linear(hidden_size * layers, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        x = torch.cat([video, audio], dim=1)
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, self._length)
        x = x.permute(0, 2, 1)
        _, (state, _) = self._lstm_layer(x)
        state = state.permute(1, 0, 2).contiguous()
        state = state.view(-1, state.shape[1] * state.shape[2])
        return self._output(state)


class MondayFCN5(nn.Module):
    def __init__(self, inner_size=1024, bil_size=32, dropout=0.3):
        super().__init__()
        self._layer_1 = nn.Linear(1024 + 128, bil_size)
        self._layer_2 = nn.Bilinear(1024 + 128, bil_size, inner_size)
        self._dropout = nn.Dropout(dropout)

        self._activation = nn.Tanhshrink()
        self._output = nn.Linear(inner_size, YOUTUBE8M_LABELS_N)

    def forward(self, video, audio):
        features = torch.cat([video, audio], dim=1)
        x = self._layer_1(features)
        x = self._activation(x)
        x = self._layer_2(features, x)
        x = self._activation(x)
        x = self._dropout(x)
        return self._output(x)


class FFHierarchy(nn.Module):
    def __init__(self, in_size, levels, levels_order):
        super().__init__()
        sizes = [in_size]
        for level_size in levels[:-1]:
            sizes.append(sizes[-1] + level_size)

        self._projections = nn.ModuleList([nn.Linear(in_level_size, out_level_size)
                                           for in_level_size, out_level_size in zip(sizes, levels)])
        self._levels_order = levels_order

    def forward(self, x):
        outputs = []
        for proj in self._projections:
            cur_out = proj(x)
            outputs.append(cur_out)
            x = torch.cat([x, cur_out], dim=1)
        outputs_sorted = [outputs[i] for i in self._levels_order]
        return torch.cat(outputs_sorted, dim=1)


class ResNetWithFFHierarchy(nn.Module):
    def __init__(self, inner_size=2000, inner_act=nn.ReLU, resnet_kwargs={}, ff_hier_kwargs={}):
        super().__init__()
        self._resnet = ResNetLike(output_size=inner_size, **resnet_kwargs)
        self._inner_act = inner_act()
        self._hier = FFHierarchy(inner_size, **ff_hier_kwargs)

    def forward(self, video, audio):
        x = self._resnet(video, audio)
        x = self._inner_act(x)
        return self._hier(x)


class RNNHierarchy(nn.Module):
    """
    Something similar to https://www.sfu.ca/~nnauata/BINN_CVPR_Workshop_2017.pdf
    """
    def __init__(self, in_size, model_size, levels, levels_order, rnn_class='torch.nn.RNN', layers_n=1):
        super().__init__()
        self._rnn = load_class(rnn_class)(input_size=in_size, hidden_size=model_size, num_layers=layers_n,
                                          batch_first=True, bidirectional=True)
        self._out = nn.ModuleList([nn.Linear(2*model_size, level_size) for level_size in levels])
        self._levels_order = levels_order

    def forward(self, x):
        """
        :param x: BatchSize x InSize
        :return: BatchSize x InSize
        """
        x_as_seq = x.unsqueeze(1).expand(-1, len(self._levels_order), -1)  # BatchSize x LevelsN x InSize
        model_repr, _ = self._rnn(x_as_seq)  # BatchSize x LevelsN x 2*ModelSize
        outputs = [level_proj(model_repr[:, layer_i]) for layer_i, level_proj in enumerate(self._out)]
        outputs_sorted = [outputs[i] for i in self._levels_order]
        return torch.cat(outputs_sorted, dim=1)


class ResNetWithRNNHierarchy(nn.Module):
    def __init__(self, inner_size=2000, inner_act=nn.ReLU, resnet_kwargs={}, rnn_hier_kwargs={}):
        super().__init__()
        self._resnet = ResNetLike(labels_n=inner_size, **resnet_kwargs)
        self._inner_act = inner_act()
        self._hier = RNNHierarchy(inner_size, **rnn_hier_kwargs)

    def forward(self, video, audio):
        x = self._resnet(video, audio)
        x = self._inner_act(x)
        return self._hier(x)


class RepRNNHierarchy(nn.Module):
    def __init__(self, in_size, model_size, levels, levels_order, rnn_class='torch.nn.RNN', layers_n=1):
        super().__init__()
        self._input = nn.Linear(in_size, model_size)
        self._rnn = load_class(rnn_class)(input_size=model_size, hidden_size=model_size // 2, num_layers=1,
                                          batch_first=True, bidirectional=True)
        self._layers_n = layers_n
        self._out = nn.ModuleList([nn.Linear(model_size, level_size) for level_size in levels])
        self._levels_order = levels_order

    def forward(self, x):
        """
        :param x: BatchSize x InSize
        :return: BatchSize x InSize
        """
        x = self._input(x)  # BatchSize x ModelSize
        model_repr = x.unsqueeze(1).expand(-1, len(self._levels_order), -1)  # BatchSize x LevelsN x ModelSize
        for _ in range(self._layers_n):
            model_repr, _ = self._rnn(model_repr)  # BatchSize x LevelsN x 2*ModelSize
        outputs = [level_proj(model_repr[:, layer_i]) for layer_i, level_proj in enumerate(self._out)]
        outputs_sorted = [outputs[i] for i in self._levels_order]
        return torch.cat(outputs_sorted, dim=1)


class ResNetWithRepRNNHierarchy(nn.Module):
    def __init__(self, inner_size=2000, inner_act=nn.ReLU, resnet_kwargs={}, rnn_hier_kwargs={}):
        super().__init__()
        self._resnet = ResNetLike(labels_n=inner_size, **resnet_kwargs)
        self._inner_act = inner_act()
        self._hier = RepRNNHierarchy(inner_size, **rnn_hier_kwargs)

    def forward(self, video, audio):
        x = self._resnet(video, audio)
        x = self._inner_act(x)
        return self._hier(x)
