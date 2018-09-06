import numpy as np
import torch


class DataLoaderWithMixUp(object):
    def __init__(self, data_loader, alpha=0.2):
        self._data_loader = data_loader
        self._alpha = alpha

    def __iter__(self):
        previous_input_batch = None
        previous_output_batch = None

        for input_batch, output_batch in self._data_loader:
            if previous_input_batch is None:
                previous_input_batch = input_batch
                previous_output_batch = output_batch
                continue

            assert previous_output_batch.shape == output_batch.shape

            coefficients = torch.from_numpy(np.random.beta(self._alpha,
                                                           self._alpha,
                                                           size=output_batch.shape[0]).astype("float32"))
            if input_batch[0].is_cuda:
                coefficients = coefficients.cuda()

            mixed_input_batch = []

            for previous_modality, current_modality in zip(previous_input_batch, input_batch):
                coef_size = [-1] + [1 for _ in range(len(previous_modality.size()) - 1)]
                cur_mod_coef = coefficients.view(*coef_size).expand_as(previous_modality)
                mixed_modality = cur_mod_coef * previous_modality + (1 - cur_mod_coef) * current_modality
                mixed_input_batch.append(mixed_modality)

            output_coef = coefficients.unsqueeze(-1).expand_as(previous_output_batch)
            mixed_output_batch = output_coef * previous_output_batch + (1 - output_coef) * output_batch

            previous_input_batch = input_batch
            previous_output_batch = output_batch

            yield mixed_input_batch, mixed_output_batch
