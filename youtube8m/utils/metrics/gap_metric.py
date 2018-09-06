import numpy as np


class GAPScorer(object):
    def __init__(self, top_k=20):
        self.scores_list = []
        self.targets_list = []
        self.top_k = top_k
        self.n_pos = 0

    def add(self, pred, target):
        top_idxs = np.argsort(pred, axis=1)[:, ::-1][:, :self.top_k]
        self.n_pos += target.sum()
        n_repeats = min(self.top_k, top_idxs.shape[1])
        rows_idxs = np.arange(len(top_idxs))[..., np.newaxis].repeat(n_repeats, axis=1)
        top_targets = target[rows_idxs, top_idxs]
        top_preds = pred[rows_idxs, top_idxs]
        self.scores_list.append(top_preds.reshape(-1))
        self.targets_list.append(top_targets.reshape(-1))

    @property
    def current_value(self):
        scores_arr = np.concatenate(self.scores_list)
        targets_arr = np.concatenate(self.targets_list)
        idxs = np.argsort(scores_arr)[::-1]
        sorted_targets = targets_arr[idxs]
        ranks = np.arange(len(idxs)) + 1
        n_pos_at_rank = np.cumsum(sorted_targets)
        AP = (n_pos_at_rank * sorted_targets / ranks).sum() / self.n_pos
        return AP


class GAPScorerFilteredColumns(GAPScorer):
    def __init__(self, from_col, to_col, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_col = from_col
        self._to_col = to_col

    def add(self, pred, target):
        filtered_pred = pred[:, self._from_col:self._to_col]
        filtered_target = target[:, self._from_col:self._to_col]
        super().add(filtered_pred, filtered_target)
