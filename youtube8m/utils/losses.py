import torch
from torch.nn import functional as F

from youtube8m.video_level_nn_models.defaults import YOUTUBE8M_LABELS_N


def soft_ranking_example_loss(y_pred, y_target, top_neg_count=30, margin=1):
    inf_score = 100000
    batch_sz = y_pred.shape[0]
    y_target = y_target.type(torch.cuda.ByteTensor)
    min_score = y_pred.min()
    max_npos = y_target.sum(dim=1).max()
    pos_scores = torch.zeros_like(y_pred) + inf_score
    neg_scores = torch.ones_like(y_pred) * (min_score - 1)
    pos_scores[y_target] = y_pred[y_target]
    neg_scores[1 - y_target] = y_pred[1 - y_target]
    valid_pos, _ = torch.topk(pos_scores, max_npos, dim=1, largest=False)
    valid_neg, _ = torch.topk(neg_scores, top_neg_count, dim=1, largest=True)
    diff = valid_neg.view(batch_sz, 1, top_neg_count) - valid_pos.view(batch_sz, max_npos, 1)
    return torch.nn.functional.softplus(diff + margin).mean()


def _get_pos_neg_scores_diff(y_pred, y_target, top_neg_count):
    batch_sz = y_target.shape[0]
    y_pred = y_pred.view(-1)
    y_target = y_target.view(-1)
    y_target = (y_target > 0).type(torch.cuda.ByteTensor)
    positive_predictions = torch.masked_select(y_pred, y_target)
    negative_predictions = torch.masked_select(y_pred, 1 - y_target)
    top_neg, _ = torch.topk(negative_predictions, top_neg_count * batch_sz)
    diff = top_neg - positive_predictions.view(-1, 1)
    return diff


def _get_full_pos_neg_scores_diff(y_pred, y_target, top_neg_count):
    batch_sz = y_target.shape[0]
    y_pred = y_pred.view(-1)
    y_target = y_target.view(-1)
    y_target_binary = (y_target > 0).type(torch.cuda.ByteTensor)

    positive_predictions = torch.masked_select(y_pred, y_target_binary)
    negative_predictions = torch.masked_select(y_pred, 1 - y_target_binary)

    positive_targets = torch.masked_select(y_target, y_target_binary)
    negative_targets = torch.masked_select(y_target, 1 - y_target_binary)
    top_neg, neg_idxs = torch.topk(negative_predictions, top_neg_count * batch_sz)
    scores = torch.cat([positive_predictions, top_neg], dim=0)
    target = torch.cat([positive_targets, negative_targets[neg_idxs]], dim=0)
    pairwise_diff = scores - scores.view(-1, 1)
    relation_mask = (target < target.view(-1, 1)).type(torch.cuda.FloatTensor)
    misranked_diffs = pairwise_diff * relation_mask
    return misranked_diffs


def soft_ranking_loss(y_pred, y_target, top_neg_count=30, margin=1):
    diff = _get_pos_neg_scores_diff(y_pred, y_target, top_neg_count)
    #diff = _get_full_pos_neg_scores_diff(y_pred, y_target, top_neg_count)
    nonz = torch.nonzero(diff)
    soft_diff = torch.nn.functional.softplus(diff[nonz[:, 0], nonz[:, 1]] + margin)
    return torch.mean(soft_diff)


def bce_with_soft_rank_with_logits(y_pred, y_target, alpha=0.1):   
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_target)
    loss = loss + alpha * soft_ranking_loss(y_pred, y_target)
    return loss


def sigmoid_ranking_loss_with_logits(y_pred, y_target, top_neg_count=30):
    diff = _get_pos_neg_scores_diff(y_pred, y_target, top_neg_count)
    sigmoid_diff = torch.nn.functional.sigmoid(diff)
    return torch.mean(torch.log(1 + sigmoid_diff))


def lsep_loss(y_pred, y_target, top_neg_count=30):
    diff = _get_pos_neg_scores_diff(y_pred, y_target, top_neg_count)
    lsep = torch.log1p(torch.exp(diff).mean())
    return lsep


def _get_inverse_idx(tensor):
    idx = torch.arange(start=tensor.size(0) - 1, end=-1, step=-1).type(torch.cuda.LongTensor)
    return idx


def compute_rank_weights(y_pred, y_target, inverse=False):
    init_shape = y_pred.shape
    _, argsort = torch.sort(y_pred.view(-1), descending=True)
    _, inverse_argsort = torch.sort(argsort)
    y_target = y_target.type(torch.cuda.ByteTensor).detach() 
    sorted_target = y_target.view(-1)[argsort]
    idxs_range = torch.arange(len(sorted_target)).type(torch.cuda.LongTensor)
    float_target = sorted_target.type(torch.cuda.FloatTensor)
    inverse_idx = _get_inverse_idx(float_target)
    n_pos_after_rank = float_target[inverse_idx].cumsum(0)[inverse_idx]
    n_neg_at_rank = (1 - float_target).cumsum(0)
    pos_idxs = torch.masked_select(idxs_range, sorted_target)
    neg_idxs = torch.masked_select(idxs_range, 1 - sorted_target)
    weights = torch.ones_like(y_pred).view(-1).cuda()
    weights[neg_idxs] = torch.max(weights[neg_idxs], n_pos_after_rank[neg_idxs])
    weights[pos_idxs] = torch.max(n_neg_at_rank[pos_idxs], weights[pos_idxs])
    #weights = weights[inverse_argsort]
    if inverse:
        weights = 1. / weights
    return weights


def weighted_bce_with_logits(y_pred, y_target):
    weights = compute_rank_weights(y_pred, y_target)
    weights.requires_grad = False
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_target, weight=weights)


def bce_with_logits_poisson(y_pred, y_target, coef=0.0001):
   n_predicted = torch.nn.functional.sigmoid(y_pred).sum(dim=1)
   n_actual = y_target.sum(dim=1)
   pois = torch.nn.functional.poisson_nll_loss(n_predicted, n_actual, log_input=False)
   bce = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_target)
   return pois * coef + bce

class SoftRankingLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_target, top_neg_count=30):
        return soft_ranking_loss(y_pred, y_target, top_neg_count).view(1, 1)


class SigmoidRankingLossWithLogits(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_target, top_neg_count=30):
        return sigmoid_ranking_loss_with_logits(y_pred, y_target, top_neg_count).view(1, 1)


class ParallelWrapper(torch.nn.Module):
    def __init__(self, criterion_module):
        super().__init__()
        self._criterion = torch.nn.DataParallel(criterion_module)

    def forward(self, y_pred, y_target, **kwargs):
        return self._criterion(y_pred, y_target, **kwargs).mean()


parallel_soft_ranking_loss = ParallelWrapper(SoftRankingLoss())
parallel_sigmoid_ranking_loss_with_logits = ParallelWrapper(SigmoidRankingLossWithLogits())


def bce_with_logits_labels_tags(pred, target, labels_weight=1, tags_weight=1):
    labels_pred, labels_target = pred[:, :YOUTUBE8M_LABELS_N], target[:, :YOUTUBE8M_LABELS_N]
    tags_pred, tags_target = pred[:, YOUTUBE8M_LABELS_N:], target[:, YOUTUBE8M_LABELS_N:]
    return (labels_weight * F.binary_cross_entropy_with_logits(labels_pred, labels_target)
            + tags_weight * F.binary_cross_entropy_with_logits(tags_pred, tags_target))


def bce_with_logits_labels1_tags100(pred, target):
    return bce_with_logits_labels_tags(pred, target, labels_weight=1, tags_weight=100)


def bce_with_logits_labels1_tags20(pred, target):
    return bce_with_logits_labels_tags(pred, target, labels_weight=1, tags_weight=20)


def bce_with_logits_labels1_tags2(pred, target):
    return bce_with_logits_labels_tags(pred, target, labels_weight=1, tags_weight=2)
