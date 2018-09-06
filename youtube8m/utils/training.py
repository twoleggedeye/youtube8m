import collections
import glob

try:
    import horovod.torch as hvd
    HOROVOD_INSTALLED = True
except ImportError:
    HOROVOD_INSTALLED = False

import logging
import os
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from youtube8m.utils.io import save_model, load_class, save_pickle, ensure_dir_exists, \
    move_data_to_device, save_yaml, load_model, copy_model
from youtube8m.utils.metrics.gap_metric import GAPScorer, GAPScorerFilteredColumns
from youtube8m.video_level_nn_models.chunked_dataset import make_video_audio_labels_dataset, ChunkedWriter
from youtube8m.video_level_nn_models.defaults import YOUTUBE8M_LABELS_N
from youtube8m.utils.parallel_data_loader import ParallelDataLoader


_LOGGER = logging.getLogger()


def wrap_metric_for_pytorch(metric_func, *args, **kwargs):
    @functools.wraps(metric_func)
    def _impl(pred, target):
        pred = pred.cpu().data.numpy()
        target = target.cpu().data.numpy()
        return metric_func(pred, target, *args, **kwargs)
    return _impl


FAST_METRICS = {}


FULL_METRICS = {
    'GAP': GAPScorer
}


def gap_only_labels(from_col=0, to_col=YOUTUBE8M_LABELS_N):
    return GAPScorerFilteredColumns(from_col, to_col)


FULL_HIER_METRICS = {
    'GAP': gap_only_labels
}


def mean_over_dicts(dicts):
    result = collections.defaultdict(float)
    norm = 0
    for dct in dicts:
        for key, value in dct.items():
            result[key] += value
        norm += 1
    return {k: v / norm for k, v in result.items()}


def get_best_values(dicts, criterion='loss'):
    result = None
    best_value = np.inf
    for dct in dicts:
        cur_loss = dct[criterion]
        if cur_loss < best_value:
            best_value = cur_loss
            result = dct
    return result


def get_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.cpu().data.numpy())
    return float(x)


def run_epoch(network, batch_gen, criterion=F.binary_cross_entropy, optimizer=None, metrics_def=FAST_METRICS,
              max_batches=None, grad_clip=10, logger=_LOGGER, cuda=True, predictions_storage=None,
              print_metrics_frequency=1000, use_horovod=False):
    if use_horovod:
        assert HOROVOD_INSTALLED, "It seems like horovod is requested, but is not installed"

    metrics = []

    if 'GAP' in metrics_def:
        gap_scorer = metrics_def['GAP']()
    else:
        gap_scorer = None

    for batch_i, (inputs, targets) in enumerate(batch_gen):
        if max_batches is not None and batch_i >= max_batches:
            break

        if print_metrics_frequency is not None and batch_i % print_metrics_frequency == 0:
            logger.info('Batch #{}'.format(batch_i))

        inputs = move_data_to_device(inputs, cuda)
        targets = move_data_to_device(targets, cuda)

        pred = network(*inputs)

        if gap_scorer is not None:
            gap_scorer.add(pred.cpu().data.numpy(), targets.cpu().data.numpy())

        if criterion is not None:
            loss_value = criterion(pred, targets)

            if optimizer:
                optimizer.zero_grad()
                loss_value.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
                optimizer.step()

            cur_metrics = dict(loss=float(loss_value.cpu().data.numpy()))
            if not use_horovod or hvd.rank() == 0:
                for metric_name, metric_func in metrics_def.items():
                    if metric_name == 'GAP':
                        continue
                    cur_metrics[metric_name] = get_float(metric_func(pred, targets))

                if print_metrics_frequency is not None and batch_i % print_metrics_frequency == 0:
                    logger.info('Metrics after batch #{}: {}'.format(batch_i, cur_metrics))

            metrics.append(cur_metrics)

        if predictions_storage:
            predictions_storage.append(pred.cpu().data.numpy())

    mean_metrics = mean_over_dicts(metrics)

    if gap_scorer is not None:
        mean_metrics['GAP'] = gap_scorer.current_value

    return mean_metrics


DEFAULT_LR_SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau
DEFAULT_LR_SCHEDULER_KWARGS = dict(factor=0.3)

DEFAULT_OPTIMIZER = torch.optim.Adam
DEFAULT_OPTIMIZER_KWARGS = dict(lr=1e-3)


def train_loop(network, train_batch_gen, val_batch_gen=None, criterion=F.binary_cross_entropy,
               optimizer_ctor=DEFAULT_OPTIMIZER, optimizer_kwargs=DEFAULT_OPTIMIZER_KWARGS,
               epochs=20, lr_scheduler_ctor=DEFAULT_LR_SCHEDULER, lr_scheduler_kwargs=DEFAULT_LR_SCHEDULER_KWARGS,
               max_batches_per_epoch=None, train_metrics_def=FAST_METRICS, val_metrics_def=FULL_METRICS,
               early_stopping_patience=5, logger=_LOGGER, use_horovod=False, cuda=True,
               last_model_fname=None, best_model_fname=None, metrics_fname=None, val_predictions_fname=None,
               print_metrics_frequency=100, max_batches_per_val=None):

    optimizer = optimizer_ctor([p for p in network.parameters() if p.requires_grad],
                               **optimizer_kwargs)

    if use_horovod:
        assert HOROVOD_INSTALLED, "It seems like horovod is requested, but is not installed"

        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=network.named_parameters())
        hvd.broadcast_parameters(network.state_dict(), root_rank=0)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer, **lr_scheduler_kwargs)
    else:
        lr_scheduler = None

    original_network = network

    if cuda:
        if not use_horovod:
            network = nn.DataParallel(original_network)
        network.cuda()
    else:
        network.cpu()

    train_metrics = []
    val_metrics = []
    best_val_loss = np.inf
    best_val_loss_epoch = 0

    for epoch_i in range(epochs):
        try:
            logger.info('Epoch #{} training started'.format(epoch_i))
            network.train()
            cur_train_metrics = run_epoch(network, train_batch_gen, criterion=criterion,
                                          optimizer=optimizer, max_batches=max_batches_per_epoch,
                                          metrics_def=train_metrics_def, logger=logger, cuda=cuda,
                                          print_metrics_frequency=print_metrics_frequency, use_horovod=use_horovod)
            train_metrics.append(cur_train_metrics)
            logger.info('Mean train metrics after epoch #{}: {}'.format(epoch_i, cur_train_metrics))

            if use_horovod:
                hvd.broadcast_parameters(network.state_dict(), root_rank=0)

            if val_batch_gen is not None:
                logger.info('Epoch #{} validation started'.format(epoch_i))
                network.eval()
                with torch.no_grad():
                    cur_val_metrics = run_epoch(network, val_batch_gen, criterion=criterion,
                                                max_batches=max_batches_per_val, cuda=cuda,
                                                logger=logger, metrics_def=val_metrics_def,
                                                print_metrics_frequency=print_metrics_frequency, use_horovod=use_horovod)
                val_metrics.append(cur_val_metrics)
                logger.info('Mean validation metrics after epoch #{}: {}'.format(epoch_i, cur_val_metrics))

                if 'GAP' in val_metrics_def:
                    cur_val_loss = 1. - cur_val_metrics['GAP']
                else:
                    cur_val_loss = cur_val_metrics['loss']

                if best_model_fname and cur_val_loss < best_val_loss:
                    if not use_horovod or hvd.rank() == 0:
                        save_model(original_network, best_model_fname, val_loss=cur_val_loss, epoch=epoch_i)
                    best_val_loss = cur_val_loss
                    best_val_loss_epoch = epoch_i
                    logger.info('New best val loss {} on epoch #{}'.format(best_val_loss, epoch_i))

                if lr_scheduler:
                    lr_scheduler.step(cur_val_loss, epoch=epoch_i)

                if epoch_i - best_val_loss_epoch > early_stopping_patience:
                    logger.info('Early stopping on epoch {} after {} epochs '
                                'with no improvement over best {} validation loss'.format(epoch_i,
                                                                                          epoch_i - best_val_loss_epoch,
                                                                                          best_val_loss))
                    break

            if metrics_fname and (not use_horovod or hvd.rank() == 0):
                save_pickle(train_metrics, metrics_fname + '.train.pickle')
                save_pickle(val_metrics, metrics_fname + '.val.pickle')

        except KeyboardInterrupt:
            break

    if last_model_fname and (not use_horovod or hvd.rank() == 0):
        save_model(original_network, last_model_fname)

    if val_batch_gen is not None and val_predictions_fname and not use_horovod:
        logger.info('Predicting with latest model')
        network.eval()
        with ChunkedWriter(val_predictions_fname) as predictions_storage:
            run_epoch(network, val_batch_gen, criterion=criterion, max_batches=max_batches_per_val,
                      cuda=cuda, predictions_storage=predictions_storage)

    return train_metrics, val_metrics


def train_folds(config, out_dir, logger=_LOGGER, use_horovod=False, load='', folds_ids=None):
    all_folds = list(range(len(list(glob.glob(os.path.join(config['src_data'], 'fold_*'))))))
    if folds_ids is None:
        folds_ids = all_folds
    best_train_metrics = []
    best_val_metrics = []

    train_kwargs = config.get('train_kwargs', {})

    if 'lr_scheduler_ctor' in train_kwargs:
        train_kwargs['lr_scheduler_ctor'] = load_class(train_kwargs['lr_scheduler_ctor'])

    if 'optimizer_ctor' in train_kwargs:
        train_kwargs['optimizer_ctor'] = load_class(train_kwargs['optimizer_ctor'])

    if 'train_metrics_def' in train_kwargs:
        train_kwargs['train_metrics_def'] = load_class(train_kwargs['train_metrics_def'])

    if 'val_metrics_def' in train_kwargs:
        train_kwargs['val_metrics_def'] = load_class(train_kwargs['val_metrics_def'])

    if 'criterion' in train_kwargs:
        train_kwargs['criterion'] = load_class(train_kwargs['criterion'])

    dataset_reader_kwargs = config.get('dataset_reader_kwargs', {})
    train_reader_kwargs = config.get('train_dataset_reader_kwargs', dataset_reader_kwargs)
    val_reader_kwargs = config.get('val_dataset_reader_kwargs', dataset_reader_kwargs)

    dataset_reader_ctor = load_class(config['dataset_reader']) if 'dataset_reader' in config \
        else make_video_audio_labels_dataset

    train_dataset_reader_ctor = load_class(config['train_dataset_reader']) if 'train_dataset_reader' in config \
        else dataset_reader_ctor
    val_dataset_reader_ctor = load_class(config['val_dataset_reader']) if 'val_dataset_reader' in config \
        else dataset_reader_ctor

    for fold_i in folds_ids:
        logger.info('Fold #{} started'.format(fold_i))
        train_folds = all_folds[:fold_i] + all_folds[fold_i + 1:]
        logger.info('Train folds: {}'.format(train_folds))
        train_folds_glob = os.path.join(config['src_data'],
                                        'fold_[{}]'.format(''.join(map(str, train_folds))))
        logger.info('Train folds glob: {}'.format(train_folds_glob))

        enable_mixup = "mixup_class" in config
        train_data = train_dataset_reader_ctor(train_folds_glob, batch_size=config['batch_size'],
                                               drop_small_batch=enable_mixup, **train_reader_kwargs)

        logger.info('Val folds: {}'.format(fold_i))
        val_fold = os.path.join(config['src_data'], 'fold_{}'.format(fold_i))
        val_data = val_dataset_reader_ctor(val_fold, batch_size=config['batch_size'], shuffle=False,
                                           **val_reader_kwargs)

        if enable_mixup:
            train_data = load_class(config['mixup_class'])(train_data, **config.get("mixup_args", {}))

        if config.get("use_parallel_dataloader"):
            train_data = ParallelDataLoader(train_data, **config.get("parallel_dataloader_train_kwargs", {}))

        network = load_class(config['model'])(**config.get('model_kwargs', {}))

        model_path = os.path.join(load, 'folds', 'fold_{}'.format(fold_i), 'best.model')
        if load and os.path.exists(model_path):
            network.load_state_dict(torch.load(model_path))
            logger.info('Loaded model from {}'.format(model_path))
        else:
            logger.info('Created new model')

        fold_out_dir = os.path.join(out_dir, 'folds', 'fold_{}'.format(fold_i))
        if not use_horovod or hvd.rank() == 0:
            ensure_dir_exists(fold_out_dir)

        train_metrics, val_metrics = train_loop(network,
                                                train_data, val_batch_gen=val_data,
                                                last_model_fname=os.path.join(fold_out_dir, 'last.model'),
                                                best_model_fname=os.path.join(fold_out_dir, 'best.model'),
                                                metrics_fname=os.path.join(fold_out_dir, 'metrics'),
                                                val_predictions_fname=os.path.join(fold_out_dir, 'val_predictions'),
                                                logger=logger, use_horovod=use_horovod, **train_kwargs)
        best_train_metrics.append(get_best_values(train_metrics))
        best_val_metrics.append(get_best_values(val_metrics))

    if not use_horovod or hvd.rank() == 0:
        save_yaml(dict(train_metrics=mean_over_dicts(best_train_metrics),
                       val_metrics=mean_over_dicts(best_val_metrics)),
                  os.path.join(out_dir, 'average_best_metrics.yaml'))


def predict(config, out_dir, logger=_LOGGER):
    dataset_reader_ctor = load_class(config['dataset_reader']) if 'dataset_reader' in config \
        else make_video_audio_labels_dataset
    batch_gen = dataset_reader_ctor(config['src_data'], batch_size=config['batch_size'])

    network = load_model(config['model'], config.get('model_kwargs', {}))
    logger.info('Model loaded')
    if config.get('cuda', True):
        network.cuda()
        network = nn.DataParallel(network)
    else:
        network.cpu()
    network.eval()

    copy_model(config['model'], os.path.join(out_dir, 'model'))

    with torch.no_grad(), ChunkedWriter(os.path.join(out_dir, 'predictions')) as predictions_storage:
        run_epoch(network, batch_gen, criterion=None, logger=logger,
                  predictions_storage=predictions_storage, **config.get('run_kwargs', {}))
