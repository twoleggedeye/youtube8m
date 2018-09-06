#!/usr/bin/env python

import argparse
import horovod.torch as hvd
import os
import shutil
import torch

from youtube8m.utils.io import load_yaml, ensure_dir_exists
from youtube8m.utils.logger import setup_logger
from youtube8m.utils.training import train_folds


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--config', type=str, required=True, help='Config')
    aparser.add_argument('--out', type=str, required=True, help='Where to store results')
    aparser.add_argument('--load', type=str, default='', help='Directory to load model from')
    aparser.add_argument('--folds_ids', type=str, help='Folds to learn on separated by comma')

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    args = aparser.parse_args()

    if args.folds_ids is None:
        folds_ids = None
    else:
        folds_ids = [int(fold) for fold in args.folds_ids.split(',')]

    if hvd.rank() == 0:
        ensure_dir_exists(args.out)
        shutil.copy2(args.config, os.path.join(args.out, 'config.yaml'))

    config = load_yaml(args.config)

    logger_out_file = None
    stderr = False
    if hvd.rank() == 0:
        stderr = True
        logger_out_file = os.path.join(args.out, 'global.log')

    logger = setup_logger(stderr=stderr, out_file=logger_out_file)
    logger.info('Loaded config from {}'.format(args.config))
    logger.info('Writing results to {}'.format(args.out))

    train_folds(config, args.out, logger=logger, use_horovod=True, load=args.load, folds_ids=folds_ids)


if __name__ == '__main__':
    main()
