#!/usr/bin/env python

import argparse
import os
import shutil

from youtube8m.utils.io import load_yaml, ensure_dir_exists
from youtube8m.utils.logger import setup_logger
from youtube8m.utils.training import train_folds, predict


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--config', type=str, required=True, help='Config')
    aparser.add_argument('--out', type=str, required=True, help='Where to store results')
    aparser.add_argument('--load', type=str, default='', help='Directory to load model from')
    aparser.add_argument('--folds_ids', type=str, help='Folds to learn on separated by comma')

    commands = aparser.add_subparsers(dest='command', help='What to do')

    fit_predict_parser = commands.add_parser('fit_predict')

    predict_parser = commands.add_parser('predict')
    predict_parser.add_argument('--model', type=str, help='Path to serialized model (to override model from config)')

    args = aparser.parse_args()

    if args.folds_ids is None:
        folds_ids = None
    else:
        folds_ids = [int(fold) for fold in args.folds_ids.split(',')]

    ensure_dir_exists(args.out)
    shutil.copy2(args.config, os.path.join(args.out, 'config.yaml'))

    config = load_yaml(args.config)

    logger = setup_logger(out_file=os.path.join(args.out, 'global.log'))
    logger.info('Loaded config from {}'.format(args.config))
    logger.info('Writing results to {}'.format(args.out))

    if args.command == 'fit_predict':
        train_folds(config, args.out, logger=logger, load=args.load, folds_ids=folds_ids)

    elif args.command == 'predict':
        if args.model:
            config['model'] = args.model
        predict(config, args.out, logger=logger)

    else:
        logger.info('No command specified')


if __name__ == '__main__':
    main()
