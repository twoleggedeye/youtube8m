#!/usr/bin/env python3.6

from youtube8m.utils.tfrecord_utils import read_tfrecord_and_save_as_numpy

import argparse
import joblib
import os


def main():
    parser = argparse.ArgumentParser("This script converts source tf records to numpy arrays and saves them")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--is-test", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    files = os.listdir(args.input_path)
    files = filter(lambda name: name.endswith(".tfrecord"), files)

    input_output_paths = []

    for file_name in files:
        tfrecord_path = os.path.join(args.input_path, file_name)
        output_path = os.path.join(args.output_path, file_name.replace(".tfrecord", ""))
        input_output_paths.append((tfrecord_path, output_path))

    parallel = joblib.Parallel(n_jobs=args.n_jobs, backend="multiprocessing", verbose=5)
    parallel(joblib.delayed(read_tfrecord_and_save_as_numpy)(tfrecord_path, output_path, not args.is_test) \
             for tfrecord_path, output_path in input_output_paths)


if __name__ == "__main__":
    main()
