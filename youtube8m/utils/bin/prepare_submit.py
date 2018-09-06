#!/usr/bin/env python3.6

from youtube8m.utils.io import sort_files_natural
from youtube8m.utils.submit_utils import prepare_submit

import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser("This script loads predictions from numpy and prepares submit file")
    parser.add_argument("predictions_path")
    parser.add_argument("test_chunks_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    video_ids = []
    test_chunk_paths = sort_files_natural(os.listdir(args.test_chunks_path))

    for chunk_name_path in test_chunk_paths:
        video_ids_path = os.path.join(args.test_chunks_path, chunk_name_path, "video_ids")
        video_ids_chunk = np.load(video_ids_path)
        video_ids += video_ids_chunk.tolist()

    prepare_submit(video_ids, args.predictions_path, args.output_path)


if __name__ == "__main__":
    main()
