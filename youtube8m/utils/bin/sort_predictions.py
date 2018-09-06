from youtube8m.utils.io import sort_files_natural

import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser("This script loads predictions, sorts them by video_id and saves in several chunks")
    parser.add_argument("predictions_path")
    parser.add_argument("id_chunks_path")
    parser.add_argument("output_path")
    parser.add_argument("--chunk-size", type=int, default=10000)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    video_ids = []
    id_chunks_paths = sort_files_natural(os.listdir(args.id_chunks_path))

    for chunk_name_path in id_chunks_paths:
        video_ids_path = os.path.join(args.id_chunks_path, chunk_name_path, "video_ids")
        video_ids_chunk = np.load(video_ids_path)
        video_ids += video_ids_chunk.tolist()

    video_ids = np.array(video_ids)

    predictions = []
    for chunk_name in sort_files_natural(os.listdir(args.predictions_path)):
        chunk_path = os.path.join(args.predictions_path, chunk_name)

        chunk = np.load(chunk_path)
        predictions.append(chunk)

    predictions = np.concatenate(predictions)

    sort_order = np.argsort(video_ids)
    predictions = predictions[sort_order]

    chunk_id = 0
    for chunk_start in range(0, len(predictions), args.chunk_size):
        chunk_predictions = predictions[chunk_start: chunk_start + args.chunk_size]

        chunk_output_path = os.path.join(args.output_path, "chunk_{}".format(chunk_id))
        with open(chunk_output_path, "wb") as fout:
            np.save(fout, chunk_predictions)

        chunk_id += 1


if __name__ == "__main__":
    main()
