#!/usr/bin/env python3.6

import argparse
import numpy as np
import os
import tqdm


def save_chunk(output_path, chunk_id, video_ids, mean_rgb, mean_audio):
    video_ids_array = np.array(video_ids)
    mean_rgb_array = np.array(mean_rgb)
    mean_audio_array = np.array(mean_audio)

    output_path_chunk = os.path.join(output_path, "chunk_{}".format(chunk_id))
    os.makedirs(output_path_chunk, exist_ok=True)

    np.save(os.path.join(output_path_chunk, "video_ids"), video_ids_array)
    np.save(os.path.join(output_path_chunk, "rgb"), mean_rgb_array)
    np.save(os.path.join(output_path_chunk, "audio"), mean_audio_array)


def main():
    parser = argparse.ArgumentParser("This script combines blobs of test dataset and saves them in several chunks")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--chunk-size", type=int, default=500000, help="The number of examples per one chunk")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    in_memory_video_ids = []
    in_memory_mean_rgb_ = []
    in_memory_mean_audio = []

    data_folders = os.listdir(args.input_path)

    next_chunk_id = 0

    for data_folder_name in tqdm.tqdm(data_folders):
        video_ids_path = os.path.join(args.input_path, data_folder_name, "video_ids.npz")
        mean_rgb_path = os.path.join(args.input_path, data_folder_name, "mean_rgb.npz")
        mean_audio_path = os.path.join(args.input_path, data_folder_name, "mean_audio.npz")

        video_ids_blob = np.load(video_ids_path)["arr_0"]
        mean_rgb_blob = np.load(mean_rgb_path)["arr_0"]
        mean_audio_blob = np.load(mean_audio_path)["arr_0"]

        for video_id, mean_rgb, mean_audio in zip(video_ids_blob, mean_rgb_blob, mean_audio_blob):
            in_memory_video_ids.append(video_id)
            in_memory_mean_rgb_.append(mean_rgb)
            in_memory_mean_audio.append(mean_audio)

            if len(in_memory_video_ids) == args.chunk_size:
                save_chunk(
                    args.output_path,
                    next_chunk_id,
                    in_memory_video_ids,
                    in_memory_mean_rgb_,
                    in_memory_mean_audio)

                in_memory_video_ids.clear()
                in_memory_mean_rgb_.clear()
                in_memory_mean_audio.clear()

                next_chunk_id += 1

    save_chunk(
        args.output_path,
        next_chunk_id,
        in_memory_video_ids,
        in_memory_mean_rgb_,
        in_memory_mean_audio)


if __name__ == "__main__":
    main()
