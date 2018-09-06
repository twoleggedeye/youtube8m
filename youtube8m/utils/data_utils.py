import numpy as np
import os
import pickle

from youtube8m.utils.io import sort_files_natural


def shuffle_and_save_chunk(output_path, fold_id, chunk_id, arrays, file_names, use_pickle=False):
    numpy_arrays = [np.array(array) for array in arrays]

    output_path_fold = os.path.join(output_path, "fold_{}".format(fold_id))
    output_path_chunk = os.path.join(output_path_fold, "chunk_{}".format(chunk_id))

    os.makedirs(output_path_chunk, exist_ok=True)

    indexes = np.arange(start=0, stop=len(numpy_arrays[0]))
    np.random.shuffle(indexes)

    numpy_arrays = [array[indexes] for array in numpy_arrays]

    for array, file_name in zip(numpy_arrays, file_names):
        file_path = os.path.join(output_path_chunk, file_name)
        with open(file_path, "wb") as fout:
            if use_pickle:
                pickle.dump(array, fout)
            else:
                np.save(fout, array)


def get_video_id_to_fold_id_dict(fold_split_path):
    fold_paths = os.listdir(fold_split_path)
    fold_paths = sort_files_natural(fold_paths)

    result = {}
    for fold_path_name in fold_paths:
        fold_id = int(fold_path_name.split("_")[-1])
        fold_path = os.path.join(fold_split_path, fold_path_name)
        chunk_file_paths = os.listdir(fold_path)

        for chunk_file_path_name in chunk_file_paths:
            chunk_path = os.path.join(fold_path, chunk_file_path_name)
            video_ids_path = os.path.join(chunk_path, "video_ids")
            video_ids = np.load(video_ids_path)

            for video_id in video_ids:
                result[video_id] = fold_id

    return result
