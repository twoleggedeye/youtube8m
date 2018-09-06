from youtube8m.utils.constants import FOLD_COUNT

import os
import numpy as np


def get_total_posititive_labels(full_labels_data_path):
    result = []

    for fold_id in range(FOLD_COUNT):
        fold_result = 0
        fold_path = os.path.join(full_labels_data_path, "fold_{}".format(fold_id))

        chunk_paths = os.listdir(fold_path)
        for chunk_name in chunk_paths:
            labels_path = os.path.join(fold_path, chunk_name, "labels")
            labels = np.load(labels_path)

            for video_labels in labels:
                fold_result += len(video_labels)

        result.append(fold_result)

    return result
