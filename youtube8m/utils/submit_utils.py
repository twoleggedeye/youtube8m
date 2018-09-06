import os
import numpy as np

from youtube8m.utils.io import sort_files_natural

NUM_PREDICTIONS = 20


def prepare_submit(video_ids, predictions_path, output_path):
    predictions = []
    current_video_id_index = 0
    for chunk_name in sort_files_natural(os.listdir(predictions_path)):
        chunk_path = os.path.join(predictions_path, chunk_name)

        chunk = np.load(chunk_path)
        for video_predictions in chunk:
            video_id = video_ids[current_video_id_index]
            labels = np.argsort(video_predictions)[::-1]

            label_confidence_pairs = []
            for label in labels[:NUM_PREDICTIONS]:
                confidence = video_predictions[label]
                label_confidence_pairs.append((label, confidence))

            predictions.append((video_id, label_confidence_pairs))
            current_video_id_index += 1

    predictions = sorted(predictions)
    with open(output_path, "w") as fout:
        fout.write("VideoId,LabelConfidencePairs\n")
        for video_id, pairs in predictions:
            label_confidence_list = []
            for label, confidence in pairs:
                label_confidence_list.append(str(label))
                label_confidence_list.append(str(confidence))

            fout.write("{},{}\n".format(video_id, " ".join(label_confidence_list)))
