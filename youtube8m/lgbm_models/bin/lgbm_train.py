from youtube8m.utils.constants import FOLD_COUNT, VIDEO_LEVEL_FOLDS_PATH
from youtube8m.lgbm_models.utils import get_total_posititive_labels

from lightgbm import LGBMClassifier
import os
import numpy as np
import pickle
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path")
    parser.add_argument("test_data_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    folds_features = []
    folds_targets = []
    folds_video_ids = []

    for fold_id in range(FOLD_COUNT):
        fold_path = os.path.join(args.train_data_path, "fold_{}".format(fold_id))
        features = np.load(os.path.join(fold_path, "features"))
        labels = np.load(os.path.join(fold_path, "labels"))
        video_ids = np.load(os.path.join(fold_path, "video_ids"))

        folds_features.append(features)
        folds_targets.append(labels)
        folds_video_ids.append(video_ids)

    test_features = np.load(os.path.join(args.test_data_path, "features"))
    test_video_ids = np.load(os.path.join(args.test_data_path, "video_ids"))
    test_candidates = np.load(os.path.join(args.test_data_path, "candidates"))

    total_positive_labels_by_fold = get_total_posititive_labels(VIDEO_LEVEL_FOLDS_PATH)

    os.makedirs(args.output_path, exist_ok=True)
    shutil.copy(__file__, os.path.join(args.output_path, "train.py"))

    for fold_id in range(FOLD_COUNT):
        train_features = folds_features[:fold_id] + folds_features[fold_id + 1:]
        val_features = folds_features[fold_id]

        train_targets = folds_targets[:fold_id] + folds_targets[fold_id + 1:]
        val_targets = folds_targets[fold_id]

        train_video_ids = folds_video_ids[:fold_id] + folds_video_ids[fold_id + 1:]
        val_video_ids = folds_video_ids[fold_id]

        train_features = np.concatenate(train_features)
        train_targets = np.concatenate(train_targets)

        model = LGBMClassifier(n_estimators=3000, learning_rate=0.01, n_jobs=28)
        model.fit(train_features, train_targets, early_stopping_rounds=30,
                  eval_set=[(val_features, val_targets), (train_features, train_targets)])

        output_fold_path = os.path.join(args.output_path, "fold_{}".format(fold_id))
        os.makedirs(output_fold_path, exist_ok=True)

        with open(os.path.join(output_fold_path, "model"), "wb") as fout:
            pickle.dump(model, fout)

        y_pred = model.predict(val_features)

        with open(os.path.join(output_fold_path, "predictions"), "wb") as fout:
            np.save(fout, y_pred)

        with open(os.path.join(output_fold_path, "labels"), "wb") as fout:
            np.save(fout, val_targets)

        with open(os.path.join(output_fold_path, "video_ids"), "wb") as fout:
            np.save(fout, val_video_ids)

        test_pred = model.predict(test_features)

        with open(os.path.join(output_fold_path, "test_predictions"), "wb") as fout:
            np.save(fout, test_pred)

        with open(os.path.join(output_fold_path, "test_video_ids"), "wb") as fout:
            np.save(fout, test_video_ids)

        with open(os.path.join(output_fold_path, "test_candidates"), "wb") as fout:
            np.save(fout, test_candidates)

        print("Start gap computing")

        last_video_id = None
        last_pred = []

        gap_targets = []
        gap_scores = []

        for i in range(len(val_video_ids)):
            video_id = val_video_ids[i]
            if video_id != last_video_id:
                if last_video_id is not None:
                    last_pred = sorted(last_pred)[-20:]
                    for example in last_pred:
                        gap_scores.append(example[0])
                        gap_targets.append(example[1])

                    last_pred = []

                last_video_id = video_id

            last_pred.append((y_pred[i], val_targets[i]))

        if last_video_id is not None:
            last_pred = sorted(last_pred)[-20:]
            for example in last_pred:
                gap_scores.append(example[0])
                gap_targets.append(example[1])

        gap_targets = np.array(gap_targets)
        gap_scores = np.array(gap_scores)

        idxs = np.argsort(gap_scores)[::-1]
        sorted_targets = gap_targets[idxs]
        ranks = np.arange(len(idxs)) + 1
        n_pos_at_rank = np.cumsum(sorted_targets)
        AP = (n_pos_at_rank * sorted_targets / ranks).sum() / total_positive_labels_by_fold[fold_id]
        print("GAP on fold {0}:".format(fold_id), AP)


if __name__ == "__main__":
    main()
