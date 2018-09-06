from youtube8m.utils.constants import FOLD_COUNT, VIDEO_LEVEL_FOLDS_PATH, VIDEO_LEVEL_TEST_PATH,\
                                      FRAME_LEVEL_FOLDS_PATH, FRAME_LEVEL_TEST_PATH

import subprocess
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--frame-level", action="store_true")

    args = parser.parse_args()

    data_path = VIDEO_LEVEL_FOLDS_PATH
    test_data_path = VIDEO_LEVEL_TEST_PATH
    if args.frame_level:
        data_path = FRAME_LEVEL_FOLDS_PATH
        test_data_path = FRAME_LEVEL_TEST_PATH

    for fold_id in range(FOLD_COUNT):
        fold_path = os.path.join(args.path, "folds", "fold_{}".format(fold_id))
        val_args = "{} {} {}".format(os.path.join(fold_path, "predictions/predictions"),
                                     os.path.join(data_path, "fold_{}".format(fold_id)),
                                     os.path.join(fold_path, "predictions/sorted"))
        test_args = "{} {} {}".format(os.path.join(fold_path, "test_predictions/predictions"), test_data_path,
                                      os.path.join(fold_path, "test_predictions/sorted"))

        cmd = "python sort_predictions.py"

        subprocess.check_call(cmd + " " + val_args, shell=True)
        subprocess.check_call(cmd + " " + test_args, shell=True)


if __name__ == "__main__":
    main()
