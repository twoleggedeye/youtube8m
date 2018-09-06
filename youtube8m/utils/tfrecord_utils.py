import numpy as np
import os

import tensorflow as tf


def read_tfrecord(path, is_training, float_type):
    video_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []

    for example in tf.python_io.tf_record_iterator(path):
        tf_example = tf.train.Example.FromString(example).features
        video_ids.append(tf_example.feature["id"].bytes_list.value[0].decode(encoding="UTF-8"))
        if is_training:
            labels.append(np.array(tf_example.feature["labels"].int64_list.value))
        mean_rgb.append(np.array(tf_example.feature["mean_rgb"].float_list.value, dtype=float_type))
        mean_audio.append(np.array(tf_example.feature["mean_audio"].float_list.value, dtype=float_type))

    video_ids = np.array(video_ids)
    mean_rgb = np.array(mean_rgb)
    mean_audio = np.array(mean_audio)
    labels = np.array(labels)

    return video_ids, mean_rgb, mean_audio, labels


def read_big_tfrecord(path, is_training):
    video_ids = []
    labels = []
    rgb = []
    audio = []

    for example in tf.python_io.tf_record_iterator(path):
        tf_example = tf.train.SequenceExample.FromString(example)
        video_ids.append(tf_example.context.feature["id"].bytes_list.value[0].decode(encoding="UTF-8"))
        if is_training:
            labels.append(np.array(tf_example.context.feature["labels"].int64_list.value))

        rgb_feature_list = tf_example.feature_lists.feature_list["rgb"].feature
        audio_feature_list = tf_example.feature_lists.feature_list["audio"].feature

        rgb_numpy = []
        audio_numpy = []
        for rgb_feature, audio_feature in zip(rgb_feature_list, audio_feature_list):
            rgb_numpy.append(np.fromstring(rgb_feature.bytes_list.value[0], dtype=np.uint8))
            audio_numpy.append(np.fromstring(audio_feature.bytes_list.value[0], dtype=np.uint8))

        rgb.append(np.array(rgb_numpy))
        audio.append(np.array(audio_numpy))

    video_ids = np.array(video_ids)
    rgb = np.array(rgb)
    audio = np.array(audio)
    labels = np.array(labels)

    return video_ids, rgb, audio, labels


def read_tfrecord_and_save_as_numpy(tfrecord_path, output_path, is_training=True, float_type="float16"):
    video_ids, mean_rgb, mean_audio, labels = read_tfrecord(tfrecord_path, is_training, float_type)

    os.makedirs(output_path, exist_ok=True)

    np.savez(os.path.join(output_path, "video_ids.npz"), video_ids)
    np.savez(os.path.join(output_path, "mean_rgb.npz"), mean_rgb)
    np.savez(os.path.join(output_path, "mean_audio.npz"), mean_audio)

    if is_training:
        np.savez(os.path.join(output_path, "labels.npz"), labels)
