# Youtube-8m repository

### First, prepare the data (convert it to numpy, split into several folds):

```
python youtube8m/utils/bin/convert_to_numpy.py path_to_train_and_val_tfrecords_video_level /home/video_level_numpy_tmp
python youtube8m/utils/bin/convert_to_numpy.py path_to_test_tfrecords_video_level /home/video_level_numpy_test_tmp --is-test
python youtube8m/utils/bin/split_into_folds.py /home/video_level_numpy_tmp /home/video_level_numpy
python youtube8m/utils/bin/combine_test_blobs.py /home/video_level_numpy_test_tmp /home/video_level_numpy_test
python youtube8m/utils/bin/big_dataset_convert_and_split.py path_to_train_and_val_tfrecords_frame_level /home/video_level_numpy /home/frame_level_numpy
python youtube8m/utils/bin/big_dataset_convert_test.py path_to_test_tfrecords_frame_level /home/frame_level_numpy_test
```

### Then, train all L1 models.

Use config files from configs/ folder. Do not forget to use correct src_data section (see details in a config file).

For each config file, call:

```
python youtube8m/video_level_nn_models/bin/nntool.py --config path_to_config_file --out /home/model1_output_path fit_predict
```

### Prepare features for L2 model

For each trained model from L1:

1) Sort predictions by video_ids

```
python youtube8m/utils/bin/sort_all_predictions.py path_to_model
```

If you used frame-level features for training a model, add flag `--frame-level` to the command.

2) Average test predictions from different folds

```
python youtube8m/utils/bin/average_test_predictions.py path_to_model
```

If you used frame-level features for training a model, add flag `--frame-level` to the command.

3) Add path_to_model to youtube8m/utils/bin/ready_models

4) Merge predictions from different models

```
python youtube8m/utils/bin/merge_predictions.py youtube8m/utils/bin/ready_models output_path_with_merged_chunks
```

5) Extract features for LGBM:

```
python youtube8m/utils/bin/prepare_features_for_lgbm.py path_with_merged_chunks output_path_with_prepared_features
```


### Train LGBM model

```
python youtube8m/lgbm_models/bin/lgbm_train.py path_with_prepared_features/folds path_with_prepared_features/test output_path
```

### Build soft labels index

```
python youtube8m/utils/bin/build_soft_labels_index.py lgbm_output_path lgbm_path_with_prepared_features output_soft_labels_index_path
```

Then, train tf models separately: EnsembleModelA, EnsembleModelB, …, EnsembleModelF and use OOF predictions for training EnsembleModel. Note, you need to use correct soft_labels_path in NumpyY8MAggFeaturesReader.

After this procedure, you will be able to run eval.py and inference.py and get final predictions and a final metagraph.

