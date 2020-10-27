# WaveTransformer Repository
Welcome to the repository of the paper [WaveTransformer: A Novel Architecture for Audio Captioning Based on Learning Temporal and Time-Frequency Information](https://arxiv.org/abs/2010.11098) 
If you want to reproduce the results of the paper and know what you are doing, then jump ahead, get the pre-trained weights from [here](https://github.com/haantran96/wavetransformer/tree/main/outputs/models) and run the inference code as shown [here](#using-the-pre-trained-weights-for-inference)
If you want to re-train WaveTransformer, then you can use the master branch, as it has the code based on the most up-to-date version of PyTorch.

There is also an [on-line demo](https://haantran96.github.io/wavetransformer-web-demo/) of the WaveTransformer.

If you need some help on using WaveTransformer, please read the following instructions.
- [How do I use WaveTransformer](#how-do-i-use-wavetransformer)
  * [Setting up the environment](#setting-up-the-environment)
  * [Dataset setup](#dataset-setup)
  * [Create a dataset](#create-a-dataset)
  * [Using the pre-trained weights for inference](#using-the-pre-trained-weights-for-inference)
  * [Re-training WaveTransformer](#re-training-wavetransformer)
- [Acknowledgement](#acknowledgement)

# How do I use WaveTransformer
## Setting up the environment

To start using the audio captioning WaveTransformer, firstly you have to set-up the code. Please note bold that the code in this repository is tested with Python 3.7 or 3.6.

To set-up the code, you have to do the following:

    1. Clone this repository.
    2. Install dependencies

Use the following command to clone this repository at your terminal:
```
$ git clone git@github.com:haantran96/wavetransformer.git
```
To install the dependencies, you can use pip. It is advisable to run this system with a virtual environment to avoid package conflicts
```
$ pip install -r requirement_pip.txt
```

## Dataset setup
Please go to DCASE2020's Baseline repository, part [Preparing the data](https://github.com/audio-captioning/dcase-2020-baseline#preparing-the-data) to download and set up the data.

## Create a dataset
There are 2 method to create dataset from the audio files:

**Method 1:**

  Clone [this repository](https://github.com/audio-captioning/clotho-dataset) and follow its instructions to create dataset.
  
**Method 2:**
  
  -In the `main_settings_$ID.yaml`, change to the following line:
  ```
  workflow:
    dataset_creation: Yes
  ```
  -In `dirs_and_files_$ID.yaml`:
  ```
    features_dirs:
    output: 'data_splits'
    development: *dev
    evaluation: *eva
    validation: *val
  audio_dirs:
    downloaded: 'clotho_audio_files'
    output: 'data_splits_audio'
    development: *dev
    evaluation: *eva
    validation: *val
  annotations_dir: 'clotho_csv_files'
  pickle_files_dir: 'pickles'
  ```
  Please note that you need to create directory for audios in `data/clotho_audio_files` and the csv files in `data/clotho_csv_files`.
  
  then run:
  ```python main.py -c main_settings -j $ID```

The result of the dataset creation process will be the creation of the directories:

    1. `data/data_splits`,
    2. `data/data_splits/development`,
    3. `data/data_splits/evaluation`, and
    4. `data/pickles`

The directories in data/data_splits have the input and output examples for the optimization and assessment of the baseline DNN. The data/pickles directory holds the pickle files that have the frequencies of the words and characters (so one can use weights in the objective function) and the correspondence of words and characters with indices.

**Note**: Once you have created the dataset, there is no need to create it every time. That is, after you create the dataset using the baseline system, then you can set
```
workflow:
  dataset_creation: No
```
at the `settings/main_settings.yaml` file.

## Using the pre-trained weights for inference
The pre-trained weights are stored at [outputs/models directory](https://github.com/haantran96/wavetransformer/tree/main/outputs/models). Please be noted that the pre-trained weights are different for each different model.

In the `settings` folder, there are the following files:
1. `dirs_and_files.yaml`: Stores the locations of the according files. For example:
```
root_dirs:
  outputs: 'outputs'
  data: 'data'
# -----------------------------------
dataset:
  development: &dev 'development'
  evaluation: &eva 'evaluation'
  validation: &val 'validation'
  features_dirs:
    output: 'data_splits'
    development: *dev
    evaluation: *eva
    validation: *val
  audio_dirs:
    downloaded: 'clotho_audio_files'
    output: 'data_splits_audio'
    development: *dev
    evaluation: *eva
    validation: *val
  annotations_dir: 'clotho_csv_files'
  pickle_files_dir: 'pickles'
  files:
    np_file_name_template: 'clotho_file_{audio_file_name}_{caption_index}.npy'
    words_list_file_name: 'words_list.p'
    words_counter_file_name: 'words_frequencies.p'
    characters_list_file_name: 'characters_list.p'
    characters_frequencies_file_name: 'characters_frequencies.p'
    validation_files_file_name: 'validation_file_names.p'
# -----------------------------------
model:
  model_dir: 'models'
  checkpoint_model_name: 'model_name.pt'
  pre_trained_model_name: 'best_model_name.pt'
# -----------------------------------
logging:
  logger_dir: 'logging'
  caption_logger_file: 'caption_file.txt'
```

Most important directories are: `feature_dirs/output` and `model`, as you must specify the locations of the `/data` and model paths according. Noted: by default, the code will save current best model as `best_checkpoint_model_name.pt`, so it is advisable to always set `model/pre_trained_model_name` as `best_checkpoint_model_name.pt`.

2. `main_settings.yaml`. As mentioned, if you have already created the database, please set `dataset_creation: No`. For inference, please set `dnn_training: No` as shown below:
```
workflow:
  dataset_creation: No
  dnn_training: No
  dnn_evaluation: Yes
# ---------------------------------
dataset_creation_settings: !include dataset_creation.yaml
# -----------------------------------
feature_extraction_settings: !include feature_extraction.yaml
# -----------------------------------
dnn_training_settings: !include method.yaml
# -----------------------------------
dirs_and_files: !include dirs_and_files.yaml
# EOF
```

3. `method.yaml`: contain different hyperparameters. This is the setting for the best models:
```
model: !include model.yaml
# ----------------------
data:
  input_field_name: 'features'
  output_field_name: 'words_ind'
  load_into_memory: No
  batch_size: 12 
  shuffle: Yes
  num_workers: 30
  drop_last: Yes
  use_multiple_mode: No
  use_validation_split: Yes 
# ----------------------
training:
  nb_epochs: 300
  patience: 10
  loss_thr: !!float 1e-4
  optimizer:
    lr: !!float 1e-3
  grad_norm:
    value: !!float 1.
    norm: 2
  force_cpu: No
  text_output_every_nb_epochs: !!int 10
  nb_examples_to_sample: 100
  use_class_weights: Yes
  use_y: Yes
  clamp_value_freqs: -1  # -1 is for ignoring
  # EOF
```
4. `model.yaml`: The settings are different for different models. However, this line should be set to "Yes" to do the inference:
```use_pre_trained_model: Yes```

*Please use the according files for reference:
- `best_model_16_3_9.pt`: [model_ht_12_16_3.yaml](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_16_3.yaml)
- `best_model_37_8.pt`: [model_ht_12_37.yaml](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_37.yaml)
- `best_model_43_3.pt`: [model_ht_12_37.yaml](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_37.yaml)
However, these hyperparameters should be changed as:
```
  inner_kernel_size_encoder: 5
  inner_padding_encoder: 2
  pw_kernel_encoder: 5
  pw_padding_encoder: 2
```
- `best_model_44_7.pt`: [model_ht_12_37.yaml](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_37.yaml)
However, these hyperparameters should be changed as:
```
  inner_kernel_size_encoder: 5
  inner_padding_encoder: 2
  pw_kernel_encoder: 5
  pw_padding_encoder: 2
  merge_mode_encoder: 'mean'
```
- `best_model_39_5.pt`: [model_ht_12_39.yaml](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_39.yaml)
- `best_model_38_5.pt`: [model_ht_12_39.yaml](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_39.yaml)
However, these hyperparameters should be changed as:
```
  inner_kernel_size_encoder: 5
  inner_padding_encoder: 2
```

Finally, to run the whole inference code:
```
python main.py -c main_settings -j $ID
```
`main_settings` should be the same name with your `main_settings.yaml` file.

## Re-training WaveTransformer
The process for retraining are the same like inference. However, you must change as the following:
1. `main_settings.yaml`. As mentioned, if you have already created the database, please set `dataset_creation: No`. For training, please set `dnn_training: Yes` as shown below:
```
workflow:
  dataset_creation: No
  dnn_training: Yes
  dnn_evaluation: Yes
```
2. `method.yaml`: make changes as to the indicated hyperparameters
3. `model.yaml`: this line should be set to "No" to do the training (from scratch):

```use_pre_trained_model: No```

If you wish to continue training, you can also set `use_pre_trained_model` to `Yes`.

# Acknowledgement
The implementation of the codebase is adapted (with some modifications) from the following works:
1. For WaveNet implementation: https://www.kaggle.com/c/liverpool-ion-switching/discussion/145256
2. For Transformer implementation: https://nlp.seas.harvard.edu/2018/04/03/attention.html
3. For beam search decoding: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
4. For Depthwise separable convolution implementation: https://github.com/dr-costas/dnd-sed
