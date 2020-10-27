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
Clone [this repository](https://github.com/audio-captioning/clotho-dataset) and follow its instructions to create dataset.
  
The result of the dataset creation process will be the creation of the directories:

    1. `data/data_splits`,
    2. `data/data_splits/development`,
    3. `data/data_splits/evaluation`, and
    4. `data/pickles`

The directories in data/data_splits have the input and output examples for the optimization and assessment of the baseline DNN. The data/pickles directory holds the pickle files that have the frequencies of the words and characters (so one can use weights in the objective function) and the correspondence of words and characters with indices.


## Using the pre-trained weights for inference
The pre-trained weights are stored at [outputs/models directory](https://github.com/haantran96/wavetransformer/tree/main/outputs/models). Please be noted that the pre-trained weights are different for each different model.
**Note bold**: To use the caption evaluate tools you need to have Java installed and enabled.
Before being able to run the code for the evaluation of the predictions, you have first to run the script `get_stanford_models.sh` in the `coco_caption` directory.
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
  pickle_files_dir: 'WT_pickles'
  files:
    np_file_name_template: 'clotho_file_{audio_file_name}_{caption_index}.npy'
    words_list_file_name: 'WT_words_list.p'
    characters_list_file_name: 'WT_characters_list.p'
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

**Note bold 2**: To obtain the exactly same results as we had in the paper, please use the same [word indices and character indices](https://github.com/haantran96/wavetransformer/tree/main/data/WT_pickles) that we had already generated.

Then, please specify the directory as shown in the dirs_and_files.yaml:
```
...
  pickle_files_dir: 'WT_pickles'
  files:
    np_file_name_template: 'clotho_file_{audio_file_name}_{caption_index}.npy'
    words_list_file_name: 'WT_words_list.p'
    characters_list_file_name: 'WT_characters_list.p'
...
```


2. `main_settings_$ID.yaml`. For inference, please set `dnn_training: No` as shown below:
```
workflow:
  dnn_training: No
  dnn_evaluation: Yes
dnn_training_settings: !include method.yaml
# -----------------------------------
dirs_and_files: !include dirs_and_files.yaml
# EOF
```

3. `method_$ID.yaml`: contain different hyperparameters. This is the setting for the best models:
```
model: !include model_$ID.yaml
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
  use_y: Yes
  clamp_value_freqs: -1  # -1 is for ignoring
  # EOF
```
4. `model_$ID.yaml`: The settings are different for different models. However, this line should be set to "Yes" to do the inference:
```use_pre_trained_model: Yes```

*Please use the according files for reference:
- `best_model_43_3.pt`: [WT](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_37.yaml)
- `best_model_16_3_9.pt`: [WT_temp](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_16_3.yaml)
- `best_model_44_7.pt`: [WT_avg](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_37.yaml)
Please change accordingly to the following hyperparameters:
```
  inner_kernel_size_encoder: 5
  inner_padding_encoder: 2
  pw_kernel_encoder: 5
  pw_padding_encoder: 2
  merge_mode_encoder: 'mean'
```
- `best_model_38_5.pt`: [WT_tf](https://github.com/haantran96/wavetransformer/blob/main/settings/model_ht_12_39.yaml)

***For beam search:*** In order to use beam search, please set in the yaml model files:

```beam_size: 2``` or larger than 1

Our results are obtained with beam size 2. You can set the beam size larger, but inference time can vary.

Finally, to run the whole inference code:
```
python main.py -c main_settings_$ID -j $id_nr
```
`main_settings_$ID` should be the same name with your `main_settings_$ID.yaml` file.

## Re-training WaveTransformer
The process for retraining are the same like inference. However, you must change as the following:
1. `main_settings_$ID.yaml`:
```
workflow:
  dnn_training: Yes
  dnn_evaluation: Yes
```
2. `method_$ID.yaml`: make changes as to the indicated hyperparameters
3. `model_$ID.yaml`: this line should be set to "No" to do the training (from scratch):

```use_pre_trained_model: No```

If you wish to continue training, you can also set `use_pre_trained_model` to `Yes`.

# Acknowledgement
The implementation of the codebase is adapted (with some modifications) from the following works:
1. For WaveNet implementation: https://www.kaggle.com/c/liverpool-ion-switching/discussion/145256
2. For Transformer implementation: https://nlp.seas.harvard.edu/2018/04/03/attention.html
3. For beam search decoding: https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
4. For Depthwise separable convolution implementation: https://github.com/dr-costas/dnd-sed
