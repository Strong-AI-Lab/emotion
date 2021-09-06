# INTERSPEECH 2021

This directory contains the files necessary to run the experiments from
our paper at INTERSPEECH 2021:

A. Keesing, Y. S. Koh, and M. Witbrock, "[Acoustic Features and Neural
Representations for Categorical Emotion Recognition from
Speech](https://www.isca-speech.org/archive/interspeech_2021/keesing21_interspeech.html),"
in Interspeech 2021, Aug. 2021

## Preprocessing scripts
There are a few scripts to do all of the preprocessing and feature
extraction.
 - In each of the directories in the `datasets` directory in this
   repository, run the associated `process.py` script with the location
   of the downloaded dataset. More information is in
   [`datasets/README.md`](../../datasets/README.md).
 - Run `audeep.sh` and `deepspectrum.sh` to extract corresponding
   features, **before** running `extract_features.sh`.
 - Make sure to download the AudioSet pretrained models (VGGish and
   YAMNet) in the root `third_party/` directory. Also download wav2vec
   pretrained models (see
   https://github.com/pytorch/fairseq/tree/master/examples/wav2vec).
 - Run `extract_features.sh`. This will create features files in
   `features/*/` with each dataset having its own directory.
 - Run `run_experiments.sh`. This will run all the experiments and put
   results in the results directory.


## Results
The notebook [display_results.ipynb](display_results.ipynb) can be used
to view results. I will update the notebook with additional results and
analyses soon.
