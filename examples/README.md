# Examples

This is a basic within-corpus and cross-corpus experimental setup.

## Datasets
Run `ertk-dataset setup` for each of CREMA-D, RAVDESS, EMO-DB:
```
ertk-dataset setup CREMA-D /path/to/CREMA-D ./CREMA-D
```

## Features
Run the `extract_features.sh` script to extract eGeMAPS, Wav2vec, and
log mel spectrogram features.

## Experiments
Run the `run_exps.sh` script to run experiments.
