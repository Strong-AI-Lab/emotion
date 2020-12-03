# ALTA 2020
This code and the `alta2020` commit tag accompanies the paper submitted
to the ALTA 2020 conference [[1]](#References).

## Datasets
The datasets used are the 15 datasets specified in the `datasets`
directory. In order to run the scripts properly, each dataset folder
must be named according to the IDs given in
`../../emotion_recognition/corpora.py` (lowercase). Each directory must
have a `files.txt` file and a `labels.csv` file in it. `files.txt`
contains a list of absolute paths to audio files to process.
`labels.csv` is a CSV file with records mapping clip names to emotions.
A clip name is simply the basename of the audio file without the file
extension, and a label must be one of the labels specified in the
`CorpusInfo` structure in `corpora.py`.

## Running preprocessing and experiments
The script [run_all.sh](run_all.sh) will run the whole preprocessing and
training pipeline for each corpus, generating results in the `results`
directory.

*NOTE: This script must be run from the root of the repository, i.e.*
```
./papers/alta2020/run_all.sh
```

## References
- [1] A. Keesing, I. Watson, and M. Witbrock. "Convolutional and
  recurrent neural networks for speech emotion recognition", in The 18th
  Annual Workshop of the Australasian Language Technology Association,
  Jan. 2021.
