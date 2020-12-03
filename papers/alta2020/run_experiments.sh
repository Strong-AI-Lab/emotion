#!/usr/bin/sh

# This script runs the main experiment script on various combinations of
# classifier and features on a given dataset.


# SVM-RBF with IS09 features for reference
python papers/alta2020/run_experiment.py --kind rbf --reps 5 --data output/$1/IS09.nc --results results/comparative2020/$1/svm_rbf/IS09.csv

# Aldeneh et al. (2017) classifier
python papers/alta2020/run_experiment.py --kind aldeneh2017 --batch_size 32 --reps 3 --data output/$1/logmel_40.nc --pad 64 --results results/comparative2020/$1/aldeneh2017/logmel_40.csv
python papers/alta2020/run_experiment.py --kind aldeneh2017 --batch_size 32 --reps 3 --data output/$1/logmel_240.nc --pad 64 --results results/comparative2020/$1/aldeneh2017/logmel_240.csv
python papers/alta2020/run_experiment.py --kind aldeneh2017 --batch_size 32 --reps 3 --data output/$1/spectrograms-0.025-0.010-240-60.nc --pad 64 --results results/comparative2020/$1/aldeneh2017/spectrograms_240.csv
# This one wasn't actually presented in the paper
python papers/alta2020/run_experiment.py --kind aldeneh2017 --batch_size 32 --reps 3 --data output/$1/spectrograms-0.025-0.010-40-60.nc --pad 64 --results results/comparative2020/$1/aldeneh2017/spectrograms_40.csv

# Latif et al. (2019) classifier
python papers/alta2020/run_experiment.py --kind latif2019 --batch_size 32 --reps 3 --data output/$1/raw_audio.nc --clip 80000 --results results/comparative2020/$1/latif2019/raw_audio.csv

# Zhang et al. (2019) classifier
python papers/alta2020/run_experiment.py --kind zhang2019 --batch_size 16 --reps 1 --data output/$1/raw_audio.nc --clip 80000 --results results/comparative2020/$1/zhang2019/raw_audio.csv

# Zhao et al. (2019) classifier
python papers/alta2020/run_experiment.py --kind zhao2019 --batch_size 64 --reps 3 --data output/$1/spectrograms-0.025-0.010-40-60.nc --pad 512 --results results/comparative2020/$1/zhao2019/spectrograms_40.csv
