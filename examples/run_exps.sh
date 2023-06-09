#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=1

# Within-corpus
for dataset in CREMA-D EMO-DB RAVDESS; do
    ertk-cli exp2 ${dataset}.yaml data.features=eGeMAPS model.type=sk/lr model.param_grid=\${cwdpath:../conf/clf/sk/lr/grids/default.yaml} results=results_within/lr/eGeMAPS.csv
    ertk-cli exp2 ${dataset}.yaml data.features=wav2vec_c_mean model.type=sk/lr model.param_grid=\${cwdpath:../conf/clf/sk/lr/grids/default.yaml} results=results_within/lr/wav2vec.csv
    ertk-cli exp2 ${dataset}.yaml data.features=eGeMAPS model.type=sk/svm model.param_grid=\${cwdpath:../conf/clf/sk/svm/grids/rbf.yaml} results=results_within/svm/eGeMAPS.csv
    ertk-cli exp2 ${dataset}.yaml data.features=wav2vec_c_mean model.type=sk/svm model.param_grid=\${cwdpath:../conf/clf/sk/svm/grids/rbf.yaml} results=results_within/svm/wav2vec.csv

    ertk-cli exp2 exp_loco.yaml training.normalise=none data.features=logmel-0.05-0.025-80 data.pad_seq=100 data.clip_seq=512 model.type=tf/aldeneh2017 model.config=\${cwdpath:aldeneh2017.yaml} results=results_within/aldeneh2017/melspec.csv
    ertk-cli exp2 exp_loco.yaml training.normalise=none data.features=logmel-0.05-0.025-80 data.pad_seq=256 data.clip_seq=256 model.type=tf/zhao2019 model.config=\${cwdpath:zhao2019.yaml} results=results_within/zhao2019/melspec.csv
done

# Cross-corpus

# SVM and logistic regression experiments
ertk-cli exp2 exp_loco.yaml data.features=eGeMAPS model.type=sk/lr model.param_grid=\${cwdpath:../conf/clf/sk/lr/grids/default.yaml} results=results_cross/lr/eGeMAPS.csv
ertk-cli exp2 exp_loco.yaml data.features=wav2vec_c_mean model.type=sk/lr model.param_grid=\${cwdpath:../conf/clf/sk/lr/grids/default.yaml} results=results_cross/lr/wav2vec.csv
ertk-cli exp2 exp_loco.yaml data.features=eGeMAPS model.type=sk/svm model.param_grid=\${cwdpath:../conf/clf/sk/svm/grids/rbf.yaml} results=results_cross/svm/eGeMAPS.csv
ertk-cli exp2 exp_loco.yaml data.features=wav2vec_c_mean model.type=sk/svm model.param_grid=\${cwdpath:../conf/clf/sk/svm/grids/rbf.yaml} results=results_cross/svm/wav2vec.csv

# Sequence models
ertk-cli exp2 exp_loco.yaml training.normalise=none data.features=logmel-0.05-0.025-80 data.pad_seq=100 data.clip_seq=512 model.type=tf/aldeneh2017 model.config=\${cwdpath:aldeneh2017.yaml} results=results_cross/aldeneh2017/melspec.csv
ertk-cli exp2 exp_loco.yaml training.normalise=none data.features=logmel-0.05-0.025-80 data.pad_seq=256 data.clip_seq=256 model.type=tf/zhao2019 model.config=\${cwdpath:zhao2019.yaml} results=results_cross/zhao2019/melspec.csv

# ertk-cli exp2 exp_loco.yaml training.normalise=none data.features=logmel-0.05-0.025-80 data.pad_seq=100 data.clip_seq=512 model.type=pt/aldeneh2017 model.config=\${cwdpath:../conf/clf/pt/aldeneh2017/default.yaml}
