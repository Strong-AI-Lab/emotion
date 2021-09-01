#!/usr/bin/env bash

# This script runs all the experiments from the Interspeech 2021 paper
# as well as additional experiments using "online" normalisation and
# proper inner cross-validation (with GridSearchCV).

# Only report TF internal errors (quite spammy otherwise)
export TF_CPP_MIN_LOG_LEVEL=2

all_corpora=(CaFE CREMA-D DEMoS EMO-DB EmoFilm eNTERFACE IEMOCAP JL MSP-IMPROV Portuguese RAVDESS SAVEE ShEMO SmartKom TESS URDU VENEC)
BASE_DIR="$(dirname "$0")"
SCRIPTS_DIR="$BASE_DIR/../../scripts"
EXP_SCRIPT="$SCRIPTS_DIR/training/class_cv.py"
DATASETS_DIR="$BASE_DIR/../../datasets"
FEATURES_DIR="$BASE_DIR/../../features"
RESULTS_DIR="$BASE_DIR/results"

for corpus in "${all_corpora[@]}"; do
    corpus_yaml="$DATASETS_DIR/$corpus/corpus.yaml"
    for features in IS09 IS13 eGeMAPS GeMAPS mean_mfcc_64 boaw_20_500 boaw_50_1000 boaw_100_5000 audeep wav2vec wav2vec2 vq-wav2vec yamnet vggish densenet121 densenet169 densenet201; do
        for clf in svm/linear svm/poly2 svm/poly3 svm/rbf mlp/1layer mlp/2layer mlp/3layer rf depinto2020; do
            echo "$corpus" $features $clf

            # Experiments reported in paper
            common_opts=(--features "$FEATURES_DIR/$corpus/$features.nc" --clf "$clf" --noinner_cv --balanced --results "$RESULTS_DIR/norm_offline/$corpus/${clf/\//_}/$features.csv")
            case $corpus in
                CaFE|EMO-DB|JL|Portuguese|SAVEE|TESS)
                    # Leave-one-speaker-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition speaker --kfold -1 --normalise speaker "${common_opts[@]}"
                ;;
                CREMA-D|DEMoS|eNTERFACE|RAVDESS|ShEMO|SmartKom|VENEC)
                    # Leave-one-speaker-group-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition speaker --kfold 6 --normalise speaker "${common_opts[@]}"
                ;;
                IEMOCAP|MSP-IMPROV)
                    # Leave-one-session-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition session --kfold -1 --normalise speaker "${common_opts[@]}"
                ;;
                EmoFilm)
                    # Leave-one-language-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition language --kfold -1 --normalise language "${common_opts[@]}"
                ;;
                URDU)
                    # Leave-one-speaker-out, use global normalisation
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition speaker --kfold 6 --normalise all "${common_opts[@]}"
                ;;
            esac

            # "Online" normalisation experiments with proper inner CV
            common_opts=(--features "$FEATURES_DIR/$corpus/$features.nc" --clf "$clf" --normalise online --balanced --results "$RESULTS_DIR/norm_online/$corpus/${clf/\//_}/$features.csv")
            case $corpus in
                CaFE|EMO-DB|JL|SAVEE)
                    # Leave-one-speaker-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition speaker --kfold -1 --inner_kfold 2 "${common_opts[@]}"
                ;;
                TESS|Portuguese)
                    # Leave-one-speaker-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition speaker --kfold -1 --inner_kfold 2 --noinnergroup "${common_opts[@]}"
                ;;
                CREMA-D|DEMoS|eNTERFACE|RAVDESS|ShEMO|SmartKom|URDU|VENEC)
                    # Leave-one-speaker-group-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition speaker --kfold 6 --inner_kfold 2 "${common_opts[@]}"
                ;;
                IEMOCAP|MSP-IMPROV)
                    # Leave-one-session-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition session --kfold -1 --inner_kfold 2 "${common_opts[@]}"
                ;;
                EmoFilm)
                    # Leave-one-language-out
                    python "$EXP_SCRIPT" "$corpus_yaml" --partition language --kfold -1 --inner_kfold 2 "${common_opts[@]}"
                ;;
            esac
        done
    done
done
