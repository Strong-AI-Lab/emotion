for corpus in cafe crema-d demos emodb emofilm enterface iemocap jl \
        msp-improv portuguese ravdess savee shemo smartkom tess; do
    for config in eGeMAPSv01a GeMAPSv01a IS09_emotion IS13_ComParE; do
        python scripts/comparative.py $corpus svm $config --kind linear
        python scripts/comparative.py $corpus svm $config --kind poly2
        python scripts/comparative.py $corpus svm $config --kind poly3
        python scripts/comparative.py $corpus svm $config --kind rbf
        python scripts/comparative.py $corpus dnn $config --kind basic
        python scripts/comparative.py $corpus dnn $config --kind 1layer
        python scripts/comparative.py $corpus dnn $config --kind 2layer
        python scripts/comparative.py $corpus dnn $config --kind 3layer
    done
done
