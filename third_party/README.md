# Third-party tools
We include a number of third-party tools in this repository, for
convenience. The relevant LICENCE files are in each directory.

## openSMILE
[openSMILE](https://github.com/audeering/opensmile) is a standard
feature extraction toolkit for emotion recognition from speech. The
binaries of openSMILE are distributed along with stock and custom config
files.

## openXBOW
[openXBOW](https://github.com/openXBOW/openXBOW) is a tool for vector
quantization, clustering and creation of bag-of-words from a sequence of
audio features. It can be used to create so-called bag-of-audio-words
(BoAW) for subsequent classification. The openXBOW JAR file is
distributed.

## auDeep
[auDeep](https://github.com/auDeep/auDeep) is a convolutional and
recurrent neural network for representation learning from spectrograms.
We include a slightly modified repository as dependency. These
modifications include saving per epoch instead of per batch, and a
custom metadata parser using `files.txt`. There are some scripts that
use auDeep that must be run inside the auDeep Docker container, which
can be built in the main auDeep directory:
```
docker build -t audeep .
```

## AudioSet models
The [AudioSet
models](https://github.com/tensorflow/models/tree/master/research/audioset/)
from the TensorFlow models repo are included here for generating
embeddings that can be used in subsequence classifiers. The necessary
files for both
[VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
and
[YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
are included, along with an `embeddings.py` script in each directory
that can be used to generate the embeddings. You'll need to manually
download the model files as specified in the instructions in the models
repo.
