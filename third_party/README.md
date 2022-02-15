# Third-party tools
We include a number of third-party tools in this repository, for
convenience. The relevant LICENCE files are in each directory.

## auDeep
[auDeep](https://github.com/auDeep/auDeep) is a convolutional and
recurrent neural network for representation learning from spectrograms.
We include a slightly modified repository as dependency. These
modifications include saving per epoch instead of per batch, and a
custom metadata parser using `files.txt`. There are some scripts that
use auDeep that must be run with the version of auDeep from this repo.

A Docker image can also be built from the main auDeep directory:
```
docker build -t audeep .
```
