Emotion Datasets
================

A large number of emotional speech datasets have processing scripts and
metadata available in the ERTK repository. Currently the scripts only
deal with audio data, but in future we intend to support video data.


Currently supported datasets:
-----------------------------
* `AESDD
  <http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/>`_
* `ASED <https://github.com/wgwangang/ASED_V1>`_
* `BAVED
  <https://www.kaggle.com/a13x10/basic-arabic-vocal-emotions-dataset>`_
* `CaFE <https://zenodo.org/record/1478765>`_
* `CMU-MOSEI
  <http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/>`_
* `CREMA-D <https://github.com/CheyneyComputerScience/CREMA-D>`_
* `DEMoS <https://zenodo.org/record/2544829>`_
* `EESC
  <https://metashare.ut.ee/repository/browse/estonian-emotional-speech-corpus/4d42d7a8463411e2a6e4005056b40024a19021a316b54b7fb707757d43d1a889/>`_
* `EMO-DB <http://emodb.bilderbar.info/>`_
* `EmoFilm <https://zenodo.org/record/1326428>`_
* `EmoryNLP <https://github.com/declare-lab/MELD/>`_
* `EmoV-DB <https://github.com/numediart/EmoV-DB/>`_
* `EMOVO <http://voice.fub.it/activities/corpora/emovo/index.html>`_
* `ESD <https://github.com/HLTSingapore/Emotional-Speech-Data/>`_
* `eNTERFACE <http://www.enterface.net/results/>`_
* `IEMOCAP <https://sail.usc.edu/iemocap/>`_
* `JL-corpus <https://www.kaggle.com/tli725/jl-corpus>`_
* `MELD <https://github.com/declare-lab/MELD/>`_
* `MESD <https://data.mendeley.com/datasets/cy34mh68j9/3>`_
* `MESS <https://zenodo.org/record/3813437>`_
* `MLendSND
  <https://www.kaggle.com/datasets/jesusrequena/mlend-spoken-numerals>`_
* `MSP-IMPROV
  <https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html>`_
* `MSP-Podcast
  <https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html>`_
* `Oréau <https://zenodo.org/record/4405783>`_
* `Portuguese <https://link.springer.com/article/10.3758/BRM.42.1.74>`_
* `RAVDESS <https://zenodo.org/record/1188976>`_
* `SAVEE <http://kahlan.eps.surrey.ac.uk/savee/>`_
* `SEMAINE <https://semaine-db.eu/>`_
* `ShEMO <https://github.com/mansourehk/ShEMO>`_
* `SmartKom
  <https://clarin.phonetik.uni-muenchen.de/BASRepository/index.php>`_
* `SUBESCO <https://zenodo.org/record/4526477>`_
* `TESS <https://tspace.library.utoronto.ca/handle/1807/24487/>`_
* `URDU <https://github.com/siddiquelatif/URDU-Dataset/>`_
* `VENEC (Public subset)
  <https://www.nature.com/articles/s41562-019-0533-6>`_. The full
  dataset is available on request.
* `VIVAE <https://zenodo.org/record/4066235>`_


Standardising datasets
----------------------
Each dataset has a subdirectory in the ``datasets`` directory. This
subdirectory contains a ``process.py`` script, which takes a path to the
original dataset, and converts it to a simple standard format, consisting
of annotation CSVs and audio files. The script also generates a
``corpus.yaml`` file, which contains metadata about the dataset.

You can run the ``process.py`` script for a dataset, such as EMO-DB, as
follows::

    cd datasets/EMO-DB
    python process.py /path/to/EMO-DB
