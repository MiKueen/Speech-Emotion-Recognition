
Emotion Recognition and Intent Detection through Speech
=======================================================

This project recognises emotions through speech. It also detects intent and extract entities from the message. It uses NeMo as its base for speech recognition and Snips NLU for intent detection.


NeMo
~~~~

NeMo (Neural Modules) is a toolkit for creating AI applications using **neural modules** - conceptual blocks of neural networks that take *typed* inputs and produce *typed* outputs. Such modules typically represent data layers, encoders, decoders, language models, loss functions, or methods of combining activations.

NeMo consists of: 

* **NeMo Core**: fundamental building blocks for all neural models and type system.
* **NeMo collections**: pre-built neural modules for particular domains such as automatic speech recognition (nemo_asr), natural language processing (nemo_nlp) and text synthesis (nemo_tts).


**Introduction**

See `this video <https://nvidia.github.io/NeMo/>`_ for a quick walk-through.

**Requirements**

1) Python 3.6 or 3.7
2) PyTorch 1.4.* with GPU support
3) (optional for best performance) NVIDIA APEX. Install from here: https://github.com/NVIDIA/apex

**Getting started**

THE LATEST STABLE VERSION OF NeMo is **0.9.0** (which is available via PIP).

NVIDIA's `NGC PyTorch container <https://ngc.nvidia.com/catalog/containers/nvidia:pytorch>`_ is recommended as it already includes all the requirements above.

* Pull the docker: ``docker pull nvcr.io/nvidia/pytorch:19.11-py3``
* Run: ``docker run --runtime=nvidia -it -v <nemo_github_folder_path>:/NeMo --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:19.11-py3``

.. code-block:: bash

    pip install nemo-toolkit  # installs NeMo Core
    pip install nemo-asr # installs NeMo ASR collection
    pip install nemo-nlp # installs NeMo NLP collection
    pip install nemo-tts # installs NeMo TTS collection


**Documentation**

`NeMo documentation <https://nvidia.github.io/NeMo/>`_

See `examples/start_here` to get started with the simplest example. The folder `examples` contains several examples to get you started with various tasks in NLP and ASR.


**Tutorials**

* `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
* `Natural language processing <https://nvidia.github.io/NeMo/nlp/intro.html>`_


Snips NLU
~~~~~~~~~

Snips NLU (Natural Language Understanding) is a Python library that allows to extract structured information from sentences written in natural language.
The NLU engine first detects what the intention of the user is (a.k.a. intent), then extracts the parameters (called slots) of the query. The developer can then use this to determine the appropriate action or response.

.. code-block:: bash

    pip install snips-nlu  # installs snips-nlu
    python -m snips_nlu download en  # installs language resource for snips


**Running Application**

* Using pretrained model:

1) From ``<nemo_git_root>/examples/applications/asr_service`` folder do: ``export FLASK_APP=asr_service.py`` and start service: ``flask run --host=0.0.0.0 --port=6006``
2) Open ``recognize.html`` with any browser and upload a .wav file

*Note: the service will only work correctly with single channel 16Khz .wav files*.

* Training own model on Librispeech data:

1) Get data

These scripts will download and convert LibriSpeech into format expected by nemo_asr:

.. code-block:: bash

    # note that this script requires sox to be installed
    # to install sox on Ubuntu, simply do: sudo apt-get install sox
    # and then: pip install sox
    # get_librispeech_data.py script is located under <nemo_git_repo_root>/scripts
    python get_librispeech_data.py --data_root=data --data_set=dev_clean,train_clean_100
    # To get all LibriSpeech data, do:
    # python get_librispeech_data.py --data_root=data --data_set=ALL

After download and conversion, your data folder should contain 2 json files:

- dev_clean.json
- train_clean_100.json

2) Move to ``<nemo_git_root>/examples/asr/notebooks`` folder and run ``ASR using Librispeech dataset.ipynb`` file.
3) Now follow steps 1 and 2 of pretrained model to see the application in service.


Author
~~~~~~

MiKueen: https://github.com/MiKueen


Citation
~~~~~~~~

@misc{nemo2019,
    title={NeMo: a toolkit for building AI applications using Neural Modules},
    author={Oleksii Kuchaiev and Jason Li and Huyen Nguyen and Oleksii Hrinchuk and Ryan Leary and Boris Ginsburg and Samuel Kriman and Stanislav Beliaev and Vitaly Lavrukhin and Jack Cook and Patrice Castonguay and Mariya Popova and Jocelyn Huang and Jonathan M. Cohen},
    year={2019},
    eprint={1909.09577},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
