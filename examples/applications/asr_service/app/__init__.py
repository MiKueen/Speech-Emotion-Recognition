# Copyright (c) 2019 NVIDIA Corporation
import os
from flask import Flask
from ruamel.yaml import YAML

import nemo
import nemo_asr


app = Flask(__name__)
# make sure WORK_DIR exists before calling your service
# in this folder, the service will store received .wav files and constructed
# .json manifests
#WORK_DIR = "/NeMo/examples/applications/asr_service/app/workdir"
#MODEL_YAML = "/NeMo/examples/asr/configs/jasper_an4.yaml"
#CHECKPOINT_ENCODER = "/NeMo/examples/asr/notebooks/an4_checkpoints/JasperEncoder-STEP-3000.pt"
#CHECKPOINT_DECODER = "/NeMo/examples/asr/notebooks/an4_checkpoints/JasperDecoderForCTC-STEP-3000.pt"

WORK_DIR = "/NeMo/examples/applications/asr_service/app/librispeech_workdir"
MODEL_YAML = "/NeMo/examples/asr/configs/quartznet15x5.yaml"
CHECKPOINT_ENCODER = "/NeMo/examples/asr/notebooks/quartznet15x5/JasperEncoder-STEP-247400.pt"
CHECKPOINT_DECODER = "/NeMo/examples/asr/notebooks/quartznet15x5/JasperDecoderForCTC-STEP-247400.pt"

# Set this to True to enable beam search decoder
ENABLE_NGRAM = False
# This is only necessary if ENABLE_NGRAM = True. Otherwise, set to empty string
LM_PATH = ""


# Read model YAML
yaml = YAML(typ="safe")
with open(MODEL_YAML) as f:
    jasper_model_definition = yaml.load(f)
labels = jasper_model_definition['labels']

import torch
from nemo.core import DeviceType

# Instantiate necessary Neural Modules
# Note that data layer is missing from here
neural_factory = nemo.core.NeuralModuleFactory(
    placement=(DeviceType.GPU if torch.cuda.is_available() else DeviceType.CPU),
    backend=nemo.core.Backend.PyTorch)
data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
        factory=neural_factory)
jasper_encoder = nemo_asr.JasperEncoder(
    jasper=jasper_model_definition['JasperEncoder']['jasper'],
    activation=jasper_model_definition['JasperEncoder']['activation'],
    feat_in=jasper_model_definition[
        'AudioToMelSpectrogramPreprocessor']['features'])
jasper_encoder.restore_from(CHECKPOINT_ENCODER, local_rank=0)
jasper_decoder = nemo_asr.JasperDecoderForCTC(
    feat_in=1024,
    num_classes=len(labels))
jasper_decoder.restore_from(CHECKPOINT_DECODER, local_rank=0)
greedy_decoder = nemo_asr.GreedyCTCDecoder()

if ENABLE_NGRAM and os.path.isfile(LM_PATH):
    beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
        vocab=labels,
        beam_width=64,
        alpha=2.0,
        beta=1.0,
        lm_path=LM_PATH,
        num_cpus=max(os.cpu_count(), 1))
else:
    print("Beam search is not enabled")

from app import routes  # noqa
if __name__ == '__main__':
    app.run()


