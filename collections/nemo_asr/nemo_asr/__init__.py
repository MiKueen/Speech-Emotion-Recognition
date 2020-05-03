# Copyright 2019 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from nemo.core import Backend

from .audio_preprocessing import *
from .beam_search_decoder import BeamSearchDecoderWithLM
from .data_layer import (
        AudioToTextDataLayer, KaldiFeatureDataLayer, TranscriptDataLayer)
from .greedy_ctc_decoder import GreedyCTCDecoder
from .jasper import JasperEncoder, JasperDecoderForCTC
from .las.misc import JasperRNNConnector
from .losses import CTCLossNM

__all__ = ['Backend',
           'AudioToTextDataLayer',
           'AudioPreprocessing',
           'AudioPreprocessor',
           'AudioToMFCCPreprocessor',
           'AudioToMelSpectrogramPreprocessor',
           'AudioToSpectrogramPreprocessor',
           'MultiplyBatch',
           'SpectrogramAugmentation',
           'KaldiFeatureDataLayer',
           'TranscriptDataLayer',
           'GreedyCTCDecoder',
           'BeamSearchDecoderWithLM',
           'JasperEncoder',
           'JasperDecoderForCTC',
           'JasperRNNConnector',
           'CTCLossNM']


name = "nemo_asr"
backend = Backend.PyTorch
__version__ = "0.9.0"
