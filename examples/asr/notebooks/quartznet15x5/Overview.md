# QuartzNet-15x5 

## Overview
QuarzNet is a Jasper-like network which uses separable convolutions and larger filter sizes. It has comparable accuracy to Jasper while having much fewer parameters.

## Version
This version should be used with NeMo 0.9 and above

## Datasets
LibriSpeech +- 10% speed perturbation and Mozilla’s Common Voice En (Full Validated Set - validated.tsv) Note that this includes their train, dev and test splits.

#### Display Name
- QuartzNet15x5

#### Owner
- okuchaiev@nvidia.com

#### BuiltBy
- built by Nvidia
 #### Publisher
- Nvidia
 #### Application
- speech recognition

#### Training Framework
- PyTorch with NeMo

#### Inference Framework
- PyTorch

#### GPU Model - Alex to get back after discussion with TensorRT team?
- V100

#### Precision
- FP16

#### Description
- QuartzNet15x5 Encoder and Decoder checkpoints trained with NeMo. NVIDIA’s Apex/Amp O1 optimization level was used for training on V100 GPUs.

#### Labels
- lower-cased English characters and ‘ (apostrophe) 

#### Version
- 0.1

#### Accuracy - free text (key:value)?
- LibriSpeech Dev-Clean (Greedy) 3.98%
- LibriSpeech Dev-Other (Greedy) 10.84%

#### Batch Size
- 8 GPUs, 64 per GPU
