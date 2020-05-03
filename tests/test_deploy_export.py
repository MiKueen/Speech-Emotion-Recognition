# Copyright (c) 2019 NVIDIA Corporation
import os
from pathlib import Path

import torch
from ruamel.yaml import YAML

from .common_setup import NeMoUnitTest
from .context import nemo, nemo_asr, nemo_nlp


class TestDeployExport(NeMoUnitTest):
    def setUp(self) -> None:
        self.nf = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.GPU)

    def __test_export_route(self, module, out_name, mode,
                            input_example=None):
        out = Path(out_name)
        if out.exists():
            os.remove(out)

        self.nf.deployment_export(
            module=module,
            output=out_name,
            input_example=input_example,
            d_format=mode)

        self.assertTrue(out.exists())
        if out.exists():
            os.remove(out)

    def test_simple_module_export(self):
        simplest_module = \
            nemo.backends.pytorch.tutorials.TaylorNet(dim=4, factory=self.nf)
        self.__test_export_route(module=simplest_module,
                                 out_name="simple.pt",
                                 mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
                                 input_example=None)

    def test_TokenClassifier_module_export(self):
        t_class = nemo_nlp.TokenClassifier(hidden_size=512, num_classes=16,
                                           use_transformer_pretrained=False)
        self.__test_export_route(module=t_class,
                                 out_name="t_class.pt",
                                 mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
                                 input_example=torch.randn(16, 16, 512).cuda())

    def test_TokenClassifier_module_onnx_export(self):
        t_class = nemo_nlp.TokenClassifier(hidden_size=512, num_classes=16,
                                           use_transformer_pretrained=False)
        self.__test_export_route(module=t_class,
                                 out_name="t_class.onnx",
                                 mode=nemo.core.DeploymentFormat.ONNX,
                                 input_example=torch.randn(16, 16, 512).cuda())

    def test_jasper_decoder_export_ts(self):
        j_decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024,
                                                 num_classes=33)
        self.__test_export_route(module=j_decoder,
                                 out_name="j_decoder.ts",
                                 mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
                                 input_example=None)

    def test_hf_bert_ts(self):
        bert = nemo_nlp.huggingface.BERT(
            pretrained_model_name="bert-base-uncased")
        input_example = (torch.randint(low=0, high=16, size=(2, 16)).cuda(),
                         torch.randint(low=0, high=1, size=(2, 16)).cuda(),
                         torch.randint(low=0, high=1, size=(2, 16)).cuda())
        self.__test_export_route(module=bert,
                                 out_name="bert.ts",
                                 mode=nemo.core.DeploymentFormat.TORCHSCRIPT,
                                 input_example=input_example)

    def test_hf_bert_pt(self):
        bert = nemo_nlp.huggingface.BERT(
            pretrained_model_name="bert-base-uncased")
        self.__test_export_route(module=bert,
                                 out_name="bert.pt",
                                 mode=nemo.core.DeploymentFormat.PYTORCH)

    def test_jasper_encoder_to_onnx(self):
        with open("tests/data/jasper_smaller.yaml") as file:
            yaml = YAML(typ="safe")
            jasper_model_definition = yaml.load(file)

        jasper_encoder = nemo_asr.JasperEncoder(
            conv_mask=False,
            feat_in=jasper_model_definition[
                'AudioToMelSpectrogramPreprocessor']['features'],
            **jasper_model_definition['JasperEncoder']
        )

        self.__test_export_route(module=jasper_encoder,
                                 out_name="jasper_encoder.onnx",
                                 mode=nemo.core.DeploymentFormat.ONNX,
                                 input_example=(
                                     torch.randn(16, 64, 256).cuda(),
                                     torch.randn(256).cuda()))
