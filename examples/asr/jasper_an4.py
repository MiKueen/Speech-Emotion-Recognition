# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
import os

from ruamel.yaml import YAML

import nemo
import nemo.utils.argparse as nm_argparse
from nemo.utils.lr_policies import CosineAnnealing
import nemo_asr
from nemo_asr.helpers import monitor_asr_train_progress, \
    process_evaluation_batch, process_evaluation_epoch, word_error_rate, \
    post_process_predictions, post_process_transcripts


def main():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()],
        description='AN4 ASR',
        conflict_handler='resolve')

    # Overwrite default args
    parser.add_argument("--train_dataset", type=str,
                        help="training dataset path")
    parser.add_argument("--eval_datasets", type=str, nargs=1,
                        help="validation dataset path")

    # Create new args
    parser.add_argument("--lm", default="./an4-lm.3gram.binary", type=str)
    parser.add_argument("--test_after_training", action='store_true')
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--beta1", default=0.95, type=float)
    parser.add_argument("--beta2", default=0.25, type=float)
    parser.set_defaults(
        model_config="./configs/jasper_an4.yaml",
        train_dataset="/home/mrjenkins/TestData/an4_dataset/an4_train.json",
        eval_datasets="/home/mrjenkins/TestData/an4_dataset/an4_val.json",
        work_dir="./tmp",
        checkpoint_dir="./tmp",
        optimizer="novograd",
        num_epochs=50,
        batch_size=32,
        eval_batch_size=16,
        lr=0.02,
        weight_decay=0.005,
        checkpoint_save_freq=1000,
        eval_freq=100,
        amp_opt_level="O1"
    )

    args = parser.parse_args()
    betas = (args.beta1, args.beta2)

    wer_thr = 0.20
    beam_wer_thr = 0.15

    nf = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        random_seed=0,
        log_dir=args.work_dir,
        checkpoint_dir=args.checkpoint_dir,
        create_tb_writer=True,
        cudnn_benchmark=args.cudnn_benchmark
    )
    tb_writer = nf.tb_writer
    checkpoint_dir = nf.checkpoint_dir
    args.checkpoint_dir = nf.checkpoint_dir

    # Load model definition
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        jasper_params = yaml.load(f)

    vocab = jasper_params['labels']
    sample_rate = jasper_params['sample_rate']

    # build train and eval model
    train_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
    train_dl_params.update(jasper_params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]

    data_layer = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.train_dataset,
        sample_rate=sample_rate,
        labels=vocab,
        batch_size=args.batch_size,
        **train_dl_params
    )

    num_samples = len(data_layer)
    total_steps = int(num_samples * args.num_epochs / args.batch_size)
    print("Train samples=", num_samples, "num_steps=", total_steps)

    data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
        sample_rate=sample_rate,
        **jasper_params["AudioToMelSpectrogramPreprocessor"]
    )

    # data_augmentation = nemo_asr.SpectrogramAugmentation(
    #     **jasper_params['SpectrogramAugmentation']
    # )

    eval_dl_params = copy.deepcopy(jasper_params["AudioToTextDataLayer"])
    eval_dl_params.update(jasper_params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]

    data_layer_eval = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.eval_datasets,
        sample_rate=sample_rate,
        labels=vocab,
        batch_size=args.eval_batch_size,
        **eval_dl_params
    )

    num_samples = len(data_layer_eval)
    nf.logger.info(f"Eval samples={num_samples}")

    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=jasper_params["AudioToMelSpectrogramPreprocessor"]["features"],
        **jasper_params["JasperEncoder"])

    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=jasper_params["JasperEncoder"]["jasper"][-1]["filters"],
        num_classes=len(vocab))

    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(vocab))

    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    # Training model
    audio, audio_len, transcript, transcript_len = data_layer()
    processed, processed_len = data_preprocessor(
        input_signal=audio, length=audio_len)
    encoded, encoded_len = jasper_encoder(
        audio_signal=processed, length=processed_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    predictions = greedy_decoder(log_probs=log_probs)
    loss = ctc_loss(log_probs=log_probs, targets=transcript,
                    input_length=encoded_len, target_length=transcript_len)

    # Evaluation model
    audio_e, audio_len_e, transcript_e, transcript_len_e = data_layer_eval()
    processed_e, processed_len_e = data_preprocessor(
        input_signal=audio_e, length=audio_len_e)
    encoded_e, encoded_len_e = jasper_encoder(
        audio_signal=processed_e, length=processed_len_e)
    log_probs_e = jasper_decoder(encoder_output=encoded_e)
    predictions_e = greedy_decoder(log_probs=log_probs_e)
    loss_e = ctc_loss(log_probs=log_probs_e, targets=transcript_e,
                      input_length=encoded_len_e,
                      target_length=transcript_len_e)
    nf.logger.info(
        "Num of params in encoder: {0}".format(jasper_encoder.num_weights))

    # Callbacks to print info to console and Tensorboard
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss, predictions, transcript, transcript_len],
        print_func=lambda x: monitor_asr_train_progress(x, labels=vocab),
        get_tb_values=lambda x: [["loss", x[0]]],
        tb_writer=tb_writer,
    )

    checkpointer_callback = nemo.core.CheckpointCallback(
        folder=checkpoint_dir, step_freq=args.checkpoint_save_freq)

    eval_tensors = [loss_e, predictions_e, transcript_e, transcript_len_e]
    eval_callback = nemo.core.EvaluatorCallback(
        eval_tensors=eval_tensors,
        user_iter_callback=lambda x, y: process_evaluation_batch(
            x, y, labels=vocab),
        user_epochs_done_callback=process_evaluation_epoch,
        eval_step=args.eval_freq,
        tb_writer=tb_writer)

    nf.train(
        tensors_to_optimize=[loss],
        callbacks=[train_callback, eval_callback, checkpointer_callback],
        optimizer=args.optimizer,
        lr_policy=CosineAnnealing(total_steps=total_steps),
        optimization_params={
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "momentum": args.momentum,
            "betas": betas,
            "weight_decay": args.weight_decay,
            "grad_norm_clip": None},
        batches_per_step=args.iter_per_step
    )

    if args.test_after_training:
        # Create BeamSearch NM
        beam_search_with_lm = nemo_asr.BeamSearchDecoderWithLM(
            vocab=vocab,
            beam_width=64,
            alpha=2.,
            beta=1.5,
            lm_path=args.lm,
            num_cpus=max(os.cpu_count(), 1))
        beam_predictions = beam_search_with_lm(
            log_probs=log_probs_e, log_probs_length=encoded_len_e)
        eval_tensors.append(beam_predictions)

        evaluated_tensors = nf.infer(eval_tensors)
        greedy_hypotheses = post_process_predictions(
            evaluated_tensors[1], vocab)
        references = post_process_transcripts(
            evaluated_tensors[2], evaluated_tensors[3], vocab)
        wer = word_error_rate(
            hypotheses=greedy_hypotheses, references=references)
        nf.logger.info("Greedy WER: {:.2f}".format(wer * 100))
        assert wer <= wer_thr, (
            "Final eval greedy WER {:.2f}% > than {:.2f}%".format(
                wer*100, wer_thr*100))

        beam_hypotheses = []
        # Over mini-batch
        for i in evaluated_tensors[-1]:
            # Over samples
            for j in i:
                beam_hypotheses.append(j[0][1])

        beam_wer = word_error_rate(
            hypotheses=beam_hypotheses, references=references)
        nf.logger.info("Beam WER {:.2f}%".format(beam_wer * 100))
        assert beam_wer <= beam_wer_thr, (
            "Final eval beam WER {:.2f}%  > than {:.2f}%".format(
                beam_wer*100, beam_wer_thr*100))
        assert beam_wer <= wer, (
            "Final eval beam WER > than the greedy WER.")

        # Reload model weights and train for extra 10 epochs
        checkpointer_callback = nemo.core.CheckpointCallback(
            folder=checkpoint_dir, step_freq=args.checkpoint_save_freq,
            force_load=True)

        nf.reset_trainer()
        nf.train(
            tensors_to_optimize=[loss],
            callbacks=[train_callback, checkpointer_callback],
            optimizer=args.optimizer,
            optimization_params={
                "num_epochs": args.num_epochs + 10,
                "lr": args.lr,
                "momentum": args.momentum,
                "betas": betas,
                "weight_decay": args.weight_decay,
                "grad_norm_clip": None},
            reset=True
        )

        evaluated_tensors = nf.infer(eval_tensors[:-1])
        greedy_hypotheses = post_process_predictions(
            evaluated_tensors[1], vocab)
        references = post_process_transcripts(
            evaluated_tensors[2], evaluated_tensors[3], vocab)
        wer_new = word_error_rate(
            hypotheses=greedy_hypotheses, references=references)
        nf.logger.info("New greedy WER: {:.2f}%".format(wer_new * 100))
        assert wer_new <= wer * 1.1, (
            f"Fine tuning: new WER {wer * 100:.2f}% > than the previous WER "
            f"{wer_new * 100:.2f}%")


if __name__ == "__main__":
    main()
