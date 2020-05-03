from __future__ import unicode_literals, print_function

from flask import Blueprint, request, jsonify
from pathlib import Path
import io

from snips_nlu import SnipsNLUEngine, load_resources
from snips_nlu.default_configs import CONFIG_EN

import json
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import re

from werkzeug.utils import secure_filename

from app import app, data_preprocessor, jasper_encoder, jasper_decoder, \
    greedy_decoder, neural_factory, MODEL_YAML, WORK_DIR, ENABLE_NGRAM
try:
    from app import beam_search_with_lm
except ImportError:
    print("Not using Beam Search Decoder with LM")
    ENABLE_NGRAM = False
import nemo
import nemo_asr


config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)


# Load dataset for intent detection
try:
    with io.open(str(Path('app', 'static', 'samples', 'dataset.json'))) as fr:
        dataset = json.load(fr)
except Exception as e:
    print("Could not load dataset {}".format(str(e)))

# Train dataset    
nlu_engine = SnipsNLUEngine(resources=load_resources("snips_nlu_en"))
nlu_engine.fit(dataset)


# Load class names
print("[INFO]: Loading Classes")
classNames = np.load("/NeMo/examples/applications/asr_service/app/emotion-classification/new_classes.npy")

# Load tokenizer pickle file
print("[INFO]: Loading Tokens")
with open('/NeMo/examples/applications/asr_service/app/emotion-classification/new_tokenizer.pickle', 'rb') as handle:
        Tokenizer = pickle.load(handle)

# Load model
print("[INFO]: Loading Model")
model = tf.keras.models.load_model("/NeMo/examples/applications/asr_service/app/emotion-classification/new_model.h5", compile=False)
model._make_predict_function()


# Preprocess text
print("[INFO]: Preprocessing")
MAX_LENGTH = maxlen = 400

def preprocess_text(sen):
    
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


def wav_to_text(manifest, greedy=True):
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(MODEL_YAML) as f:
        jasper_model_definition = yaml.load(f)
    labels = jasper_model_definition['labels']

    # Instantiate necessary neural modules
    data_layer = nemo_asr.AudioToTextDataLayer(
        shuffle=False,
        manifest_filepath=manifest,
        labels=labels, batch_size=1)

    # Define inference DAG
    audio_signal, audio_signal_len, _, _ = data_layer()
    processed_signal, processed_signal_len = data_preprocessor(
        input_signal=audio_signal,
        length=audio_signal_len)
    encoded, encoded_len = jasper_encoder(audio_signal=processed_signal,
                                          length=processed_signal_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    predictions = greedy_decoder(log_probs=log_probs)

    if ENABLE_NGRAM:
        print('Running with beam search')
        beam_predictions = beam_search_with_lm(
            log_probs=log_probs, log_probs_length=encoded_len)
        eval_tensors = [beam_predictions]

    if greedy:
        eval_tensors = [predictions]

    tensors = neural_factory.infer(tensors=eval_tensors)
    if greedy:
        from nemo_asr.helpers import post_process_predictions
        prediction = post_process_predictions(tensors[0], labels)
    else:
        prediction = tensors[0][0][0][0][1]
    return prediction


result_template = """
<html>
<h3 align="center">Transcription Result</h3>
   <body style="border:3px solid green">
   <div align="center">
   <p>Transcription time: {0}</p>
   <p>Transcripted text: {1}</p>
   <p>Predicted Emotion: {2}</p>
   <p>Predicted Intent: {3}</p>
   </div>
   </body>
</html>
"""


@app.route('/predict', methods=['GET', 'POST'])
def transcribe_file():
    if request.method == 'POST':
        # upload wav_file to work directory
        f = request.files['file']
        greedy = True
        if request.form.get('beam'):
            if not ENABLE_NGRAM:
                return ("Error: Beam Search with ngram LM is not enabled "
                        "on this server")
            greedy = False
        file_path = os.path.join(WORK_DIR, secure_filename(f.filename))
        f.save(file_path)
        # create manifest
        manifest = dict()
        manifest['audio_filepath'] = file_path
        manifest['duration'] = 18000
        manifest['text'] = 'todo'
        with open(file_path+".json", 'w') as fout:
            fout.write(json.dumps(manifest))
        start_t = time.time()
        transcription = wav_to_text(file_path + ".json", greedy=greedy)
        total_t = time.time() - start_t
        
        required_output = dict()
        
        # Tokenize and pad sentence
        with session.as_default():
            with session.graph.as_default():
                sentence_processed = Tokenizer.texts_to_sequences(transcription)
                sentence_processed = np.array(sentence_processed)
                sentence_padded = tf.keras.preprocessing.sequence.pad_sequences(sentence_processed, padding='post', maxlen=MAX_LENGTH)
                predicted_class = model.predict(sentence_padded)
                intent_text = "".join(transcription)
                parsing = nlu_engine.parse(intent_text)
                intent_result = json.loads(json.dumps(parsing, indent=2))
                required_output['intent'] = intent_result.get('intent')
                required_output['slots'] = intent_result.get('slots')
                
        result = result_template.format(total_t, transcription, classNames[np.argmax(predicted_class)],required_output)
        return str(result)
    else:
        return str("Success!!")


@app.route('/')
@app.route('/index')
def index():
    return "Hello from NeMo ASR webservice!"
