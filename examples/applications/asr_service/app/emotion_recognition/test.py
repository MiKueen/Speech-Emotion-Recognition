import numpy as np
import tensorflow as tf
import pickle
import re

# Your input sentence
sentence = ["im feeling down today, the day was bad"]

# Load class names
print("[INFO]: Loading Classes")
classNames = np.load("/kueen/NeMo/examples/applications/asr_service/app/emotion-classification/new_classes.npy")

# Load tokenizer pickle file
print("[INFO]: Loading Tokens")
with open('/NeMo/examples/applications/asr_service/app/emotion-classification/new_tokenizer.pickle', 'rb') as handle:
        Tokenizer = pickle.load(handle)

# Load model
print("[INFO]: Loading Model")
model = tf.keras.models.load_model("/NeMo/examples/applications/asr_service/app/emotion-classification/new_model.h5", compile=False)

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


# Tokenize and pad sentence
sentence_processed = Tokenizer.texts_to_sequences(sentence)
sentence_processed = np.array(sentence_processed)
sentence_padded = tf.keras.preprocessing.sequence.pad_sequences(sentence_processed, padding='post', maxlen=MAX_LENGTH)

# Get prediction for sentence
print("""[INFO]: Prediction\n\t{}""".format(sentence[0]))
result = model.predict(sentence_padded)

# Show prediction
print("-"*20)
print("[INFO]: Emotion class for given text is: {}".format(classNames[np.argmax(result)]))

