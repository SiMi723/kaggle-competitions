import pickle
import json
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "models/bilstm_best.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

TOKENIZER_PATH = "models/tokenizer.pkl"

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("Tokenizer loaded successfully.")


# --------------------------------------------------
# Load configuration
# --------------------------------------------------

CONFIG_PATH = "models/config.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MAX_LEN = config["max_len"]

print("Configuration loaded successfully.")
print("MAX_LEN:", MAX_LEN)


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep alphabets, numbers and basic punctuation
    text = re.sub(
        r"[^a-zA-Z0-9\s!?.,']",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_question(question):

    # Clean text
    cleaned_question = clean_text(question)

    # Convert text to integer sequence
    sequence = tokenizer.texts_to_sequences(
        [cleaned_question]
    )

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Model prediction
    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # Classification threshold
    prediction = int(probability >= 0.5)

    return prediction, float(probability)


# --------------------------------------------------
# Test inference
# --------------------------------------------------

if __name__ == "__main__":

    test_questions = [
        "Why do people believe conspiracy theories?",
        "What is the capital of France?",
        "How can I improve my programming skills?",
        "Why are people so emotional?"
    ]

    for question in test_questions:

        prediction, probability = predict_question(question)

        label = (
            "Insincere"
            if prediction == 1
            else "Sincere"
        )

        print("\nQuestion:", question)
        print("Prediction:", label)
        print("Probability:", probability)