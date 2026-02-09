import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import nltk
import re

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================
# ✅ DOWNLOAD NLTK DATA (RUNS ONCE)
# ============================================

nltk.download('punkt')


# ============================================
# ✅ FILE PATHS (MODEL + TOKENIZER)
# ============================================

DATA_FOLDER = "COMMENTARY_INTL_MATCH"
MODEL_PATH = "cricket_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"


# ============================================
# ✅ SMALL NLTK PREPROCESSING FUNCTION
# ============================================

def clean_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation
    text = re.sub(r'[^a-z\s]', '', text)

    # nltk tokenization
    tokens = word_tokenize(text)

    # join back to sentence
    return " ".join(tokens)


# ============================================
# ✅ STEP 1: LOAD CSV DATA
# ============================================

all_lines = []

print("\n📌 Reading CSV files...\n")

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        path = os.path.join(DATA_FOLDER, file)
        df = pd.read_csv(path)

        if "Commentary" in df.columns:
            all_lines.extend(df["Commentary"].dropna().astype(str).tolist())

print("✅ Total Raw Lines:", len(all_lines))


# ============================================
# ✅ STEP 2: LIMIT DATASET SIZE + CLEAN USING NLTK
# ============================================

MAX_LINES = 30000

corpus = [clean_text(line) for line in all_lines[:MAX_LINES]]

print("✅ Using Lines for Training:", len(corpus))


# ============================================
# ✅ STEP 3: TOKENIZATION
# ============================================

if os.path.exists(TOKENIZER_PATH):

    print("✅ Loading saved tokenizer...")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

else:

    print("📌 Creating tokenizer...")
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(corpus)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Tokenizer saved!")


total_words = len(tokenizer.word_index) + 1
print("✅ Vocabulary Size:", total_words)


# ============================================
# ✅ STEP 4: CREATE SEQUENCES
# ============================================

MAX_SEQ_LEN = 15
input_sequences = []

print("\n📌 Creating Training Sequences...\n")

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    token_list = token_list[:MAX_SEQ_LEN]

    for i in range(1, len(token_list)):
        input_sequences.append(token_list[:i+1])

input_sequences = pad_sequences(
    input_sequences,
    maxlen=MAX_SEQ_LEN,
    padding="pre"
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

print("✅ Training Samples:", X.shape[0])


# ============================================
# ✅ STEP 5: BUILD MODEL
# ============================================

def build_model():

    model = Sequential([
        Embedding(total_words, 100),

        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),

        LSTM(64),
        Dropout(0.2),

        Dense(total_words, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


# ============================================
# ✅ STEP 6: TRAIN OR LOAD MODEL
# ============================================

if os.path.exists(MODEL_PATH):

    print("\n✅ Loading saved model...\n")
    model = load_model(MODEL_PATH)

else:

    print("\n📌 Training model for first time...\n")

    model = build_model()
    model.summary()

    early_stop = EarlyStopping(monitor="loss", patience=2)

    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    model.save(MODEL_PATH)
    print("\n✅ Model saved successfully!")


print("\n✅ Model Ready!\n")


# ============================================
# ✅ STEP 7: TOP-K SAMPLING
# ============================================

def top_k_sampling(predictions, k=8, temperature=0.8):

    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-9) / temperature

    top_k_indices = predictions.argsort()[-k:]
    top_k_probs = np.exp(predictions[top_k_indices])
    top_k_probs /= np.sum(top_k_probs)

    return np.random.choice(top_k_indices, p=top_k_probs)


# ============================================
# ✅ STEP 8: GENERATION FUNCTION
# ============================================

def generate_commentary(seed_text, next_words=20):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=MAX_SEQ_LEN - 1,
            padding="pre"
        )

        predictions = model.predict(token_list, verbose=0)[0]

        predicted_index = top_k_sampling(predictions)

        output_word = tokenizer.index_word.get(predicted_index, "")
        seed_text += " " + output_word

    return seed_text


# ============================================
# ✅ STEP 9: INTERACTIVE GENERATOR
# ============================================

print("🏏 Cricket Commentary Generator Ready!\n")

while True:

    prompt = input("\nEnter prompt (or type 'exit'): ")

    if prompt.lower() == "exit":
        print("✅ Exiting...")
        break

    result = generate_commentary(prompt)

    print("\n🏏 Generated Commentary:\n")
    print(result)