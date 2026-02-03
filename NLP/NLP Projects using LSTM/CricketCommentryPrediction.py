import os
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ============================================
# ✅ STEP 1: LOAD SMALL SUBSET
# ============================================

DATA_FOLDER = "COMMENTARY_INTL_MATCH"

all_lines = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_FOLDER, file))
        if "Commentary" in df.columns:
            all_lines.extend(df["Commentary"].dropna().astype(str).tolist())

print("Total Lines:", len(all_lines))

# ✅ Only 8000 lines for fast training
corpus = [line.lower().strip() for line in all_lines[:8000]]
print("Using Lines:", len(corpus))


# ============================================
# ✅ STEP 2: TOKENIZATION
# ============================================

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

total_words = len(tokenizer.word_index) + 1
print("Vocabulary Size:", total_words)


# ============================================
# ✅ STEP 3: SIMPLE SEQUENCE CREATION
# ============================================

sequences = tokenizer.texts_to_sequences(corpus)

max_len = 12
X = pad_sequences(sequences, maxlen=max_len, padding="pre")

# Input = all except last word
y = X[:, -1]
X = X[:, :-1]

print("Training Samples:", X.shape)


# ============================================
# ✅ STEP 4: LIGHT MODEL (FAST)
# ============================================

model = Sequential([
    Embedding(total_words, 64, input_length=max_len-1),
    LSTM(64),
    Dense(total_words, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ============================================
# ✅ STEP 5: TRAIN FAST
# ============================================

stop = EarlyStopping(monitor="loss", patience=2)

model.fit(
    X, y,
    epochs=5,
    batch_size=32,
    callbacks=[stop]
)

print("\n✅ Training Done Fast!\n")


# ============================================
# ✅ STEP 6: GENERATION FUNCTION
# ============================================

def generate(seed, next_words=15):

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding="pre")

        predicted = np.argmax(model.predict(token_list, verbose=0))
        word = tokenizer.index_word.get(predicted, "")

        seed += " " + word

    return seed


# ============================================
# ✅ TEST
# ============================================

print(generate("rohit plays a", 20))