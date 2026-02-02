import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -----------------------------
# Load Dataset
# -----------------------------
with open("cricket_data.txt", "r") as file:
    data = file.read().lower().split("\n")

print("Total Lines:", len(data))

# -----------------------------
# Tokenization
# -----------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

total_words = len(tokenizer.word_index) + 1
print("Total Vocabulary Size:", total_words)

# -----------------------------
# Create Input Sequences
# -----------------------------
input_sequences = []

for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding
max_seq_len = max([len(seq) for seq in input_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")

# Split into X and y
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# -----------------------------
# Build LSTM Model
# -----------------------------
model = Sequential([
    Embedding(total_words, 100, input_length=max_seq_len-1),
    LSTM(150),
    Dense(total_words, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# -----------------------------
# Train Model
# -----------------------------
model.fit(X, y, epochs=50, verbose=1)