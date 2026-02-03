import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# ============================================
# 1. LOAD DATASET
# ============================================

print("\n📌 Loading Cricket Commentary Dataset...\n")

with open("cricket_data.txt", "r", encoding="utf-8") as file:
    data = file.read().lower().split("\n")

data = [line for line in data if line.strip() != ""]

print("✅ Total Commentary Lines:", len(data))


# ============================================
# 2. TOKENIZATION
# ============================================

print("\n📌 Tokenizing Text...\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

total_words = len(tokenizer.word_index) + 1
print("✅ Vocabulary Size:", total_words)


# ============================================
# 3. CREATE TRAINING SEQUENCES
# ============================================

print("\n📌 Creating Input Sequences...\n")

input_sequences = []

for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

max_seq_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences, maxlen=max_seq_len, padding="pre"
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print("✅ X Shape:", X.shape)
print("✅ y Shape:", y.shape)


# ============================================
# 4. BUILD BIDIRECTIONAL LSTM MODEL
# ============================================

print("\n📌 Building LSTM Model...\n")

model = Sequential([
    Embedding(total_words, 150, input_length=max_seq_len - 1),

    Bidirectional(LSTM(200, return_sequences=False)),

    Dense(total_words, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


# ============================================
# 5. TRAIN MODEL WITH EARLY STOPPING
# ============================================

print("\n📌 Training Model...\n")

early_stop = EarlyStopping(monitor="loss", patience=5)

model.fit(
    X, y,
    epochs=200,
    verbose=1,
    callbacks=[early_stop]
)


# ============================================
# 6. TEMPERATURE SAMPLING FUNCTION
# ============================================

def sample_with_temperature(predictions, temperature=1.0):
    """
    Temperature Sampling to avoid repetitive text.
    Higher temperature = more randomness
    Lower temperature = more accuracy
    """

    predictions = np.asarray(predictions).astype("float64")

    predictions = np.log(predictions + 1e-9) / temperature

    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)

    return np.random.choice(len(predictions), p=predictions)


# ============================================
# 7. TEXT GENERATION FUNCTION
# ============================================

def generate_commentary(seed_text, next_words=25, temperature=0.8):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences([seed_text])[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_seq_len - 1,
            padding="pre"
        )

        predictions = model.predict(token_list, verbose=0)[0]

        predicted_index = sample_with_temperature(
            predictions,
            temperature=temperature
        )

        output_word = tokenizer.index_word.get(predicted_index, "")

        seed_text += " " + output_word

    return seed_text


# ============================================
# 8. TEST THE MODEL
# ============================================

print("\n🎯 Cricket Commentary Generator Ready!\n")

while True:
    prompt = input("\nEnter Starting Commentary (or type 'exit'): ")

    if prompt.lower() == "exit":
        print("\n✅ Exiting Generator...")
        break

    result = generate_commentary(
        prompt,
        next_words=30,
        temperature=0.9
    )

    print("\n🏏 Generated Commentary:\n")
    print(result)