import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import time
import base64

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# ============================================
# ✅ CONFIG
# ============================================

MODEL_PATH = "cricket_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_SEQ_LEN = 15

# background image (put inside project folder)
BACKGROUND_IMAGE = "stadium.jpg"

# avatar image
AVATAR_IMAGE = "ai_commentator.png"

# sound file
SOUND_FILE = "success.mp3"


# ============================================
# ✅ LOAD MODEL + TOKENIZER
# ============================================

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


model, tokenizer = load_resources()


# ============================================
# ✅ BACKGROUND IMAGE FUNCTION
# ============================================

def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_bg(BACKGROUND_IMAGE)


# ============================================
# ✅ DARK THEME CSS
# ============================================

st.markdown("""
<style>

.stApp {
    color: white;
}

h1 {
    color: #c084fc;
    text-align: center;
}

.block-container {
    background: rgba(15,15,30,0.85);
    padding: 2rem;
    border-radius: 15px;
}

.stButton>button {
    background-color: #7c4dff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.result-box {
    background-color: #1a1a2e;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #7c4dff;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# ✅ TOP-K SAMPLING
# ============================================

def top_k_sampling(predictions, k=8, temperature=0.8):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-9) / temperature

    top_k_indices = predictions.argsort()[-k:]
    top_k_probs = np.exp(predictions[top_k_indices])
    top_k_probs /= np.sum(top_k_probs)

    return np.random.choice(top_k_indices, p=top_k_probs)


# ============================================
# ✅ GENERATE COMMENTARY
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
# ✅ TYPING EFFECT
# ============================================

def typing_effect(text):
    placeholder = st.empty()
    typed_text = ""

    for word in text.split():
        typed_text += word + " "
        placeholder.markdown(
            f"<div class='result-box'>{typed_text}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.05)


# ============================================
# ✅ STREAMLIT UI
# ============================================

st.title("🏏 AI Cricket Commentary Generator")

col1, col2 = st.columns([1, 5])

with col1:
    st.image(AVATAR_IMAGE, width=90)

with col2:
    st.write("AI Commentator powered by LSTM Neural Language Model")

st.divider()

user_input = st.text_input(
    "Enter starting commentary",
    placeholder="e.g. virat hits a"
)

num_words = st.slider("Words to generate", 5, 40, 20)


if st.button("Generate Commentary"):

    if user_input.strip() == "":
        st.warning("Please enter a starting sentence.")
    else:
        with st.spinner("AI commentator is thinking..."):
            result = generate_commentary(user_input, num_words)

        typing_effect(result)

        # play sound after generation
        audio_file = open(SOUND_FILE, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")