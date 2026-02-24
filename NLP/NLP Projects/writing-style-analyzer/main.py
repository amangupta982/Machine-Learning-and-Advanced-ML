import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import clean_text
from feature_engineering import extra_features
from model import train_gmm
from predict import predict_style

MODEL_PATH = "gmm_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ---------------------------------
# FUNCTION: TRAIN MODEL (ONLY ONCE)
# ---------------------------------

def train_and_save_model():

    print("Training model...")

    texts = [
        "The research paper presents a detailed analysis of data models.",
        "Hey bro that movie was awesome lol",
        "The algorithm optimizes system performance efficiently.",
        "I feel so happy and excited today!",
        "This study evaluates statistical methods.",
        "haha that was funny man",
        "Machine learning model improves accuracy.",
        "I love this amazing experience so much"
    ]

    cleaned_texts = [clean_text(t) for t in texts]

    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_features = vectorizer.fit_transform(cleaned_texts).toarray()

    # Save vectorizer
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    extra_feat = extra_features(cleaned_texts)
    X = np.hstack((tfidf_features, extra_feat))

    gmm = train_gmm(X)

    print("Model trained and saved successfully!")

    return gmm, vectorizer


# ---------------------------------
# FUNCTION: LOAD MODEL
# ---------------------------------

def load_existing_model():

    print("Loading existing model...")

    with open(MODEL_PATH, "rb") as f:
        gmm = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return gmm, vectorizer


# ---------------------------------
# MAIN LOGIC
# ---------------------------------

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    gmm, vectorizer = load_existing_model()
else:
    gmm, vectorizer = train_and_save_model()


# ---------------------------------
# TEST PREDICTION
# ---------------------------------

test_text = "This algorithm improves data analysis performance"

cleaned = clean_text(test_text)
vec = vectorizer.transform([cleaned]).toarray()
extra = extra_features([cleaned])

final_vec = np.hstack((vec, extra))

prediction = predict_style(gmm, final_vec)

print("Predicted Writing Style:", prediction)