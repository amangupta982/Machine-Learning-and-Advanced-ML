import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import clean_text
from feature_engineering import extra_features
from model import train_gmm
from predict import load_model, predict_style

# ---------------------------------
# SAMPLE DATA
# ---------------------------------

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

# ---------------------------------
# NLP PREPROCESSING
# ---------------------------------

cleaned_texts = [clean_text(t) for t in texts]

vectorizer = TfidfVectorizer(max_features=1000)
tfidf_features = vectorizer.fit_transform(cleaned_texts).toarray()

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ---------------------------------
# EXTRA STYLE FEATURES
# ---------------------------------

extra_feat = extra_features(cleaned_texts)

# Combine features
X = np.hstack((tfidf_features, extra_feat))

# ---------------------------------
# TRAIN MODEL (EM-GMM)
# ---------------------------------

gmm = train_gmm(X)

print("Model trained successfully!")

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