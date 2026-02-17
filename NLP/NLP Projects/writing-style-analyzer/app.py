import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from preprocess import clean_text
from feature_engineering import extra_features

# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------

gmm = pickle.load(open("gmm_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

style_map = {
    0: "Formal Writing",
    1: "Informal Writing",
    2: "Technical Writing",
    3: "Emotional Writing"
}

st.title("🧠 EM-GMM Writing Style Analyzer")

# -----------------------------
# INPUT TEXT BOX
# -----------------------------

user_text = st.text_area("Enter text to analyze:")

if st.button("Analyze Style"):

    cleaned = clean_text(user_text)
    vec = vectorizer.transform([cleaned]).toarray()
    extra = extra_features([cleaned])

    final_vec = np.hstack((vec, extra))

    # Prediction
    cluster = gmm.predict(final_vec)[0]
    probs = gmm.predict_proba(final_vec)[0]

    st.subheader("Predicted Style:")
    st.success(style_map[cluster])

    # -----------------------------
    # PROBABILITY SCORES
    # -----------------------------
    st.subheader("Style Probability Scores")

    for i, p in enumerate(probs):
        st.write(f"{style_map[i]} : {round(p*100,2)} %")

# -----------------------------
# CLUSTER VISUALIZATION
# -----------------------------

st.subheader("EM Cluster Visualization")

X = gmm.means_

pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

fig, ax = plt.subplots()
ax.scatter(reduced[:,0], reduced[:,1])

for i, txt in enumerate(style_map.values()):
    ax.annotate(txt, (reduced[i,0], reduced[i,1]))

st.pyplot(fig)