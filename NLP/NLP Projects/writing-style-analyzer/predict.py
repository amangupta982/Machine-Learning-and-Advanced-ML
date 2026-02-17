import pickle

def load_model():
    with open("gmm_model.pkl", "rb") as f:
        return pickle.load(f)

style_map = {
    0: "Formal Writing",
    1: "Informal Writing",
    2: "Technical Writing",
    3: "Emotional Writing"
}

def predict_style(gmm, vector):
    cluster = gmm.predict(vector)[0]
    return style_map.get(cluster, "Unknown")