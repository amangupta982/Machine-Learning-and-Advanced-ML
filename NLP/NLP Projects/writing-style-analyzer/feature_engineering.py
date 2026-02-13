import numpy as np

emotional_words = ["love", "happy", "sad", "angry", "excited", "amazing"]
technical_words = ["algorithm", "model", "data", "system", "analysis"]

def extra_features(texts):
    features = []

    for text in texts:
        words = text.split()

        length = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        emotion_score = sum(w in emotional_words for w in words)
        tech_score = sum(w in technical_words for w in words)

        features.append([
            length,
            avg_word_len,
            emotion_score,
            tech_score
        ])

    return np.array(features)