from sklearn.mixture import GaussianMixture
import pickle

def train_gmm(X):

    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',
        random_state=42
    )

    gmm.fit(X)

    with open("gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)

    return gmm