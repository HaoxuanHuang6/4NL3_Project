import numpy as np


class Model:
    """
    Baseline model that randomly assigns labels based on training data distribution.
    If 60% of training data is label 0, this model predicts 0 with 60% probability.
    """
    def __init__(self):
        self.phishing_rate = None

    def fit(self, X, y):
        self.phishing_rate = y.mean()
        print(f"The data phishing rate is: {self.phishing_rate:.2%}")

    def predict(self, X):
        # Use random.choice with probabilities based on training distribution
        predictions = np.random.choice(
            [0, 1],
            size=X.shape[0],
            p=[1 - self.phishing_rate, self.phishing_rate]
        )
        return predictions
