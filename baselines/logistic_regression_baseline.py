import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

class Model:
    """
    Simple baseline model that uses Logistic Regression.

    Bag-of-Words vectorization for Logistic Regression training.
    Tokenization is handled by CountVectorizer.
    """

    def __init__(self):
        self.vectorizer = CountVectorizer(
            max_features=300,
            ngram_range=(1, 1)
        )
        self.lr_model = LogisticRegression(
            max_iter=50,            
            solver="liblinear",
            C=0.01                  
        )

    def fit(self, X, y):
        # Split train (70%) and validation (15%) from the train/validation 85%
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.15 / 0.85,  # Get 15% of the original dataset
            stratify=y,
            random_state=42
        )

        X_train_bow = self.vectorizer.fit_transform(X_train[:, -1])
        X_val_bow = self.vectorizer.transform(X_val[:, -1])

        # Train Logistic Regression model
        self.lr_model.fit(X_train_bow, y_train)

    def predict(self, X):
        X_bow = self.vectorizer.transform(X[:, -1])
        return self.lr_model.predict(X_bow)
