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

        X_train_bow = self.vectorizer.fit_transform(X_train)
        X_val_bow = self.vectorizer.transform(X_val)

        # Train Logistic Regression model
        self.lr_model.fit(X_train_bow, y_train)

        # Validation metrics
        val_preds = self.lr_model.predict(X_val_bow)
        val_acc = accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, zero_division=0)

        print(f"Validation - Acc: {val_acc:.4f} - F1: {val_f1:.4f}")

    def predict(self, X):
        X_bow = self.vectorizer.transform(X)
        return self.lr_model.predict(X_bow)
        
# Load data
train_data_path = 'PATH'
print("Loading data...")
train_df = pd.read_csv(train_data_path)

print(f"Train set: {len(train_df)} samples")

# Combine text features
print("\nCombining text features...")
train_df['combined_text'] = (
    train_df['sender'].fillna('') + ' ' + 
    train_df['subject'].fillna('') + ' ' + 
    train_df['body'].fillna('')
)

# Prepare data
X_train = train_df['combined_text']
y_train = train_df['label']

# Train model
print("\nTraining model...")
model = Model()
model.fit(X_train, y_train)