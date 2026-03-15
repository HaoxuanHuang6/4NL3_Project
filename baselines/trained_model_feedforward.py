from sklearn.base import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build the FNN model
class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(64, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.output_layer(x))
        return x

class Model:
    """
    Baseline model that uses a feedforward neural network.

    TF-IDF vectorization for FNN training
    Tokenization is handled by TfidfVectorizer
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )

        self.fnn_model = None

    def fit(self, X, y):
        # Split train (70%) and validation (15%) from the train/validation 85%
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.15/0.85, # Get 15% of the original dataset
            stratify=y
        )

        X_train_tfidf = self.vectorizer.fit_transform(X_train).toarray()
        X_val_tfidf = self.vectorizer.transform(X_val).toarray()

        # Create the model
        input_dim = X_train_tfidf.shape[1]
        self.fnn_model = FNN(input_dim).to(device)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_tfidf).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
        X_val_tensor = torch.FloatTensor(X_val_tfidf).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).unsqueeze(1).to(device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.fnn_model.parameters())

        # Training loop with early stopping
        num_epochs = 50
        patience = 5
        best_val_loss = float('inf')
        patience_counter = 0

        # History dictionary to store metrics
        history = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': [],
            'precision': [], 'val_precision': [],
            'recall': [], 'val_recall': []
        }

        for epoch in range(num_epochs):
            # Training phase
            self.fnn_model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.fnn_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                train_preds.extend((outputs > 0.5).cpu().numpy().flatten())
                train_targets.extend(batch_y.cpu().numpy().flatten())

            train_loss /= len(train_loader.dataset)
            train_acc = accuracy_score(train_targets, train_preds)
            train_prec = precision_score(train_targets, train_preds, zero_division=0)
            train_rec = recall_score(train_targets, train_preds, zero_division=0)

            # Validation phase
            self.fnn_model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.fnn_model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_X.size(0)
                    val_preds.extend((outputs > 0.5).cpu().numpy().flatten())
                    val_targets.extend(batch_y.cpu().numpy().flatten())

            val_loss /= len(val_loader.dataset)
            val_acc = accuracy_score(val_targets, val_preds)
            val_prec = precision_score(val_targets, val_preds, zero_division=0)
            val_rec = recall_score(val_targets, val_preds, zero_division=0)

            # Store metrics
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['precision'].append(train_prec)
            history['val_precision'].append(val_prec)
            history['recall'].append(train_rec)
            history['val_recall'].append(val_rec)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_model_state = self.nn_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after epoch {epoch+1}")
                    # Restore best model
                    self.fnn_model.load_state_dict(best_model_state)
                    break
    
    def predict(self, X):
        # Get predictions from FNN model
        self.fnn_model.eval()

        X_tfidf = self.vectorizer.transform(X).toarray()

        with torch.no_grad():
            # Predictions
            X_tensor = torch.FloatTensor(X_tfidf).to(device)
            fnn_test_pred_proba = self.fnn_model(X_tensor).cpu().numpy()
            fnn_test_pred = (fnn_test_pred_proba > 0.5).astype(int).flatten()
        
        return fnn_test_pred