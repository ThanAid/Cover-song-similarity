"""Utility functions for training."""
import pickle
from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

class DataGenerator(Sequence):
    """
    Generator for batches in TF

    Example of use:

    train_gen = DataGenerator(X_train, y_train, 32)
    test_gen = DataGenerator(X_test, y_test, 32)

    history = model.fit(train_gen,
                        epochs=6,
                        validation_data=test_gen)
    """

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        input_a = np.array([pair[0] for pair in batch_x])
        input_b = np.array([pair[1] for pair in batch_x])
        return (input_a, input_b), np.array(batch_y)

def load_in_chunks(path: str) -> list:
    """Load data stored in chunks."""
    data: list = []
    with open(path, 'rb') as f:
        while True:
            try:
                chunk = pickle.load(f)
                data.extend(chunk)
            except EOFError:
                break
    return data

def load_split_data(X_path: str, y_path: str, test_size: float, validation_size: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split data to create train, validation and test sets."""
    X = load_in_chunks(X_path)
    y = load_in_chunks(y_path)

    validation_size = validation_size / (1 - test_size)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, shuffle=True, stratify=y_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test