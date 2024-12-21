"""Utility functions for preprocessing audio files"""
from pathlib import Path
import h5py
import numpy as np
import random
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from loguru import logger

def load_hpcp(file_path: str|Path) -> np.ndarray:
    """Loads the HPCP features from the given file path"""
    with h5py.File(file_path, 'r') as f:
        return f['hpcp'][()]
    
def create_pairs(data: dict, dissimilar_percentage: float) -> Tuple[list, list]:
    """
    Creates pairs of similar and dissimilar data.
    
    Labels: 0 for similar, 1 for dissimilar.
    :param data: The input data dictionary
    :param dissimilar_percentage: The percentage of dissimilar pairs to create
    :return: A tuple containing pairs and their labels
    """
    pairs: list = []
    pair_labels: list = []

    # Create similar pairs
    for key in data:
        if len(data[key]) < 2: # skip if only one track
            continue
        to_fill = False # bool to fill tuples
        for filename in data[key]:
            if not to_fill:
                filename1 = filename
                to_fill = True
            else:
                pairs.append((data[key][filename1], data[key][filename]))
                pair_labels.append(0) # because they are under the same parent key
                to_fill = False

    # Create dissimilar pairs
    all_keys = list(data.keys())
    num_dissimilar_pairs = int(len(pairs) * 2 *  dissimilar_percentage)
    logger.info(f"Creating {num_dissimilar_pairs} dissimilar pairs")

    for i in range(num_dissimilar_pairs):
        key1, key2 = random.sample(all_keys, 2)
        item1 = random.choice(list(data[key1]))
        item2 = random.choice(list(data[key2]))
        pairs.append((data[key1][item1], data[key2][item2]))
        pair_labels.append(1)  # because they are under different parent keys

    return pairs, pair_labels

def normalize_data(data: np.ndarray) -> np.ndarray:
    """Normalize the input data"""
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def add_noise(y: np.array) -> np.array:
    """adds Gaussian noise to a audiofile"""

    mean = np.mean(y)
    var = np.var(y)
    noise = np.random.normal(mean, var, y.shape)
    return y + noise

def pitch_shift(hpcp: np.array, n_steps: int=1) -> np.array:
    """Pitch shift the HPCP by cyclically shifting the pitch classes"""

    shifted_hpcp = np.roll(hpcp, n_steps)    
    return shifted_hpcp

def zero_pad(sequence: np.array, max_length: int) -> np.ndarray:
    """Pads the HPCP data to max length"""
    _tr_sequence = sequence.T # Pad the first dimension
    padded_sequence = pad_sequences(_tr_sequence, maxlen=max_length, dtype='float32', padding='post', value=0.0)
    return padded_sequence.T
