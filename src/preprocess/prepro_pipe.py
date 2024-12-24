"""
Preprocessing pipeline for the data.

This script contains the preprocessing pipeline for the data.
It includes functions to create pairs of similar and dissimilar data, normalize the data, and add noise to the data.

Pipeline:
1. Normalize the data
2. Add noise to the data
3. Create pairs of similar and dissimilar data
4. Save the pairs and labels to pickle file
"""
import pickle
from preprocess.utils import create_pairs, add_noise, normalize_data, zero_pad
from tqdm import tqdm
import os
from loguru import logger
import gc

class PreproPipeline:
    def __init__(self, data: dict, dissimilar_percentage: float, max_length: int, save_path: str = "") -> None:
        self.data = data
        self.dissimilar_percentage = dissimilar_percentage
        self.max_length = max_length
        self.save_path = save_path
        self.pairs = []
        self.pair_labels = []

    def preprocess(self) -> None:
        """Normalize the data."""
        for key in tqdm(self.data):
            for filename in self.data[key]:
                self.data[key][filename] = normalize_data(self.data[key][filename])
                self.data[key][filename] = add_noise(self.data[key][filename])
                self.data[key][filename] = zero_pad(self.data[key][filename], max_length=self.max_length)
        return None

    def create_pairs(self) -> None:
        """Create pairs of similar and dissimilar data."""
        logger.info("Creating pairs...")
        self.pairs, self.pair_labels = create_pairs(self.data, self.dissimilar_percentage)

        # clear up memory
        self.data = None
        gc.collect()
        return None
    
    def save_pairs(self) -> None:
        """Save the pairs and labels to pickle files in chunks."""
        logger.info("Saving pairs and labels to pickle files in chunks...")

        chunk_size = 2000 
        pairs_path = os.path.join(self.save_path, "pairs.pickle")
        labels_path = os.path.join(self.save_path, "pair_labels.pickle")

        def save_in_chunks(data, path):
            with open(path, "ab") as f:
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)

        save_in_chunks(self.pairs, pairs_path)
        save_in_chunks(self.pair_labels, labels_path)

        logger.info("Saving completed.")

    def get_pairs(self):
        return self.pairs

    def get_pair_labels(self):
        return self.pair_labels
    
    def run_pipeline(self):
        """Run the entire pipeline."""
        self.preprocess()
        self.create_pairs()
        self.save_pairs()
        return self