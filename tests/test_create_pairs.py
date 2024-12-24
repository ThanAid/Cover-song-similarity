"""Sample test using pytest."""
import pytest
from src.preprocess.utils import create_pairs

def test_create_pairs():
    """Test Create pairs function."""
    # Use mock data
    test_dict = {
        'key1': {'file1': 'data1', 'file2': 'data2', 'file3': 'data3'},
        'key2': {'file4': 'data4', 'file5': 'data5'},
        'key3': {'file6': 'data6'}
    }
    dissimilar_percentage = 0.5 
    
    pairs, labels = create_pairs(test_dict, dissimilar_percentage)
    
    assert isinstance(pairs, list), "Pairs should be a list."
    assert isinstance(labels, list), "Labels should be a list."
    assert len(pairs) == len(labels), "Pairs and labels should have the same length."
    
    # Validate similar pairs
    similar_pairs = [(pair, label) for pair, label in zip(pairs, labels) if label == 0]
    for pair, label in similar_pairs:
        assert label == 0, "Similar pairs should have label 0."
        # Validate they come from the same key
        parent_keys = [key for key, files in test_dict.items() if pair[0] in files.values() and pair[1] in files.values()]
        assert len(parent_keys) == 1, "Both items in similar pair should belong to the same key."
    
    # Validate dissimilar pairs
    dissimilar_pairs = [(pair, label) for pair, label in zip(pairs, labels) if label == 1]
    for pair, label in dissimilar_pairs:
        assert label == 1, "Dissimilar pairs should have label 1."
        # Validate they come from different keys
        parent_keys = [key for key, files in test_dict.items() if pair[0] in files.values() or pair[1] in files.values()]
        assert len(parent_keys) == 2, "Each item in dissimilar pair should belong to a different key."
    
    expected_dissimilar_pairs = int(2*len(similar_pairs) * dissimilar_percentage)
    assert len(dissimilar_pairs) == expected_dissimilar_pairs, "Number of dissimilar pairs is incorrect."

