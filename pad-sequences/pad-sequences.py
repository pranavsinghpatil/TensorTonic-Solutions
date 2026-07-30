import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:"""
    N = len(seqs)
    
    if max_len is None:
        L = max(len(seq) for seq in seqs) 
    else:
        L = max_len

    result = np.full((N, L), pad_value)
    for i, seq in enumerate(seqs):
        result[i, :len(seq)] = seq[:L]

    return result

