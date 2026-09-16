import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    # Write code here
    w2i = {word: i for i, word in enumerate(vocab)}
    vector = np.zeros(len(vocab), dtype=int)

    for token in tokens:
        if token in w2i:
            vector[w2i[token]] += 1

    return vector