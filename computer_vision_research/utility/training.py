import numpy as np


def iter_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int):
    """Generate minibatches for training.

    Args:
        X (np.ndarray): Feature vectors shape(n_samples, n_features) 
        y (np.ndarray): Labels of vectors shape(n_samples,) 
    """
    current_batch_size = batch_size
    X_batch, y_batch = X[:current_batch_size], y[:current_batch_size]
    while X.shape[0] - current_batch_size >= batch_size:
        yield X_batch, y_batch
        X_batch, y_batch = X[:current_batch_size], y[:current_batch_size]
        current_batch_size += batch_size
    else:
        last_batch_size = X.shape[0] % batch_size
        yield X[:last_batch_size], y[:last_batch_size]