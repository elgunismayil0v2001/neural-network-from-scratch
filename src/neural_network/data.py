import numpy as np


def create_batch(X : np.ndarray,
                y : np.ndarray,
                batch_size : int,
                shuffle : bool = True,
                seed : int | None = None):
    """
    Create batches of data from the input features and labels.

    Parameters:
    X (numpy.ndarray): Input features.
    y (numpy.ndarray): Corresponding labels.
    batch_size (int): Size of each batch.
    shuffle (bool): Whether to shuffle the data before creating batches.

    Returns:
    Generator yielding tuples of (batch_X, batch_y) for each batch.
    """
   
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of samples in X and y must be the same.")  
    
    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    
    if shuffle:
        rng.shuffle(indices)
        
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], y[batch_indices]
        
    