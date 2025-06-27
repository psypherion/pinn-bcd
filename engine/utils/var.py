import numpy as np

def calculate_variance(patch: np.ndarray) -> float:
    """
    Calculates the scalar variance of a given 2D image patch.

    Args:
        patch (np.ndarray): A 2D NumPy array representing an image patch.

    Returns:
        float: The variance of the pixel intensities in the patch.
    """
    if patch.ndim != 2:
        raise ValueError("Input patch must be a 2D array.")
    return np.var(patch)