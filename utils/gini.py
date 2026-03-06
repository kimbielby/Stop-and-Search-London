import numpy as np

def gini(values):
    """
    Calculate the Gini coefficient for an array of values.

    Parameters
    ----------
    values : array-like
        Array of non-negative values (e.g. property prices).
        Zero and negative values are dropped before calculation.

    Returns
    -------
    float
        Gini coefficient between 0 (perfect equality) and 1 (perfect
        inequality). Returns NaN if fewer than 2 positive values remain
        after dropping zeros.

    Notes
    -----
    Uses the standard sorted-values formula:

        G = (2 * sum(i * x_i) / (n * sum(x_i))) - ((n + 1) / n)

    where x_i are the sorted values and i is the rank (1-indexed).
    This is equivalent to the area between the Lorenz curve and the
    line of perfect equality.
    """
    values = np.asarray(values, dtype=float)
    values = values[values > 0]  # drop zero or negative values
    n = len(values)
    if n < 2:
        return np.nan
    values = np.sort(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) / (n * np.sum(values))) - ((n + 1) / n)
