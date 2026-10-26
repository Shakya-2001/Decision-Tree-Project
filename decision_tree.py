import numpy as np


def entropy(y):
    y = np.asarray(y)

    if len(y) == 0:
        return 0.0

    # Count samples belonging to each class
    classes, counts = np.unique(y, return_counts=True)

    # Calculate probability of each class
    probabilities = counts / len(y)

    # Calculate entropy
    entropy_value = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return float(entropy_value)

def information_gain(parent, left_child, right_child):
    #Calculate information gain for a binary split.
    parent = np.asarray(parent)
    left_child = np.asarray(left_child)
    right_child = np.asarray(right_child)

    n = len(parent)
    n_left = len(left_child)
    n_right = len(right_child)

    # Reject invalid splits
    if n == 0 or n_left == 0 or n_right == 0:
        return 0.0

    if n_left + n_right != n:
        raise ValueError(
            "Child sample counts must equal parent sample count."
        )

    # Parent entropy
    parent_entropy = entropy(parent)

    # Weighted entropy of children
    weighted_entropy = (
        (n_left / n) * entropy(left_child)
        + (n_right / n) * entropy(right_child)
    )

    # Information gain
    gain = parent_entropy - weighted_entropy

    return float(gain)