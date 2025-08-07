import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_distance(embedding1, embedding2, metric):
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    if metric == "cosine":
        return find_cosine_distance(embedding1, embedding2)
    elif metric == "angular":
        return find_angular_distance(embedding1, embedding2)
    elif metric == "euclidean":
        return find_euclidean_distance(embedding1, embedding2)
    elif metric == "euclidean_l2":
        norm_axis = None if embedding1.ndim == 1 else 1
        embedding1 = l2_normalize(embedding1, axis=norm_axis)
        embedding2 = l2_normalize(embedding2, axis=norm_axis)
        return find_euclidean_distance(embedding1, embedding2)
    else:
        raise ValueError(f"Invalid distance metric: {metric}")

def get_verification_threshold(metric: str) -> float:
    default_thresholds = {"cosine": 0.40, "euclidean": 4.15, "euclidean_l2": 0.95, "angular": 0.37}
    
    return default_thresholds.get(metric, 0.4)


def l2_normalize(x, axis=None):
    norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / np.clip(norm, a_min=1e-10, a_max=None)

def find_cosine_distance(a, b):
    return 1 - cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]

def find_angular_distance(a, b):
    cosine_sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
    return np.arccos(np.clip(cosine_sim, -1.0, 1.0)) / np.pi

def find_euclidean_distance(a, b):
    return np.linalg.norm(a - b)