import numpy as np
from .engine import Value


def softmax(x):
    max_logits = np.max(x.data, axis=-1, keepdims=True)
    exp_logits = np.exp(x.data - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs


def sigmoid(x):
    return 1 / (1 + np.exp(-x.data))
