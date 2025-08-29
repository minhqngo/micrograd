import numpy as np
from .engine import Value
from .nn import Module
from .functional import softmax, sigmoid

EPS = 1e-12


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        
    def __call__(self, logits, true_labels):
        probs = softmax(logits)
        probs = np.clip(probs, EPS, 1. - EPS)
        
        n_classes = len(logits.data[0])
        true_labels_oh = np.eye(n_classes)[true_labels]
        
        loss_val = -np.sum(true_labels_oh.data * np.log(probs)) / probs.shape[0]
        
        out = Value(loss_val, _children=(logits,), _op='CELoss')
        
        def _backward():
            grad = (probs - true_labels_oh) / probs.shape[0]
            logits.grad += grad * out.grad
        
        out._backward = _backward
        return out
    
    
class BinaryCrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        
    def __call__(self, logits, true_labels):
        probs = sigmoid(logits)
        probs = np.clip(probs, EPS, 1. - EPS)
        loss_val = -np.mean(true_labels * np.log(probs) + (1 - true_labels) * np.log(1 - probs))
        
        out = Value(loss_val, _children=(logits,), _op='BCELoss')
        
        def _backward():
            grad = (probs - true_labels.data) / logits.data.size
            logits.grad += grad * out.grad
        
        out._backward = _backward
        return out
