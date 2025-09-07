import numpy as np
from .engine import Value
from .nn import Module
from .functional import softmax, sigmoid

EPS = 1e-12


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        
    def __call__(self, y_pred, y_true):
        probs = softmax(y_pred)
        probs = np.clip(probs, EPS, 1. - EPS)
        
        n_classes = len(y_pred.data[0])
        true_labels_oh = np.eye(n_classes)[y_true.data]
        
        loss_val = -np.sum(true_labels_oh * np.log(probs)) / probs.shape[0]
        
        out = Value(loss_val, _children=(y_pred,), _op='CELoss')
        
        def _backward():
            grad = (probs - true_labels_oh) / probs.shape[0]
            y_pred.grad += grad * out.grad
        
        out._backward = _backward
        return out
    
    
class BinaryCrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        
    def __call__(self, y_pred, y_true):
        probs = sigmoid(y_pred)
        probs = np.clip(probs, EPS, 1. - EPS)
        loss_val = -np.mean(y_true.data * np.log(probs) + (1 - y_true.data) * np.log(1 - probs))
        
        out = Value(loss_val, _children=(y_pred,), _op='BCELoss')
        
        def _backward():
            grad = (probs - y_true.data) / y_pred.data.size
            y_pred.grad += grad * out.grad
        
        out._backward = _backward
        return out


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true):
        loss_val = np.mean((y_pred.data - y_true.data) ** 2)
        out = Value(loss_val, _children=(y_pred,), _op='MSELoss')

        def _backward():
            grad = 2 * (y_pred.data - y_true.data) / y_true.data.size
            y_pred.grad += grad * out.grad

        out._backward = _backward
        return out
