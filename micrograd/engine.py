import numpy as np


class Value:
    def __init__(self, data, _children=(), _op='', dtype=np.float32):
        self.data = np.array(data, dtype=dtype)
        self.grad = np.zeros_like(self.data)
        
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __repr__(self):
        return f"Value(shape={self.shape}, data={self.data}, grad={self.grad})"

    def __str__(self):
        return f"Value(shape={self.shape}, grad={self.grad})"
    
    @staticmethod
    def _broadcast_backward(grad, target_shape):
        if grad.shape == target_shape:
            return grad
        
        # Sum over extra leading dims
        # e.g., grad.shape=(4, 3), target_shape=(3,) -> sum over axis 0.
        ndim_diff = grad.ndim - len(target_shape)
        assert ndim_diff >= 0, f"Can't backpropagate broadcast to shape {target_shape}"
        grad = grad.sum(axis=tuple(range(ndim_diff)))
        
        sum_axes = tuple([i for i in range(len(target_shape)) if target_shape[i] == 1])
        if sum_axes:
            grad = grad.sum(axis=sum_axes, keepdims=True)
        
        return grad
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += self._broadcast_backward(out.grad, self.data.shape)
            other.grad += self._broadcast_backward(out.grad, other.data.shape)
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += self._broadcast_backward(other.data * out.grad, self.data.shape)
            other.grad += self._broadcast_backward(self.data * out.grad, other.data.shape)
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'** {other}')
        
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1
    
    def __rtruediv__(self, other):
        return self ** -1 * other
    
    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'relu')
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        
        out._backward = _backward
        return out
    
    def flatten(self):
        orig_shape = self.data.shape
        
        if len(orig_shape) <= 1:
            return self

        flattened_shape = (orig_shape[0], np.prod(orig_shape[1:]))
        flattened = self.data.reshape(flattened_shape)
        
        out = Value(flattened, _children=(self,), _op='flatten')
        
        def _backward():
            self.grad += out.grad.reshape(orig_shape)
            
        out._backward = _backward
        return out
    
    @property
    def T(self):
        out = Value(self.data.T, _children=(self,), _op='T')
        
        def _backward():
            self.grad += out.grad.T
        
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()
