import numpy as np
from .engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0)

    def parameters(self):
        return []
    
    def save_weights(self, path):
        params = {f"param_{i}": p.data for i, p in enumerate(self.parameters())}
        np.savez_compressed(path, **params)
        print(f"Model saved to {path}")
        
    def load_weights(self, path):
        try:
            weights = np.load(path)
        except FileNotFoundError:
            print(f"Error: No weights file found at {path}")
            return
        
        params = self.parameters()
        if len(params) != len(weights.files):
            print(f"Error: Architecture mismatch. Model has {len(params)} parameter arrays, "
                  f"but file has {len(weights.files)}.")
            return
        
        for i, p in enumerate(params):
            key = f"param_{i}"
            if key in weights:
                # Ensure the shape of the loaded weight matches the model's parameter shape
                if p.data.shape == weights[key].shape:
                    p.data = weights[key]
                else:
                    print(f"Error: Shape mismatch for parameter {i}. "
                          f"Model expects {p.data.shape}, but file has {weights[key].shape}.")
                    return
            else:
                print(f"Error: Parameter {key} not found in weights file.")
                return


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = Value(np.random.uniform(-1, 1, nin))
        self.b = Value(np.zeros(1))
        self.nonlin = nonlin
        
    def __call__(self, x):
        z = x @ self.w + self.b
        return z.relu() if self.nonlin else z
    
    def parameters(self):
        return [self.w, self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w.data)})"
    
    
class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
        # He initialization
        self.w = Value(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))
        self.b = Value(np.zeros(nout))
        self.nonlin = nonlin
        
    def __call__(self, x):
        z = x @ self.w + self.b
        return z.relu() if self.nonlin else z
    
    def parameters(self):
        return [self.w, self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Layer({self.w.data.shape[0]}, {self.w.data.shape[1]})"
    
    
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
