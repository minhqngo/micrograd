class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
            
    def zero_grad(self):
        for p in self.parameters:
            p.grad.fill(0)


class NesterovSGD:
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = learning_rate
        self.mu = momentum
        self.v = [0] * len(self.parameters)

    def step(self):
        for i, p in enumerate(self.parameters):
            v_prev = self.v[i]
            self.v[i] = self.mu * self.v[i] - self.lr * p.grad
            p.data += -self.mu * v_prev + (1 + self.mu) * self.v[i]

    def zero_grad(self):
        for p in self.parameters:
            p.grad.fill(0)
