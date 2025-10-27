class SGD():
    
    def __init__(self, params, learning_rate=1e-3):
        self.params = params
        self.lr = learning_rate
        
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.