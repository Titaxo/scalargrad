class SGD():
    
    def __init__(self, params, learning_rate=1e-3):
        self.step_count = 0
        self.params = params
        self.lr = learning_rate
        
    def step(self):
        self.step_count += 1
        for p in self.params:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.
            
            
class AdamW():
    
    def __init__(self, params, learning_rate=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.step_count = 0
        self.params = params
        self.lr = learning_rate
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.wd = weight_decay
        self.m = {id(p): 0. for p in self.params}
        self.v = {id(p): 0. for p in self.params}
        
    def step(self):
        self.step_count += 1
        for p in self.params:
            id_p = id(p)
            p.data -= self.lr * self.wd * p.data
            
            self.m[id_p] = self.beta1 * self.m[id_p] + (1 - self.beta1) * p.grad
            self.v[id_p] = self.beta2 * self.v[id_p] + (1 - self.beta2) * (p.grad ** 2)

            m_hat = self.m[id_p] / (1 - (self.beta1 ** self.step_count))
            v_hat = self.v[id_p] / (1 - (self.beta2 ** self.step_count))
            p.data -= self.lr * m_hat / (v_hat ** (0.5) + self.eps)
            
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.