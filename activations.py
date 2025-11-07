import math

from engine import Value



def sigmoid(x):
    res = Value()
    res.data = 1 / (1 + math.exp(-x.data))
    res._op = "sigmoid"
    res._prev = (x,)

    def backward():
        x.grad += res.grad * res.data * (1 - res.data)
    res._backward = backward

    return res

def relu(x):
    res = Value()
    res.data = max(0, x.data)
    res._op = "relu"
    res._prev = (x,)

    def backward():
        x.grad += res.grad if x.data > 0 else 0
    res._backward = backward
    
    return res

def tanh(x):
    res = Value()
    res.data = (math.exp(2*x.data) - 1) / (math.exp(2*x.data) + 1)
    res._op = "tanh"
    res._prev = (x,)

    def backward():
        x.grad += res.grad * (1 - res.data**2)
    res._backward = backward
    
    return res

def softmax(inputs):
    max_x = max(x.data for x in inputs)
    exps = [math.exp(x.data - max_x) for x in inputs]
    total = sum(exps)
    results = []
    
    for i, x in enumerate(inputs):
        y = Value(exps[i] / total)
        y._op = "softmax"
        y._prev = tuple(inputs)
        results.append(y)
    
    for i, y_i in enumerate(results):
        def _backward(i=i, y_i=y_i):
            for j, x_j in enumerate(inputs):
                grad_contrib = y_i.data * ((1 if i == j else 0) - results[j].data)
                x_j.grad += y_i.grad * grad_contrib
        y_i._backward = _backward
    return results
        
ACTIVATION_FUNCTIONS = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh, "softmax": softmax}