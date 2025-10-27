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

ACTIVATION_FUNCTIONS = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh}