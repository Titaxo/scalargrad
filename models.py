import random

from engine import Value
from activations import *
from initializers import *

# TODO: Comprobaciones de dimensiones de input
class Perceptron():
    
    def __init__(self, nin, nout=1, act_func=None, bias=True):
        self.nin = nin
        self.w =  he_init(self.nin) if act_func == "relu" else xavier_init(self.nin, nout)
        self.b = Value(0.) if bias else None
        self.act_func=ACTIVATION_FUNCTIONS.get(act_func, None)
        
    def __call__(self, x):
        out = 0.
        for wi, xi in zip(self.w, x):
            out += wi * xi
        if self.b:
            out += self.b
        if self.act_func:
            out = self.act_func(out)
        return out
    
    def parameters(self):
        return self.w + [self.b] if self.b is not None else self.w
    

class LinearLayer():
    
    def __init__(self, nin, nout, act_func=None, bias=True):
        self.nin = nin
        self.nout = nout
        self.neurons = [Perceptron(self.nin, bias=bias) for _ in range(self.nout)]
        self.act_func=ACTIVATION_FUNCTIONS.get(act_func, None)
        
    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        if self.act_func == softmax:
            outputs = self.act_func(outputs)
        elif self.act_func is not None:
            outputs = [self.act_func(o) for o in outputs]
        return outputs
    
    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
    
class MLP():
    
    def __init__(self, neurons_layer: list[int], act_funcs: list[str], biases: list[bool]):
        self.layers = [LinearLayer(nin, nout, act_func=act_func, bias=bias) for nin, nout, act_func, bias in zip(neurons_layer[:-1], neurons_layer[1:], act_funcs, biases)]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params