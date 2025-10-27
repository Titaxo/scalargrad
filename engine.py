"""
    Aquí se concentra el funcionamiento principal de la librería.
    La clase Value que se encarga de rastrear sus instancias e ir construyendo 
    un arbol computacional para después poder calcular las derivadas respecto de cada valor.
"""
import math
from collections import deque

import graphviz


class Value():

    def __init__(self, val=None, _prev=(), label=None):
        self.data = val
        self.label = label
        self._op = None
        self._prev = _prev
        self.grad = 0.
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value()
        res.data = self.data + other.data
        res._op = "+"
        res._prev = (self, other)

        def backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = backward

        return res

    def __radd__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value()
        res.data = other.data + self.data
        res._op = "+"
        res._prev = (other, self)

        def backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = backward

        return res

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value()
        res.data = self.data - other.data
        res._op = "-"
        res._prev = (self, other)

        def backward():
            self.grad += res.grad
            other.grad += -res.grad
        res._backward = backward

        return res

    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value()
        res.data = other.data - self.data
        res._op = "-"
        res._prev = (other, self)

        def backward():
            self.grad += -res.grad
            other.grad += res.grad
        res._backward = backward

        return res

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value()
        res.data = self.data * other.data
        res._op = "x"
        res._prev = (self, other)

        def backward():
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data
        res._backward = backward

        return res

    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        res = Value()
        res.data = other.data * self.data
        res._op = "x"
        res._prev = (other, self)

        def backward():
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data
        res._backward = backward

        return res

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError("This operation is only allowed for float or int exponents")
        res = Value()
        res.data = self.data ** other
        res._op = "**"
        res._prev = (self,)

        def backward():
            self.grad += res.grad * other * ((self.data) ** (other - 1))
        res._backward = backward

        return res

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        elif other.data == 0.:
            raise ZeroDivisionError("The division by zero is not allowed.")
        res = Value()
        res.data = self.data / other.data
        res._op = "/"
        res._prev = (self, other)

        def backward():
            self.grad += res.grad / other.data
            other.grad += -res.grad * self.data / (other.data ** 2)
        res._backward = backward

        return res

    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        elif self.data == 0.:
            raise ZeroDivisionError("The division by zero is not allowed.")
        res = Value()
        res.data = other.data / self.data
        res._op = "/"
        res._prev = (other, self)

        def backward():
            self.grad += -res.grad * other.data / (self.data ** 2)
            other.grad += res.grad / self.data
        res._backward = backward

        return res    

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
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

        self.grad = 1.
        for nodo in reversed(topo):
            nodo._backward()

    def draw_graph(self):
        dot = graphviz.Graph(comment="Árbol computacional")
        stack = deque([self])

        while stack:
            v = stack.popleft()
            dot.node(str(id(v)), f"{v.label}\ndata={v.data}\ngrad={v.grad}", shape="rectangle")
            if v._op:
                dot.node(str(id(v) + id(v._op)), v._op)
                dot.edge(str(id(v) + id(v._op)), str(id(v)))
            for child in v._prev:
                if v._op:
                    dot.edge(str(id(child)), str(id(v) + id(v._op)))
                else:
                    dot.edge(str(id(v) + id(v._op)), str(id(v)))

                stack.append(child)

        dot.save('grafo.dot')
