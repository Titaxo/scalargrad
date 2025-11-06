from losses import *
from engine import Value
from activations import softmax


a = Value(3)
b = Value(4)
c = Value(2)
outputs = softmax([a, b, c])
print(outputs)
loss = 5*outputs[0] + 3*outputs[1] + 2*outputs[2]
loss.backward()
print(outputs[0].grad, outputs[1].grad, outputs[2].grad)
print(a.grad, b.grad, c.grad)