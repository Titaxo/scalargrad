from engine import Value
from models import *
from optim import *
from losses import *

# perceptron = Perceptron(2, act_func="sigmoid")
# optimizer = SGD(perceptron.parameters(), learning_rate=0.1)
# X = [(0, 0), (0, 1), (1, 0), (1, 1)]
# y = [0, 0, 0, 1]

# for i in range(len(X)):
#     y_pred = perceptron(X[i])
#     loss = mse(y_pred, y[i])
#     loss.backward()
#     print(perceptron.parameters())
#     optimizer.step()
#     optimizer.zero_grad()
#     print(perceptron.parameters())

nn = MLP([2, 8, 2], act_funcs=["relu", "relu"], biases=[False, True])
print(len(nn.parameters()))
print(nn([0, 1]))