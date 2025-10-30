from engine import Value
from models import *
from optim import *
from losses import *

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

def grad_check(func, *inputs, eps=1e-6, tol=1e-4):
    vals = [inp.data for inp in inputs]
    for i in range(len(inputs)):
        vals_pos, vals_neg = vals.copy(), vals.copy()
        vals_pos[i] += eps
        vals_neg[i] -= eps
        f_pos = func(*vals_pos)
        f_neg = func(*vals_neg)
        grad_num = (f_pos - f_neg) / (2*eps)
        vals_autograd = [Value(v) for v in vals]
        func(*vals_autograd).backward()
        error = (abs(grad_num - vals_autograd[i].grad) / max(abs(grad_num), abs(vals_autograd[i].grad), 1))
        print(f"Variable {i+1}: grad_num={grad_num}, grad_auto={vals_autograd[i].grad}, error={error}")
        
        
a, b = Value(2), Value(8)
func = lambda a, b: (a * b) / (a + b)
# grad_check(func, a, b)

X, y = [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 0, 0, 1]
nn = MLP([2, 8, 4, 1], act_funcs=["relu", "sigmoid"], biases=[True, True])
optimizer = AdamW(nn.parameters(), learning_rate=1e-3, betas=(0.9, 0.999), weight_decay=0.)
# optimizer = SGD(nn.parameters(), learning_rate=1e-3)

max_epochs = 10_000
history = []
grads = {id(p):[] for p in nn.parameters()}
grad_norm_list = []

for epoch in range(max_epochs):
    loss = Value(0.)
    for x, y_true in zip(X, y):
        y_pred = nn(x)[0]
        loss += mse(y_pred, y_true)
    mean_loss = loss / len(y)
    mean_loss.backward()
    for p in nn.parameters():
        grads[id(p)].append(p.grad)
    grad_norm = math.sqrt(sum((p.grad ** 2) for p in nn.parameters()))
    grad_norm_list.append(grad_norm)
    history.append(mean_loss.data)
    print(f"Epoch {epoch+1}: loss={mean_loss.data}")
    optimizer.step()
    optimizer.zero_grad()
    

fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(3, 2, figure=fig)    

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(range(max_epochs), history, c="blue")

ax2 = fig.add_subplot(gs[1, :])
for p in nn.parameters():
    ax2.plot(range(max_epochs), grads[id(p)])

ax3 = fig.add_subplot(gs[2, :])
ax3.plot(range(max_epochs), grad_norm_list, "r", linewidth=1)
plt.show()