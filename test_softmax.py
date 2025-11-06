import random
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import models
import optim
from engine import Value
from losses import mse, cross_entropy

def make_dataset_gaussians():
    mean = [0., 0.]
    cov = [[1., 0.], [0., 1.]]
    num_points = 100
    X_0 = np.random.multivariate_normal(mean, cov, size=(num_points))
    y_0 = np.ones((len(X_0), 3)) * np.asarray([1, 0, 0])
    
    mean = [7., 7.]
    cov = [[1., 0.], [0., 1.]]
    num_points = 100
    X_1 = np.random.multivariate_normal(mean, cov, size=(num_points))
    y_1 = np.ones((len(X_1), 3)) * np.asarray([0, 1, 0])
    
    mean = [-3., -8.]
    cov = [[1., 0.], [0., 1.]]
    num_points = 100
    X_2 = np.random.multivariate_normal(mean, cov, size=(num_points))
    y_2 = np.ones((len(X_2), 3)) * np.asarray([0, 0, 1])
    
    X = np.concat([X_0, X_1, X_2])
    y = np.concat([y_0, y_1, y_2])
    
    print(f"X.shape={X.shape}, y.shape={y.shape}")
    # show_viz(X, y)
    
    return X.tolist(), y.tolist()
    
def show_viz(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.8)
    plt.xlim((-12, 12))
    plt.ylim((-12, 12))
    plt.grid()
    plt.vlines(0, -15, 15, colors="k", linewidth=0.8)
    plt.hlines(0, -15, 15, colors="k", linewidth=0.8)
    plt.show()
    
    
X, y = make_dataset_gaussians()
nn = models.MLP([2, 4, 6, 3], act_funcs=["relu", "relu", "softmax"], biases=[True, True, True])
print(len(nn.layers), len(nn.parameters()))
optimizer = optim.AdamW(nn.parameters(), learning_rate=1e-3, betas = (0.9, 0.999), weight_decay=0.)

max_epochs = 500
history = []
grads = {id(p):[] for p in nn.parameters()}
grad_norm_list = []

for epoch in range(max_epochs):
    loss = Value(0.)
    for x, y_true in zip(X, y):
        y_pred = nn(x)
        loss += cross_entropy(y_pred, y_true)
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
