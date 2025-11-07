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
    X_1 = np.random.multivariate_normal(mean, cov, size=(num_points))
    y_1 = np.ones((len(X_1), 3)) * np.asarray([0, 1, 0])
    
    mean = [-3., -8.]
    cov = [[1., 0.], [0., 1.]]
    X_2 = np.random.multivariate_normal(mean, cov, size=(num_points))
    y_2 = np.ones((len(X_2), 3)) * np.asarray([0, 0, 1])
    
    X = np.concat([X_0, X_1, X_2])
    y = np.concat([y_0, y_1, y_2])
    
    print(f"X.shape={X.shape}, y.shape={y.shape}")
    show_viz(X, y)
    
    return X, y
    
def show_viz(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.8)
    plt.xlim((-12, 12))
    plt.ylim((-12, 12))
    plt.grid()
    plt.vlines(0, -15, 15, colors="k", linewidth=0.8)
    plt.hlines(0, -15, 15, colors="k", linewidth=0.8)
    plt.show()
    
    
X, y = make_dataset_gaussians()
X = X.tolist()
y = y.tolist()
nn = models.MLP([2, 8, 3], act_funcs=["relu", "softmax"], biases=[True, True])
print(len(nn.layers), len(nn.parameters()))
optimizer = optim.AdamW(nn.parameters(), learning_rate=5e-3, betas = (0.9, 0.999), weight_decay=0.)
# optimizer = optim.SGD(nn.parameters(), learning_rate=1e-3)

max_epochs = 800
history = []
grads = {id(p):[] for p in nn.parameters()}
grads_layer = {id(layer): {id(p): [] for p in layer.parameters()} for layer in nn.layers}
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
    for layer in nn.layers:
        for p in layer.parameters():
            grads_layer[id(layer)][id(p)].append(p.grad)
    grad_norm = math.sqrt(sum((p.grad ** 2) for p in nn.parameters()))
    grad_norm_list.append(grad_norm)
    history.append(mean_loss.data)
    print(f"Epoch {epoch+1}: loss={mean_loss.data}")
    optimizer.step()
    optimizer.zero_grad()
    

## PLOT PARA OBSERVAR COMO VARÍA EL LOSS A LO LARGO DE LAS EPOCAS, LOS GRADIENTES DE TODOS LOS PARAMETROS DEL MODELO Y LA NORMA DEL VECTOR GRADIENTE

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


## PLOT PARA OBSERVAR COMO VARÍAN LOS GRADIENTES POR CAPA A LO LARGO DE LAS EPOCAS

fig = plt.figure(figsize=(12,8))
gs = gridspec.GridSpec(len(nn.layers), 2, figure=fig)
for i, layer in enumerate(nn.layers):
    ax = fig.add_subplot(gs[i, :])
    for p in layer.parameters():
        ax.plot(range(max_epochs), grads_layer[id(layer)][id(p)])
plt.show()


## PLOT PARA OBSERVAR COMO CLASIFICA LA RED NEURONAL

preds = []
for x in X:
    yp = nn(x)
    yp_argmax = np.argmax(yp)
    preds.append(yp_argmax)
    
preds = np.asarray(preds)
yp_0 = preds[preds == 0]
yp_1 = preds[preds == 1]
yp_2 = preds[preds == 2]


X = np.asarray(X)
Xpred_0 = X[preds == 0]
Xpred_1 = X[preds == 1]
Xpred_2 = X[preds == 2]


plt.scatter(Xpred_0[:, 0], Xpred_0[:, 1], c="red", alpha=0.8)
plt.scatter(Xpred_1[:, 0], Xpred_1[:, 1], c="blue", alpha=0.8)
plt.scatter(Xpred_2[:, 0], Xpred_2[:, 1], c="purple", alpha=0.8)
plt.xlim((-12, 12))
plt.ylim((-12, 12))
plt.grid()
plt.vlines(0, -15, 15, colors="k", linewidth=0.8)
plt.hlines(0, -15, 15, colors="k", linewidth=0.8)
plt.show()