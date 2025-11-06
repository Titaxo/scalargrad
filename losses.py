from engine import Value
import math

def mse(y_pred, y_true):
    if isinstance(y_pred, (list, tuple)) and isinstance(y_true, (list, tuple)):
        loss = Value(0.)
        for yp, yt in zip(y_pred, y_true):
            loss += (yp - yt)**2
        return loss
    else:
        return (y_pred - y_true)**2
    
def cross_entropy(y_pred, y_true):
    eps = 1e-9
    loss = Value(0.)
    for yp, yt in zip(y_pred, y_true):
        if yt == 1:
            # asumimos que yp.data está en (0,1)
            loss += -1*((yp + eps).log())
    return loss
