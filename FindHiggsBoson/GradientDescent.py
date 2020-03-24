import numpy as np
import datetime
from Plot import * 

def CalculateMSE(e):
    return 1/2*np.mean(e**2)

def ComputeGradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def GradientDescent(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    bar = InitProgressBar()
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad, err = ComputeGradient(y, tx, w)
        loss = CalculateMSE(err)
        w = w - gamma*grad
        ws.append(w)
        losses.append(loss)
        UpdateProgressBar(bar, n_iter/max_iters * 100)

    return losses, ws

def PredictLabels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = data.values.dot(weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


def RunGradientDescent(y, tx, initial_w, max_iters, gamma):
    print("Gradient Descent Algorithm Started...")
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = GradientDescent(y, tx, initial_w, max_iters, gamma)
    end_time = datetime.datetime.now()
    print("Gradient Descent Algorith Finished.")

    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time= {t:.3f} seconds".format(t=exection_time))    
    return gradient_ws, gradient_losses