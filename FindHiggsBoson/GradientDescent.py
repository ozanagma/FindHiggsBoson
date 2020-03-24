import numpy as np
import datetime
from Plot import * 

def CalculateMSE(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def ComputeGradient(y, tx, w):
    """Compute the gradient."""
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
        # gradient w by descent update
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        UpdateProgressBar(bar, n_iter/max_iters * 100)
        # print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws

def PredictLabels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = data.values.dot(weights)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    
    return y_pred


def RunGradientDescent(y, tx, initial_w, max_iters, gamma):
    # Start gradient descent.
    print("Gradient Descent Algorithm Started...")
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = GradientDescent(y, tx, initial_w, max_iters, gamma)
    end_time = datetime.datetime.now()
    print("Gradient Descent Algorith Finished.")

    print("Results:")
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time= {t:.3f} seconds".format(t=exection_time))
    print("Gradient Descent: RMSE Loss = {t}".format(t=np.sqrt(2 * gradient_losses[-1])))
    
    return gradient_ws, gradient_losses