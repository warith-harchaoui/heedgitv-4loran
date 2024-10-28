"""
This elbow.py module provides a set of utility functions to fit the elbow function to data.
Thanks to gradient descent, the elbow function is fitted to the data in order to determine the elbow point.

y = t_v + s_v * np.exp(- (x - t_h) / s_h) + noise

For a given x and y entries, it finds the parameters t_v, t_h, s_v, s_h that minimize the mean squared error.

elbow = t_h + 3.0 * s_h

Dependencies
- numpy
- torch

Authors:
- Warith Harchaoui, https://harchaoui.org/warith

"""


import numpy as np
import torch
from torch.autograd import Variable
import os_helper

import matplotlib.pyplot as plt

# Define how numbers are represented
torch_real_type = torch.float  # decimal
torch_integer_type = torch.long  # integer


# Define the function to fit the data
def elbow_function(x, t_v=0.0, t_h=0.0, s_v=1.0, s_h=1.0):
    s = 0
    if isinstance(s_h, float):
        s = abs(s_h)
        t_h = abs(t_h)
        t_v = abs(t_v)
        s_v = abs(s_v)
    else:
        s = torch.abs(s_h)
        t_h = torch.abs(t_h)
        t_v = torch.abs(t_v)
        s_v = torch.abs(s_v)

    tt = 0
    numpy_types = [list, np.ndarray, tuple, np.float32, np.float64]
    if any([isinstance(x, t) for t in numpy_types]):
        tt = np.exp(-(x - t_h) / (1e-4 + s))
    else:
        tt = torch.exp(-(x - t_h) / (1e-4 + s))

    y = t_v + s_v * tt
    return y


def continuous_elbow(x, res):
    return elbow_function(x, t_v=res["T_v"], t_h=res["T_h"], s_v=res["S_v"], s_h=res["S_h"])


def elbow_visu(x, y, res, output_image = None):
    from matplotlib import pyplot as plt
    z = [continuous_elbow(xx, res) for xx in x]
    # Plot the synthetic data and the fitted curve
    b = np.median(np.diff(x))
    plt.gcf().clear()
    plt.figure(figsize=(8, 4))
    plt.bar(x, y, width=0.9*b, label="Observed Occurences", color="#007AFF")
    plt.plot(x, z, label="Fitted Curve", color="#FF6961")
    plt.axvline(res["elbow"], color="#02D46A", label=f"Elbow: {res['elbow']:.2f} seconds")
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.tick_params(axis='y', which='both', left=False, right=False)
    plt.title('Elbow Analysis')
    plt.xlabel("Inter-Word Silence Duration (s)")
    plt.ylabel("Occurences")
    M = min(4.0, 10 * res["elbow"])
    M = max(M, 3* res["elbow"])
    dx = 0.01 * M
    plt.xlim([0, M + dx])
    dy = 0.01 * (np.max(y) - np.min(y))
    plt.ylim([0, np.max(y) + dy])
    if not (output_image is None):
        plt.savefig(output_image)
    


def numpy2pytorch(x, differentiable=True, dtype=torch_real_type, device=None):
    if isinstance(x, float):
        t = np.array([x]).astype(np.float32)
        t = torch.from_numpy(t)
        res = Variable(t, differentiable).type(dtype)
    else:
        t = np.array(x).astype(np.float32)
        res = Variable(torch.from_numpy(t), differentiable).type(dtype)
    if not (device is None):
        res = res.to(device)
    if differentiable:
        res = torch.nn.Parameter(res)
    return res


def pytorch2numpy(x):
    t = x.clone()
    t = t.cpu().detach().numpy()
    s = len(np.array(t).shape)
    if s == 0:
        return t
    if s == 1 and np.array(t).shape[0] == 1:
        return t.ravel()[0]
    if s > 1 and np.prod(np.array(t).shape) == 1:
        return t.ravel()
    return t


# Generate synthetic data in the form: y = t_v + s_v * np.exp(- (x - t_h) / s_h) + noise
def synthetic_data(
    x_min=0,
    x_max=6,
    nb_points=7,
    t_v=0.0,
    t_h=0.0,
    s_v=1.0,
    s_h=1.0,
    noise=0.1,
):
    x = np.linspace(x_min, x_max, nb_points)
    y = t_v + s_v * np.exp(-(x - t_h) / s_h)
    if noise > 0:
        y += noise * np.random.randn(nb_points)
    return np.array(list(zip(x, y)))



# Normalization of data (min 0 and max 1) for the default learning rate to be invariant
def normalization(x_y):
    t = np.array(x_y)
    x = t[:, 0].reshape([-1])
    y = t[:, 1].reshape([-1])
    mu = np.array([0, np.min(y)])
    sigma = np.array([np.max(x), np.max(y)])
    sigma = np.array([max(s, 1e-4) for s in sigma])
    t -= mu.reshape([1, -1])
    t /= sigma.reshape([1, -1])
    return t, mu, sigma

def abascus(x_y, device="cpu"):
    """
    Abascus algorithm to fit the elbow function to the data
    Function fit is y = t_v + s_v * exp(-(x - t_h) / s_h)

    Parameters
    ----------
    x_y : numpy array
        The data to fit: x_y[:, 0] is the x-axis and x_y[:, 1] is the y-axis
    device : str, optional
        The device to use for the computation, by default "cpu"

    Returns
    -------
    dict
        The fitted parameters: S_h, S_v, T_h, T_v, elbow, threshold_10, threshold_5, threshold_3, loss, iteration
    """
    # Normalization
    normalized_x_y, mu, sigma = normalization(x_y)
    normalized_x = normalized_x_y[:, 0].reshape([-1])
    normalized_y = normalized_x_y[:, 1].reshape([-1])

    # Fixed training data for the *non* stochastic gradient descent
    x = numpy2pytorch(normalized_x, differentiable=False, device=device)
    y = numpy2pytorch(normalized_y, differentiable=False, device=device)

    # Initial values for abascus parameters
    # Rules of thumb for initial values
    init_t_v = np.min(normalized_y)
    init_t_h = np.min(normalized_x)
    init_s_v = np.max(normalized_y) - np.min(normalized_y)
    init_s_h = 0.2  * (np.max(normalized_x) - np.min(normalized_x))

    # Abascus parameters
    s_h = numpy2pytorch(init_s_h, differentiable=True, device=device)
    s_v = numpy2pytorch(init_s_v, differentiable=True, device=device)
    t_h = numpy2pytorch(init_t_h, differentiable=True, device=device)
    t_v = numpy2pytorch(init_t_v, differentiable=True, device=device)

    # Pytorch parameters, optimizer and loss
    params = [s_h, s_v, t_h, t_v]
    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.MSELoss()

    # We stop when the loss is less than 1e-5 times the square amplitude of the data
    delta_y_square = (np.max(normalized_y) - np.min(normalized_y)) ** 2.0

    # Training loop
    iteration = 0
    error = np.inf
    losses = []
    while np.sqrt(error) > 1e-5 * np.sqrt(delta_y_square):
        # Free gradients accumulated in the previous iteration
        optimizer.zero_grad()

        # Compute the loss with respect to the current abascus parameters
        y_hat = elbow_function(x, t_v=t_v, t_h=t_h, s_v=s_v, s_h=s_h)
        loss = criterion(y, y_hat)

        # Compute the gradients
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Compute the loss and check if we have converged
        ell = float(pytorch2numpy(loss))
        losses.append(ell)
        error = losses[-1]

        iteration += 1

        if iteration % 1000 == 0:
            variation = np.max(
                np.abs(np.diff(losses[-20:])) / (np.abs(losses[-19:]) + 1e-6)
            )
            os_helper.info(
                f"Iteration: {iteration}, Error: {losses[-1]}, Variation: {variation}"
            )

            if variation < 1e-4:
                break

        if iteration > 200000:
            break

    s_h = pytorch2numpy(s_h)
    s_v = pytorch2numpy(s_v)
    t_h = pytorch2numpy(t_h)
    t_v = pytorch2numpy(t_v)

    mu_h = mu[0]
    mu_v = mu[1]
    sigma_h = sigma[0]
    sigma_v = sigma[1]

    T_v = mu_v + t_v * sigma_v
    S_v = s_v * sigma_v
    T_h = mu_h + t_h * sigma_h
    S_h = s_h * sigma_h
    elbow = abs(T_h) + 3.0 * abs(S_h)
    threshold_10 = abs(T_h) + 10.0 * abs(S_h)
    threshold_5 = abs(T_h) + 5.0 * abs(S_h)
    threshold_3 = abs(T_h) + 3.0 * abs(S_h)

    res = {
        "S_h": abs(S_h),
        "S_v": abs(S_v),
        "T_h": abs(T_h),
        "T_v": abs(T_v),
        "elbow": elbow,
        "threshold_10": threshold_10,
        "threshold_5": threshold_5,
        "threshold_3": threshold_3,
        "loss": losses[-5:],
        "iteration": iteration,
    }

    return res



if __name__ == "__main__":
        
    t_v = 1.0
    t_h = 1.0
    s_v = 2.0
    s_h = 12.5

    noise = 0.05

    x_min = 0
    x_max = 30 * t_h
    nb_points = int(x_max - x_min + 1) * 10

    # y = t_v + s_v * np.exp(- (x - t_h) / s_h) + noise

    x_y = synthetic_data(
        t_v=t_v,
        t_h=t_h,
        s_v=s_v,
        s_h=s_h,
        x_min=x_min,
        x_max=x_max,
        nb_points=nb_points,
        noise=noise,
    )
    gt = {
        "t_v": t_v,
        "t_h": t_h,
        "s_v": s_v,
        "s_h": s_h,
    }
    res = abascus(x_y, device="cpu")

    x_min = np.min(x_y[:, 0])
    x_max = np.max(x_y[:, 0])
    x = np.linspace(x_min, x_max, 100)
    x_th = numpy2pytorch(x, differentiable=False, device="cpu")
    y_th = elbow_function(
        x_th, t_v=res["T_v"], t_h=res["T_h"], s_v=res["S_v"], s_h=res["S_h"]
    )

    y_pred = pytorch2numpy(y_th)

    pred = {
        "t_v": res["T_v"],
        "t_h": res["T_h"],
        "s_v": res["S_v"],
        "s_h": res["S_h"],
    }
    # Print ground truth and predicted parameters
    print("\nParameter Comparison:")
    for k in gt:
        print(f"gt[{k}]: {gt[k]:.2f}, pred[{k}]: {pred[k]:.2f}")


