import time
import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np
from generators2d import sample_data
from utility import to_numpy, to_torch


class ICNN(torch.nn.Module):
    """Input Convex Neural Network"""

    def __init__(self, dim=2, dimh=64, num_hidden_layers=4):
        """
        The __init__ function is called when the object is instantiated.
        It sets up the attributes of an object.
        In this case, we are setting up a input convec neural network with two hidden layers and one output layer.

        :param self: Represent the instance of the class
        :param dim: Specify the dimension of the input data
        :param dimh: Set the number of hidden units in each layer
        :param num_hidden_layers: Determine the number of hidden layers in the network
        :return: Nothing
        """
        super().__init__()

        Wzs = []
        Wzs.append(nn.Linear(dim, dimh))
        for _ in range(num_hidden_layers - 1):
            Wzs.append(torch.nn.Linear(dimh, dimh, bias=False))
        Wzs.append(torch.nn.Linear(dimh, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(num_hidden_layers - 1):
            Wxs.append(nn.Linear(dim, dimh))
        Wxs.append(nn.Linear(dim, 1, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = nn.Softplus()

    def forward(self, x):
        """
        The forward function takes in a tensor x and returns the output of the
        network. The input is passed through each layer, with an activation function
        applied to each hidden layer. The final output is returned.

        :param self: Represent the instance of the class
        :param x: Pass the input data to the network
        :return: The output of the last layer, which is a linear function
        """
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)


def test_convexity(f):
    """
    The test_convexity function takes a function f as input and returns True if the function is convex.

    :param f: Test the convexity of a function
    :return: A boolean value, true if the function is convex and false otherwise
    """
    rdata = torch.randn(1024, 2).to(device)
    rdata2 = torch.randn(1024, 2).to(device)
    return np.all(
        (((f(rdata) + f(rdata2)) / 2 - f(rdata + rdata2) / 2) > 0)
        .cpu()
        .detach()
        .numpy()
    )


def compute_w2(f, g, x, y, return_loss=False):
    """
    The compute_w2 function computes the Wasserstein-2 distance between two distributions.

    :param f: Calculate the loss of f and g, while the g parameter is used to calculate the gradient of y
    :param g: Compute the gradient of g(y)
    :param x: Pass in the input data to f
    :param y: Compute the gradient of g(y)
    :param return_loss: Return the losses of f and g
    :return: The w2 distance between f and g
    """
    fx = f(x)
    gy = g(y)

    grad_gy = autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[0]

    f_grad_gy = f(grad_gy)
    y_dot_grad_gy = torch.sum(torch.multiply(y, grad_gy), axis=1, keepdim=True)

    x_squared = torch.sum(torch.pow(x, 2), axis=1, keepdim=True)
    y_squared = torch.sum(torch.pow(y, 2), axis=1, keepdim=True)

    w2 = torch.mean(f_grad_gy - fx - y_dot_grad_gy + 0.5 * x_squared + 0.5 * y_squared)
    if not return_loss:
        return w2
    g_loss = torch.mean(f_grad_gy - y_dot_grad_gy)
    f_loss = torch.mean(fx - f_grad_gy)
    return w2, f_loss, g_loss


def plot(x, y, x_pred, y_pred, savename=None):
    """
    The plot function takes in the following arguments:
        x, y: The original data points.
        x_pred, y_pred: The predicted data points.

    :param x: Plot the x points, y is used to plot y points, x_pred is used to plot the gradient of g(y)
              and y_pred is used to plot the gradient of f(x)
    :param y: Plot the data points of y
    :param x_pred: Plot the gradient of g(y)
    :param y_pred: Plot the gradient of f(x)
    :param savename: Save the plot as a file
    :return: A matplotlib object
    """
    x = to_numpy(x)
    y = to_numpy(y)
    x_pred = to_numpy(x_pred)
    y_pred = to_numpy(y_pred)

    import matplotlib.pyplot as plt

    plt.scatter(y[:, 0], y[:, 1], color="C1", alpha=0.5, label=r"$Y$")
    plt.scatter(x[:, 0], x[:, 1], color="C2", alpha=0.5, label=r"$X$")
    plt.scatter(
        x_pred[:, 0], x_pred[:, 1], color="C3", alpha=0.5, label=r"$\nabla g(Y)$"
    )
    plt.scatter(
        y_pred[:, 0], y_pred[:, 1], color="C4", alpha=0.5, label=r"$\nabla f(X)$"
    )
    plt.legend()
    if savename:
        plt.savefig(savename)
    plt.close()


def transport(model, x):
    """
    The transport function takes a model and an input, and returns the gradient of the output with respect to the input.
    This is useful for computing gradients of functions that are not differentiable in their inputs.

    :param model: Calculate the gradient of the model with respect to x
    :param x: Pass in the input data
    :return: The gradient of the model at x
    """
    return autograd.grad(torch.sum(model(x)), x)[0]


def train(f, g, x_sampler, y_sampler, batchsize=1024, reg=0):
    """
    The train function trains the transport map f and g.

    :param f: Define the function f(x)
    :param g: Define the generator
    :param x_sampler: Sample from the target distribution
    :param y_sampler: Sample from the target distribution
    :param batchsize: Specify the number of samples to be used in each iteration
    :param reg: Control the regularization of the transport maps
    :return: The transport maps f and g
    """
    def y_to_x(y):
        """
        The y_to_x function is a function that takes in an input y and returns the output x.
        The output x is calculated by taking the gradient of g at y, which gives us a vector field.
        We then use this vector field to transport our initial condition (the point 0) to our desired location.

        :param y: Determine the value of x
        :return: The transport of g along y
        """
        return transport(g, y)

    def x_to_y(x):
        """
        The x_to_y function is a wrapper for the transport function.
        It takes an x value and returns the corresponding y value,
        using f as its transport function.

        :param x: Pass the value of x to the function
        :return: The transport of f along x
        """
        return transport(f, x)

    optimizer_f = torch.optim.Adam(f.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_g = torch.optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.9))
    nepochs = 10000
    print_interval = 100
    nval = 1024

    x_val = to_torch(next(x_sampler(nval)))
    y_val = to_torch(next(y_sampler(nval)))

    start = time.time()

    for epoch in range(1, nepochs + 1):
        for _ in range(10):
            optimizer_g.zero_grad()
            x = to_torch(next(x_sampler(batchsize)))
            y = to_torch(next(y_sampler(batchsize)))
            fx = f(x)
            gy = g(y)
            grad_gy = autograd.grad(
                torch.sum(gy), y, retain_graph=True, create_graph=True
            )[0]
            f_grad_gy = f(grad_gy)
            y_dot_grad_gy = torch.sum(torch.mul(y, grad_gy), axis=1, keepdim=True)
            g_loss = torch.mean(f_grad_gy - y_dot_grad_gy)
            if reg > 0:
                g_loss += reg * torch.sum(
                    torch.stack([torch.sum(F.relu(-w.weight) ** 2) / 2 for w in g.Wzs])
                )
            g_loss.backward()
            optimizer_g.step()

        optimizer_f.zero_grad()
        x = to_torch(next(x_sampler(batchsize)))
        y = to_torch(next(y_sampler(batchsize)))
        fx = f(x)
        gy = g(y)
        grad_gy = autograd.grad(torch.sum(gy), y, retain_graph=True, create_graph=True)[
            0
        ]
        f_grad_gy = f(grad_gy)
        f_loss = torch.mean(fx - f_grad_gy)
        if reg > 0:
            f_loss += reg * torch.sum(
                torch.stack([torch.sum(F.relu(-w.weight) ** 2) / 2 for w in f.Wzs])
            )

        f_loss.backward()
        optimizer_f.step()

        if epoch % print_interval == 0:
            w2, f_loss, g_loss = compute_w2(f, g, y_val, y_val, return_loss=True)
            end = time.time()
            print(
                f"Iter={epoch}, f_loss={f_loss:0.2f}, g_loss={g_loss:0.2f}, W2={w2:0.2f}, time={end - start:0.1f}, "
                f"f_convex={test_convexity(f)}, g_convex={test_convexity(g)}"
            )
            start = end
            if plot_sample:
                x_pred = y_to_x(y_val)
                y_pred = x_to_y(x_val)
                plot(x_val, y_val, x_pred, y_pred, savename=f"tmp/epoch_{epoch}")


def main():
    """
    The main function is the entry point of the program.
    It initializes a scale and variance parameter, which are used to generate data from two different distributions.
    The names of these distributions are also initialized here, as well as lambda functions that sample from them.
    These samplers will be passed into our training function later on in this file.

    :return: The value of the last expression evaluated
    """
    scale = 5
    var = 1

    x_name = "8gaussians"
    y_name = "simpleGaussian"

    x_sampler = lambda n: sample_data(x_name, n, scale, var)
    y_sampler = lambda n: sample_data(y_name, n, scale, var)

    f = ICNN().to(device)
    g = ICNN().to(device)
    train(f, g, x_sampler, y_sampler, reg=0.1)


if __name__ == "__main__":
    plot_sample = True
    device = "cuda"
    main()
