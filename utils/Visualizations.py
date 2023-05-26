import matplotlib.pyplot as plt
import numpy as np
from utils.utils import *

def plot_distribution(X, p, q, title):
    """
    Plot the source and target distributions
    :param X: linear space in which the distributions are defined
    :param p: source distribution
    :param q: target distribution
    :param title: title of the plot
    :return: None, plot the distributions
    """
    plt.figure(figsize=(10, 4))
    plt.plot(X, p, 'b-', label='Source dist: p')
    plt.plot(X, q, 'r-', label='Target dist: q')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_marginals(X, p, q, Transport_plan, title):
    """
    Plot the marginals of the transport map
    :param X: linear space in which the distributions are defined
    :param p: source distribution
    :param q: target distribution
    :param Transport_plan: transport plan
    :param title: title of the plot
    :return: None, plot the distributions
    """
    plt.figure(figsize=(10, 4))
    n_p = len(p)
    n_q = len(q)
    dx = np.ones(n_p) / n_p
    dy = np.ones(n_q) / n_q
    plt.plot(X, p, 'b-.', label='Source dist: p')
    plt.plot(X, q, 'r-.', label='Target dist: q')
    plt.plot(X, Transport_plan.T @ dx, 'k-', label='Final source dist (q): Transport_plan.T dx')
    plt.plot(X, Transport_plan @ dy, 'g-', label='Final target dist (p): Transport_plan dy')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_signed_marginals(X, p, q, Transport_plan, title):
    """
    Plot the marginals of the transport map
    :param X: linear space in which the distributions are defined
    :param p: source distribution
    :param q: target distribution
    :param Transport_plan: transport plan
    :param title: title of the plot
    :return: None, plot the distributions
    """
    plt.figure(figsize=(10, 4))
    plt.plot(X, p, 'b-.', label='Source dist: p (signed)')
    plt.plot(X, q, 'r-.', label='Target dist: q (signed)')
    n_p = len(p)
    n_q = len(q)
    dx = np.ones(n_p) / n_p
    dy = np.ones(n_q) / n_q
    plt.plot(X, (Transport_plan.T @ dx) * np.sign(q), 'k-', label='Final source dist (q): Transport_plan.T dx (signed)')
    plt.plot(X, (Transport_plan @ dy) * np.sign(p), 'g-', label='Final target dist (p): Transport_plan dy (signed)')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_transport_map(p, q, Transport_plan, title):
    """
    Plot the transport map
    :param p: source distribution
    :param q: target distribution
    :param X: linear space in which the source distribution is defined
    :param Y: linear space in which the target distribution is defined
    :param Transport_plan: transport plan
    :param title: title of the plot
    :return: None, plot the transport map
    """
    plt.figure(figsize=(8, 8))
    plot1D_mat(p, q, Transport_plan, title)
    plt.show()

def plot_transport_map_with_marginals(p, q, Transport_plan, title):
    """
    Plot the transport map
    :param q: target distribution
    :param p: source distribution
    :param Transport_plan: transport plan
    :param title: title of the plot
    :return: None, plot the transport map
    """
    plt.figure(figsize=(8, 8))
    plot1D_mat(p, q, Transport_plan, title)
    plt.show()

