import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def Wasserstein_p(p, u, v):
    assert len(u) == len(v)
    return np.mean(np.abs(np.sort(u) - np.sort(v)) ** p) ** (1 / p)


plt.plot(norm.ppf(np.linspace(0, 1, 1000)), label="invcdf")
plt.plot(np.sort(np.random.normal(size=1000)), label="sortsample")
plt.legend()
plt.show()
