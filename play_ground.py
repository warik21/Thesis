import jax.numpy as jnp
import numpy as np

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from utils.Kantorivich_problem import *
from utils.utils import *
from utils.ot_utils import *

def im2mat(img):
    """Converts an image to matrix (one pixel per line)"""
    return img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

def mat2im(X, shape):
    """Converts back a matrix to an image"""
    return X.reshape(shape)

def minmax(img):
    return np.clip(img, 0, 1)

def solve_ott(x, y):
    geom = pointcloud.PointCloud(x, y, epsilon=1e-1)
    prob = linear_problem.LinearProblem(geom)

    solver = sinkhorn.Sinkhorn(threshold=1e-2, lse_mode=True, max_iterations=1000)
    out = solver(prob)

    f, g = out.f, out.g
    f, g = f - np.mean(f), g + np.mean(f)  # center variables, useful if one wants to compare them
    reg_ot = jnp.where(out.converged, jnp.sum(f) + jnp.sum(g), jnp.nan)
    return f, g, reg_ot

def solve_ot(a, b, x, y, ep, threshold):
    _, log = ot.sinkhorn(a, b, ot.dist(x, y), ep,
                         stopThr=threshold, method="sinkhorn_stabilized", log=True, numItermax=1000)

    f, g = ep * log["logu"], ep * log["logv"]
    f, g = f - np.mean(f), g + np.mean(f)  # center variables, useful if one wants to compare them
    reg_ot = (np.sum(f * a) + np.sum(g * b) if log["err"][-1] < threshold else np.nan)

    return f, g, reg_ot

n_p = 100
n_q = 100
n_max = 10000
eps = 1.e-2
dx = np.ones(n_p)/n_p
dy = np.ones(n_q)/n_q
eps_vec = np.logspace(-1.,-6.,10)
Fun = 'KL'
X,Y = np.linspace(0,1,n_p), np.linspace(0,1,n_q)

alpha =  make_1D_gauss(n_p, np.floor(3*n_p/4.), 1.)*1 + make_1D_gauss(n_p, np.floor(1*n_p/8.), 2.)*0.5
beta =  make_1D_gauss(n_q, np.floor(7*n_q/8.), 2.)*1

C = np.zeros([n_p,n_q],dtype=np.float64)
dist_f2 = lambda a,b : (a-b)**2
# Calculate the cost matrix, this is inefficient for large items.
for it1 in range(n_p):
    for it2 in range(n_q):
        C[it1,it2] = dist_f2(X[it1],Y[it2])

f, g = unbalanced_sinkhorn(alpha, beta, C, eps)

# Compute the OT matrix
pi = np.zeros([n_p,n_q],dtype=np.float64)
for i in range(n_p):
    for j in range(n_q):
        pi[i,j] = np.exp((f[i]+g[j]-C[i,j])/eps) * alpha[i] * beta[j]

Transport_plan = pi/np.sum(pi)

# Plots
# Plot target and source distributions
plt.figure( figsize=(10, 4))
plt.plot(X,alpha, 'b-', label='Source distribution')
plt.plot(Y,beta, 'r-', label='Target distribution')
plt.legend()
plt.title('Source and target distributions')
plt.show()

# Direct output transport plan
# Plot results
plt.figure( figsize=(10, 4))
plt.plot(X,alpha, 'b-.', label='Source distribution')
plt.plot(Y,beta, 'r-.', label='Target distribution')
plt.plot(X, Transport_plan.T @ dx, 'k-', label='Final dist: Transport_plan.T dx')
plt.plot(Y, Transport_plan @ dy, 'g-', label='Final dist: Transport_plan dy')
plt.legend()
plt.title('Source and target distributions')
plt.show()

# Plot transport plan with its marginals
plt.figure( figsize=(8, 8))
plot1D_mat(Transport_plan.T @ dx, Transport_plan @ dy, Transport_plan.T, 'Transport matrix with its marginals')
plt.show()

# Plot transport plan with its marginals
plt.figure( figsize=(8, 8))
plot1D_mat(alpha, beta, Transport_plan, 'Transport matrix with the target and source dist')
plt.show()