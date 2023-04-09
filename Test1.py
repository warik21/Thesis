# Test 1

# - Basic test on the balanced and unbalanced case using the algorithm with log-stabilization
# - The KL constraints are used

import numpy as np
import matplotlib.pyplot as plt
import ot
from utils.utils import *
from utils.ot_utils import full_scalingAlg_pot

n_p = 100
n_q = 100
n_max = 10000
eps = 1.e-2
eps_vec = np.logspace(-1.,-6.,100)
Fun = ['KL', 1.e1]
X,Y = np.linspace(0,1,n_p), np.linspace(0,1,n_q)

#p = [1.0, 2.0]
#q = [2.0, 1.0]
p = make_1D_gauss(n_p, np.floor(3*n_p/4.), 1.)*1 + make_1D_gauss(n_p, np.floor(1*n_p/8.), 2.)*0.5
q = make_1D_gauss(n_q, np.floor(7*n_q/8.), 2.)*1

#p = p/np.sum(p)
#q = q/np.sum(q)

dx = np.ones([n_p,1])/n_p
dy = np.ones([n_q,1])/n_q

C = np.zeros([n_p,n_q],dtype=np.float64)

dist_f2 = lambda a,b : (a-b)**2
dist_f1 = lambda a,b : abs(a-b)
dist_fcos = lambda a,b : -2*np.log(np.cos(np.pi*0.5*np.minimum(1.,np.abs(a-b)/.2)))
#dist_fcos = lambda a,b : np.minimum(1.,np.abs(a-b)/.2)

for it1 in range(n_p):
    for it2 in range(n_q):
        C[it1,it2] = dist_f1(X[it1],Y[it2])


Transport_plan, u, v = full_scalingAlg_pot(p, q, C, eps)
#TODO: Understand how I check whether P is valid
# Supposedly, the sum should be the same as the sum of the source
# distribution, p, and we can perform an easy check for it.

# Plots
# Plot target and source distributions
plt.figure( figsize=(10, 4))
plt.plot(X,p, 'b-', label='Source distribution')
plt.plot(Y,q, 'r-', label='Target distribution')
plt.legend()
plt.title('Source and target distributions')
plt.show()

# Direct output transport plan
# Plot results
plt.figure( figsize=(10, 4))
plt.plot(X,p, 'b-.', label='Source distribution')
plt.plot(Y,q, 'r-.', label='Target distribution')
plt.plot(X, Transport_plan.T @ dx, 'k-', label='Final dist: Transport_plan.T dx')
plt.plot(Y, Transport_plan @ dy, 'g-', label='Final dist: Transport_plan dy')
plt.legend()
plt.title('Source and target distributions')
plt.show()

# Plot transport plan with its marginals
plt.figure( figsize=(8, 8))
plot1D_mat(Transport_plan.T @ dx, Transport_plan @ dy, Transport_plan.T, 'Transport matrix Transport_plan with its marginals')
plt.show()

# Plot transport plan with its marginals
plt.figure( figsize=(8, 8))
plot1D_mat(p, q, Transport_plan, 'Transport matrix Transport_plan with the target and source dist')
plt.show()
