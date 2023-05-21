from utils.Visualizations import *
from utils.utils import *


n_p = 100
n_q = 100
n_max = 10000
eps = 1.e-2
dx = np.ones(n_p)/n_p
dy = np.ones(n_q)/n_q
eps_vec = np.logspace(-1.,-6.,10)
Fun = 'KL'
X,Y = np.linspace(0,1,n_p), np.linspace(0,1,n_q)

p = make_1D_gauss(n_p, np.floor(3 * n_p / 4.), 1.) * 1 + make_1D_gauss(n_p, np.floor(1 * n_p / 8.), 2.) * 0.5
q = make_1D_gauss(n_q, np.floor(7 * n_q / 8.), 2.) * 1

C = np.zeros([n_p,n_q],dtype=np.float64)
dist_f2 = lambda a,b : (a-b)**2
# Calculate the cost matrix, this is inefficient for large items.
for it1 in range(n_p):
    for it2 in range(n_q):
        C[it1,it2] = dist_f2(X[it1],Y[it2])

f, g, plan = unbalanced_sinkhorn(p, q, C, eps, return_plan=True)


# Plots
# Plot target and source distributions
plot_distribution(X, p, q, title='Source and target distributions')

# Direct output transport plan
# Plot results
plot_marginals(X, p, q, plan, title='Source and target distributions')

# Plot transport plan with its marginals
plot_transport_map_with_marginals(p, q, plan, title='Transport matrix with its marginals')

