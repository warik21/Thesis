# Test 1

# - Basic test on the balanced and unbalanced case using the algorithm with log-stabilization
# - The KL constraints are used

from utils.utils import *
from utils.Visualizations import *

n_p = 4
n_q = 4
n_max = 10000
eps = 1.e-1
X, Y = np.linspace(0, 1, n_p), np.linspace(0, 1, n_q)

# q = make_1D_gauss(n_p, np.floor(3*n_p/4.), 1.)*1 + make_1D_gauss(n_p, np.floor(1*n_p/8.), 2.)*0.5
# p = make_1D_gauss(n_q, np.floor(7*n_q/8.), 2.)*1
p = np.array([1, 0, 2, 0])
q = np.array([0, 1, 0, 2])

dx = np.ones([n_p, 1]) / n_p
dy = np.ones([n_q, 1]) / n_q

C = np.zeros([n_p, n_q], dtype=np.float64)

dist_f2 = lambda a, b: (a - b) ** 2
dist_f1 = lambda a, b: abs(a - b)
dist_fcos = lambda a, b: -2 * np.log(np.cos(np.pi * 0.5 * np.minimum(1., np.abs(a - b) / .2)))
# dist_fcos = lambda a,b : np.minimum(1.,np.abs(a-b)/.2)
eps_vec = np.logspace(-1., -6., 10)
Fun = ['KL', 1.e1]

# Calculate the cost matrix, this is inefficient for large items.
for it1 in range(n_p):
    for it2 in range(n_q):
        C[it1, it2] = dist_f1(X[it1], Y[it2])

p = p / np.sum(p)
q = q / np.sum(q)

# Calculate the transport plan
Transport_plan, transport_cost, u, v = calc_transport_pot_sinkhorn(p, q, C, eps)

# Plots
# Plot target and source distributions
plot_marginals(X, p, q, 'Source  and target distribution')

# Direct output transport plan
plot_marginals(X, p, q, Transport_plan, 'Transport matrix with its marginals')

# Plot transport plan with its marginals
plot_transport_map_with_marginals(p, q, Transport_plan, 'Transport matrix with its marginals')

# Plot transport plan with its marginals
plot_transport_map_with_marginals(p, q, Transport_plan, '')
