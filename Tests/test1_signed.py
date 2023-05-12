import matplotlib.pyplot as plt

from utils.utils import *
from utils.Visualizations import *

n_p = 100
n_q = 100
n_max = 10000
eps = 1.e-2
X,Y = np.linspace(0,1,n_p), np.linspace(0,1,n_q)

p = make_1D_gauss(n_p, np.floor(1 * n_p / 4.), 2.) + make_1D_gauss(n_p, np.floor(2 * n_p / 4.), 2.) * (-0.5)
q = make_1D_gauss(n_q, np.floor(5 * n_q / 8.), 2.) + make_1D_gauss(n_q, np.floor(7 * n_q / 8.), 2.) * (-0.5)

dx = np.ones([n_p,1])/n_p
dy = np.ones([n_q,1])/n_q

C = np.zeros([n_p,n_q],dtype=np.float64)

dist_f1 = lambda a,b : abs(a-b)
dist_f2 = lambda a,b : (a-b)**2
# TODO: look at pdist1, pdist2
for it1 in range(n_p):
    for it2 in range(n_q):
        C[it1,it2] = dist_f2(X[it1],Y[it2])

## To work with the signed measures
p_pos, p_neg = split_signed_measure(p)
q_pos, q_neg = split_signed_measure(q)

p_tilde = p_pos + q_neg
q_tilde = q_pos + p_neg


transport_plan_pos, Transport_cost_pos = calc_transport_pot_emd(p_pos, q_pos, C)
transport_plan_neg, Transport_cost_neg = calc_transport_pot_emd(p_neg, q_neg, C)

transport_plan = transport_plan_pos + transport_plan_neg


if sum(q_neg) == sum(p_neg):
    print("q_pos and p_pos are the same")
else:
    print("q_pos and p_pos are different")

# Check if q_neg and p_neg are the same
if sum(q_pos) == sum(p_pos):
    print("q_neg and p_neg are the same")
else:
    print("q_neg and p_neg are different")

# Plots
# plot_transport_map_with_marginals(X, Y, p, q, Transport_plan, 'Transport matrix with the source and target dist')
plot_transport_map_with_marginals(X, Y, p, q, transport_plan, 'Transport matrix with the source and target dist')

# target and source distributions
# plot_distribution(X, p, q, 'Source and target distributions')
plot_distribution(X, p_tilde, q_tilde, 'Source and target distributions')

# Marginals of the transport map
# target and source distributions
# plot_marginals(X, p, q, Transport_plan, 'Marginals of the transport map Vs target and source distributions')
# plot_marginals(X, p_tilde, q_tilde, transport_plan2, 'Marginals of the transport map Vs target and source distributions')

# not transported mass
# Positive value means mass that is left over.
# Negative values means that mass is missing.
# plot_not_transported_mass(X, Y, p, q, Transport_plan, 'Not transported mass')
# plot_not_transported_mass(X, Y, p_tilde, q_tilde, Transport_plan2, 'Not transported mass')

# Plot transport plan with its marginals
#plot_transport_map(X, Y, p, q, Transport_plan, 'Transport matrix with its marginals')

# Plot transport plan with its marginals
# plot_transport_map_with_marginals(X, Y, p, q, Transport_plan, 'Transport matrix with the source and target dist')




