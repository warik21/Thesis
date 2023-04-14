import matplotlib.pyplot as plt

from utils.utils import *
from utils.ot_utils import full_scalingAlg_pot

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
for it1 in range(n_p):
    for it2 in range(n_q):
        C[it1,it2] = dist_f1(X[it1],Y[it2])

Transport_plan, u, v = full_scalingAlg_pot(p, q, C, eps)

## To work with the signed measures
p_pos = np.zeros(p.shape)
p_neg = np.zeros(p.shape)
q_pos = np.zeros(q.shape)
q_neg = np.zeros(q.shape)

sign_p = np.sign(p)
sign_q = np.sign(q)

p_pos[sign_p>0] = p[sign_p>0]
p_neg[sign_p<0] = -p[sign_p<0]
q_pos[sign_q>0] = q[sign_q>0]
q_neg[sign_q<0] = -q[sign_q<0]

p_tilde = p_pos + q_neg
q_tilde = q_pos + p_neg

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

# target and source distributions
plt.figure( figsize=(10, 4))
plt.plot(X,p, 'b-', label='Source dist: p')
plt.plot(Y,q, 'r-', label='Target dist: q')
plt.legend()
plt.title('Source and target distributions')
plt.show()

# Marginals of the transport map
# target and source distributions
plt.figure( figsize=(10, 4))
plt.plot(X,p, 'b-.', label='Source dist: p')
plt.plot(Y,q, 'r-.', label='Target dist: q')
plt.plot(X, Transport_plan.T @ dx, 'k-', label='Final source dist (q): Transport_plan.T dx')
plt.plot(Y, Transport_plan @ dy, 'g-', label='Final target dist (p): Transport_plan dy')
plt.legend()
plt.title('Marginals of the transport map Vs target and source distributions')
plt.show()

# Marginals of the transport map -_> for the signed measures
# target and source distributions
plt.figure( figsize=(10, 4))
plt.plot(X,p, 'b-.', label='Source dist: p (signed)')
plt.plot(Y,q, 'r-.', label='Target dist: q (signed)')
plt.plot(X, (Transport_plan.T @ dx) * np.sign(q), 'k-', label='Final source dist (q): Transport_plan.T dx (signed)')
plt.plot(Y, (Transport_plan @ dy) * np.sign(p), 'g-', label='Final target dist (p): Transport_plan dy (signed)')
plt.legend()
plt.title('(SIGNED) Marginals of the transport map Vs target and source distributions')
plt.show()

# not transported mass
# Positive value means mass that is left over.
# Negative values means that mass is missing.
plt.figure( figsize=(10, 4))
plt.plot(X,p, 'b-.', label='Source dist: p')
plt.plot(Y,q, 'r-.', label='Target dist: q')
plt.plot(X, p_tilde - Transport_plan @ dy, 'g-', label='NTM from source (p - Rdy)')
plt.plot(Y, Transport_plan.T @ dx - q_tilde, 'k-', label='NTM from target (Transport_plan.Tdx - q)')
plt.legend()
plt.title('Not Transported Mass (NTM)')
plt.show()

# Plot transport plan with its marginals
plt.figure( figsize=(8, 8))
plot1D_mat(Transport_plan @ dy, Transport_plan.T @ dx, Transport_plan, 'Transport matrix with its marginals')
plt.colorbar(plt.imshow(Transport_plan))
plt.show()

# Plot transport plan with its marginals
plt.figure( figsize=(8, 8))
plot1D_mat(p, q, Transport_plan, 'Transport matrix with the target and source dist')
plt.show()




