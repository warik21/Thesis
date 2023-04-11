import matplotlib.pyplot as plt

from utils.utils import *
from utils.ot_utils import full_scalingAlg_pot

n_p = 100
n_q = 100
n_max = 100000
eps = 1.e-2
eps_vec = np.logspace(-1.,-4.,10)
lda = 1.e1
Fun = ['TV', lda]
X,Y = np.linspace(0,1,n_p), np.linspace(0,1,n_q)

p =  make_1D_gauss(n_p, np.floor(3*n_p/4.), 2.)*1 + make_1D_gauss(n_p, np.floor(1*n_p/8.), 2.)*0.5
q =  make_1D_gauss(n_q, np.floor(7*n_q/8.), 2.)*1

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
# Transport_plan2, a_t, b_t, primals, duals, pdgaps = full_scalingAlg(C,Fun,p,q,eps_vec,dx,dy,n_max)

print('*******************')
print('Transport details')
# print('Transport cost <P,C> = %f'%(np.sum(Transport_plan*C)))
print('Elements transported = %d' % (np.count_nonzero(Transport_plan - np.diag(np.diagonal(Transport_plan)))))
print('target |q - Transport_plan.Tdx|_1 = %f' % (np.sum(abs(q - Transport_plan.T @ dx))))
print('source |p - Rdy|_1 = %f' % (np.sum(abs(p - Transport_plan @ dy))))
print('Int(p) = %.2f , |p| = %.2f' % (np.sum(p), np.sum(abs(p))))
print('Int(q) = %.2f , |q| = %.2f' % (np.sum(q), np.sum(abs(q))))

# --------------------------------------#
# Plots

# target and source distributions
plt.figure(figsize=(10, 4))
plt.plot(X, p, 'b-', label='Source dist: p')
plt.plot(Y, q, 'r-', label='Target dist: q')
plt.legend()
plt.title('Source and target distributions')
plt.show()

# Marginals of the transport map
# target and source distributions
plt.figure(figsize=(10, 4))
plt.plot(X, p, 'b-.', label='Source dist: p')
plt.plot(Y, q, 'r-.', label='Target dist: q')
plt.plot(X, Transport_plan.T @ dx, 'k-', label='Final source dist (q): Transport_plan.T dx')
plt.plot(Y, Transport_plan @ dy, 'g-', label='Final target dist (p): Transport_plan dy')
plt.legend()
plt.title('Marginals of the transport map Vs target and source distributions')
plt.show()

# not transported mass
# Positive value means mass that is left over.
# Negative values means that mass is missing.
plt.figure(figsize=(10, 4))
plt.plot(X, p, 'b-.', label='Source dist: p')
plt.plot(Y, q, 'r-.', label='Target dist: q')
plt.plot(X, p - Transport_plan @ dy, 'g-', label='NTM from source (p)')
plt.plot(Y, Transport_plan.T @ dx - q, 'k-', label='NTM from target (q)')
plt.legend()
plt.title('Not Transported Mass (NTM)')
plt.show()

# Plot transport plan with its marginals
plt.figure(figsize=(8, 8))
plot1D_mat(Transport_plan @ dy, Transport_plan.T @ dx, Transport_plan, 'Transport matrix Transport_plan with its marginals')
plt.show()

# Plot transport plan with its marginals
plt.figure(figsize=(8, 8))
plot1D_mat(p, q, Transport_plan, 'Transport matrix Transport_plan with the target and source dist')
plt.show()