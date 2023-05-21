from utils.Visualizations import *
import numpy as np
import ot


n_p = 4
n_q = 4
n_max = 10000
eps = 1.e-2
X,Y = np.linspace(0,1,n_p), np.linspace(0,1,n_q)


p = np.array([1.0,-1.0,0.0,0.0])
q = np.array([0.0,0.0,-1.0,1.0])

C = np.zeros([n_p,n_q],dtype=np.float64)

dist_f1 = lambda a,b : abs(a-b)
dist_f2 = lambda a,b : (a-b)**2
for it1 in range(n_p):
    for it2 in range(n_q):
        C[it1,it2] = dist_f1(X[it1],Y[it2])

## To work with the signed measures
p_pos, p_neg = split_signed_measure(p)
q_pos, q_neg = split_signed_measure(q)

K_t : np.ndarray = np.exp(C / (-eps))

#Positives:
transport_plan_pos, transport_cost_pos = calc_transport_pot_emd(p_pos, q_pos, C)
plt.figure(figsize=(10,10))
plot1D_mat(p_pos, q_pos, transport_plan_pos,'Transport map with the source and target dist for positives')
plt.show()

#Negatives:
transport_plan_neg, transport_cost_neg = calc_transport_pot_emd(p_neg, q_neg, C)
plt.figure(figsize=(10,10))
plot1D_mat(p_neg, q_neg, transport_plan_neg,'Transport map with the source and target dist for negatives')
plt.show()

#United:
plt.figure(figsize=(10,10))
plot1D_mat(p, q, transport_cost_pos + transport_cost_neg,'Transport map with the source and target dist for original')
plt.show()


