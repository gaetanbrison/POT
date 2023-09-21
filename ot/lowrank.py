
import numpy as np
#import os
import ot 

from ot.utils import unif, dist, list_to_array
from ot.backend import get_backend
from ot.datasets import make_1D_gauss as gauss


########################################################################################
#################################### WORK IN PROGRESS ##################################
########################################################################################

## Questions: 
# How to initialize Q, R and g for lowrank algo ? (possible answer in lowrank algo implemention on github by original researchers)
# Input arguments of lowrank_sinkhorn: cost matrix M ? or X_s / X_t ?
# How to determine the rank r ? (other than 1/r > alpha)
# How to choose alpha ?

## Observations: 
# OT Matrix => full of np.nan => issues with division and overflow in multiply
# Not sure if I interpreted corretly some of the calculations in the original pseudo-code (diag of a vector ?)



################################## LR-DYSKTRA ALGORITHM ##########################################

def LR_Dysktra(eps1, eps2, eps3, r, p1, p2, alpha, stopThr):
    # p1 => poids a
    # p2 => poids b

    eps1, eps2, eps3, p1, p2 = list_to_array(eps1, eps2, eps3, p1, p2)
    nx = get_backend(eps1, eps2, eps3, p1, p2)

    # Initial inputs
    q3_1, q3_2 = nx.ones(r), nx.ones(r)
    v1_, v2_ = nx.ones(r), nx.ones(r)
    q1, q2 = nx.ones(r), nx.ones(r)
    g_ = eps3
    err = 1

    #if err > delta:
    while err > stopThr:
        u1 = p1 / np.dot(eps1, v1_)
        u2 = p2 / np.dot(eps2, v2_)

        g = nx.maximum(alpha, g_ * q3_1)
        q3_1 = (g_ * q3_1) / g
        g_ = g 

        prod1 = ((v1_ * q1) * np.dot(eps1.T, u1))
        prod2 = ((v2_ * q2) * np.dot(eps2.T, u2))
        g = (g_ * q3_2 * prod1 * prod2)**(1/3)

        v1 = g / np.dot(eps1.T,u1)
        v2 = g / np.dot(eps2.T,u2)

        q1 = (v1_ * q1) / v1
        q2 = (v2_ * q2) / v2
        q3_2 = (g_ * q3_2) / g
        
        v1_, v2_ = v1, v2
        g_ = g

        # Compute error
        err1 = nx.sum(nx.abs(u1 * (eps1 @ v1) - p1))
        err2 = nx.sum(nx.abs(u2 * (eps2 @ v2) - p2))
        err = err1 + err2

    #print(n)
    
    Q = u1[:,None] * eps1 * v1[None,:]
    R = u2[:,None] * eps2 * v2[None,:]

    return Q, R, g
    


#################################### LOW RANK SINKHORN ALGORITHM ############################################


def lowrank_sinkhorn(X_s, X_t, a=None, b=None, rank=4, reg=0, metric='sqeuclidean', alpha=1e-10, numIterMax=10000, stopThr=1e-20):

    X_s, X_t = list_to_array(X_s, X_t)
    nx = get_backend(X_s, X_t)

    ns, nt = X_s.shape[0], X_t.shape[0]
    if a is None:
        a = nx.from_numpy(unif(ns), type_as=X_s)
    if b is None:
        b = nx.from_numpy(unif(nt), type_as=X_s)
    
    M = ot.dist(X_s,X_t, metric=metric) 
    
    rank = min(ns, nt, rank)
    r = rank
    
    L = nx.sqrt((2/(alpha**4))*nx.norm(M)**2 + (reg + (2/(alpha**3))*nx.norm(M))**2) # default norm 2
    gamma = 1/(2*L)
    
    # Start values for Q, R, g (not sure ???)
    Q, R, g = nx.ones((ns,r)), nx.ones((nt,r)), nx.ones(r) 
    n_iter = 0

    while n_iter < numIterMax:
        n_iter = n_iter + 1

        eps1 = nx.exp(-gamma*(np.dot(nx.dot(M,R),(1/g)[:,None])) - ((gamma*reg)-1)*nx.log(Q))
        eps2 = nx.exp(-gamma*(np.dot(nx.dot(M.T,Q),(1/g)[:,None])) - ((gamma*reg)-1)*nx.log(R))
        omega = nx.diag(np.dot(np.dot(Q.T,M),R))
        eps3 = nx.exp(gamma*omega/(g**2) - (gamma*reg - 1)*nx.log(g))

        Q, R, g = LR_Dysktra(eps1, eps2, eps3, r, a, b, alpha, stopThr)
    
    P = np.multiply(M,np.dot(Q,np.dot(np.diag(1/g),R.T)))
    
    return P, Q, R, g




############################################################################
## Example from ot.datasets with 2 gaussian distributions 
#############################################################################


Xs, _ = ot.datasets.make_data_classif('3gauss', n=100)
Xt, _ = ot.datasets.make_data_classif('3gauss2', n=150)


P, Q, R, g = lowrank_sinkhorn(Xs,Xt)

print(np.sum(P))
# print("Q:", Q)
# print("R:", R)
# print("g;", g)




######### PREVIOUS EXAMPLE WITH DISTANCE MATRIX INPUT ##########

# n = 100  # nb bins

# # bin positions
# x = np.arange(n, dtype=np.float64)

# # Gaussian distributions
# a = gauss(n, m=20, s=5)  # m= mean, s= std
# b = gauss(n, m=60, s=10)


# # loss matrix
# C = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
# C /= C.max()