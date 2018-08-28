from __future__ import print_function
import numpy as _np
from scipy.linalg import expm as _expm
import itertools as _itrt

def Ising_tmatrix(nspins, alpha=0.1, gamma=0.95, ratematrix=False):
    """
        Implements Glaubers master equation variant of the (1D) Ising model with periodic boundary conditions

        (J Math Phys 4 294 (1963); doi: 10.1063/1.1703954)
        nspins: number of spins in model (Note: returns 2^nspins times 2^nspins matix)
        alpha: basal spin flip-rate, defines time-scale (=0.1)
        gamma: gamma is equal to tanh(\beta 2J) where J is the spin-spin coupling constant in a corresponding Ising model, and \beta is the inverse temperature. 
        ratematrix: return rate matrix as well
    """
    W = _np.zeros((2**nspins, 2**nspins))
    for i, s in enumerate(_itrt.product([-1, 1], repeat = nspins)):
        s = _np.array(s)
        for j, c in enumerate(_itrt.product([-1, 1], repeat = nspins)):
            c = _np.array(c)
            if _np.all(s == c):
                #Diagonal is filled later.
                continue
            else:
                flipped = _np.where(s!=c)[0]
                if len(flipped)==1:
                    f = flipped[0]
                    W[i, j] = 0.5*alpha*(1.-0.5*gamma*s[f]*(s[f-1]+s[(f+1)%nspins]))
                else:
                    pass
    #fill diagonal
    W[_np.diag_indices(2**nspins)] = -W.sum(axis=1)
    #compute transition matrix
    T = _expm(W)
    if ratematrix:
        return T, W
    else:
        return T

def Ising_to_discrete_state(X):
    """
        Maps a trajectory of spin-states to a corresponding trajectory of unique states. 
            Useful when estimating models with global discretization fx. MSMs. 

        X : list of ndarrays ( trajectories of spin states (T, M) ) where T is the number of time-steps and M is the number of binary spins
    
            returns:
            dts : list of lists ( discrete state trajectories)
	"""
    dts = []
    for x in X:
        x[_np.where(x==-1)] = 0
        dts.append ([int(''.join(map(str, f)), 2) for f in x])
    return dts

all_Ising_states = lambda nspins:_np.array(list(_itrt.product([-1, 1], repeat = nspins)))


