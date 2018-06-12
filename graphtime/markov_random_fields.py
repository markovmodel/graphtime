
import itertools as _itrt
import numpy as _np
class MRF(object):
  def __init__(self, couplings, fields):
    assert(isinstance(list, couplings))
    assert(isinstance(list, fields))
    assert(len(couplings) == len(fields))

    self.nsubsys = len(fields)
    self.couplings = couplings
    self.fields = fields

  def simulate(self, nsteps, start = None):
    """ assumes everything is binary """
    if not isinstance(start, np.ndarray):
       _s = 2*_np.random.binomial(1, 0.5, size = )-1
    else:
       _s = 2*start.copy()-1
    _states = _np.zeros((self.nsubsys, nsteps))
    _states[:, 0] = _s.copy()

    rnd = _np.random.rand((n_steps-1)*self.nsubsys]).reshape((self.nsubsys, nsteps - 1))
    for i in range(nsteps-1):
      tprob = 1./(1.+np.exp(-self.couplings.dot(_s)-self.fields))
      _s = 2*(rnd[:, i] < tprob).astype(int)-1
      _states[:, i + 1] = _s.copy()
   
   return _states.T

  def generate_transition_matrix(self, force = True):
    T = np.zeros((2**self.nsubsys, 2**self.numsubsys))
    for i,s_ in enumerate(_itrt.product([-1,1], repeat = nspins)):
      s = np.array(s_)
      tprob = 1./(1.+np.exp(-self.couplings.dot(s)-self.fields))
      for j,z_ in enumerate(product([-1,1], repeat=nspins)):
        z = np.array(z_)
        T[i, j] = np.prod(tprob[z==1])*np.prod(1-tprob[z==-1])
    return T


