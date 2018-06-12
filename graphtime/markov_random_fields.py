from _base import Model as _Model
import numpy as _np

class MRF(_Model):
	def __init__(self, couplings, fields):
		assert(isinstance(list, couplings))
		assert(isinstance(list, fields))
		assert(len(couplings) == len(fields))
		
    self.nsubsys = len(fields)
		self.couplings = couplings
		self.fields = fields

  def simulate(self, nsteps, start = None):
			if not isinstance(start, np.ndarray):
					_s = 2*_np.random.binomial(1, 0.5, size = )-1
			else:
					_s = 2*start.copy()-1
			_states = _np.zeros((h.shape[0], n_steps))
			_states[:, 0] = _s.copy()
			
			rnd = _np.random.rand((n_steps-1)*h.shape[0]).reshape((h.shape[0], n_steps - 1))
			for i in range(n_steps-1):
					tprob = 1./(1.+np.exp(-_J.dot(_s)-h))
					_s = 2*(rnd[:, i] < tprob).astype(int)-1
					_states[:, i + 1] = _s.copy()
			return _states.T


class BinaryMRF(_Model, MRF):

  def generate_transition_matrix(self, force = True):
    nspins=7
    T = np.zeros((2**nspins, 2**nspins))
    for i,s_ in enumerate(product([-1,1], repeat=nspins)):
        s = np.array(s_)
        if force_reversible:
            tprob = 1./(1.+np.exp(-Jp.dot(s)-h))

        else:
            
            tprob = 1./(1.+np.exp(-J.dot(s)-h))
        for j,z_ in enumerate(product([-1,1], repeat=nspins)):
            z = np.array(z_)
            T[i, j] = np.prod(tprob[z==1])*np.prod(1-tprob[z==-1])
    return T


class GeneralMRF(_Model, MRF):
  def simulate(self, nsteps, start = None):

  def generate_transition_matrix(self):
  
