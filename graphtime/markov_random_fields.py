import itertools as _itrt
import numpy as _np
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.linear_model import LogisticRegression as _LogisticRegression

class dMRF(object):
  """
		Implements a dynamic Markov random field model as described in Olsson and Noe 2018
	"""

	def __init__(self, couplings, fields, active_subsystems, estimated = False):
    assert(isinstance(list, couplings))
    assert(isinstance(list, fields))
    assert(len(couplings) == len(fields))

    self.nsubsys_ = len(fields)
    self.couplings_ = couplings
    self.fields_ = fields
		self.active_subsystems_ = active_subsystems 
		self.estimated_ = estimated 

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

def estimate_potts(strajs, lag = 1, stride = 1, fit_bias = False, regularizer = 'l1', C = 1., tol=1e-4):
    """
        Arguments:
        ----------------------
        strajs (list of ndarrays): state of each subsystem as a function of time.
        lag (int=1): lag-time used in auto-regression
        stride (int=1): data stride prior to model estimation. lag should be devisible by this quantity.
        fit_fields (bool=False): flag indicating whether fields/biases (h) are estimated.
        regularizer (str='l1'): flag for regularization scheme ('l1': Laplacian prior, 'l2': Gaussian prior )
        C (float=1.): factor inversely proportional to regularization strength.
        
        returns:
					dMRF : an estimated dynamic Markov Random Field	
    """
    Encoder = _OneHotEncoder()

    strided_strajs = [t[::stride] for t in strajs]
    P0 = np.vstack([t[:-lag//stride] for t in strided_strajs])
    Pt = np.vstack([t[lag//stride:] for t in strided_strajs]) 
    nframes_strided, nsubsys = P0.shape
    active_subsystems_0 = np.where([len(np.unique(P0[:, i]))>1 for i in range(nsubsys)])[0]
    active_subsystems_t = np.where([len(np.unique(P0[:, i]))>1 for i in range(nsubsys)])[0]
    active_subsystems = list(set(active_subsystems_0).intersection(active_subsystems_t))

    #remove constant spins
    P0 = P0[:, active_subsystems]
    Pt = Pt[:, active_subsystems]
    
    #encode multiple
    if len(np.unique(P0))>2:
        P0 = Encoder.fit_transform(P0)
    
    params_potts=[]
    intercepts_potts=[]
    
    for i in range(Pt.shape[1]):
        # if only two categories use one-versus-rest estimation mode
        mc = 'ovr'
        
        if len(np.unique(Pt[:, i]))>2:
            # if we have more than 2 states change multiclass flag to multinomial
            mc = 'multinomial'
        logr = _LogisticRegression(fit_intercept = fit_bias, C = C, penalty = regularizer, 
                                    solver = 'saga', tol=tol, multi_class=mc).fit(P0, Pt[:, i])


        params_potts.append(logr.coef_.copy())
        intercepts_potts.append(logr.intercept_)

    return params_potts, intercepts_potts, Encoder, active_subsystems


#def estimate_dynamic_MRF()
