import itertools as _itrt
import numpy as _np
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.linear_model import LogisticRegression as _LogisticRegression

class dMRF(object):
    """
        Implements a dynamic Markov random field model as described in Olsson and Noe 2018
    """

    def __init__(self, lrs, active_subsystems, lag = 1, enc = None, estimated = False):
        """
            Constructor for dMRF class.

            Arguments:
            --------------------
            lrs :  list of LogisticRegression instances (sklearn)
            active_subsystems : list of active sub-systems
            estimated : bool, indicator whether dMRF is estimated
        """
        
        self.nsubsys_ = len(lrs)
        self.lrs = lrs
        self.encoder = enc
        self.active_subsystems_ = active_subsystems
        self.estimated_ = estimated
        self.lag = lag

    def simulate(self, nsteps, start = None):
        """
            Simulate a trajectory of all active sub-systems described by dMRF.
            
            Arguments:
            --------------------
                nsteps (int) number of steps (in lag-time of model)
                start (ndarray) initial configuration of trajectory. If not given, initial state is randomized.
        """
        # if there is no initial condition generate one
        if not isinstance(start, _np.ndarray): 
            _s = _np.array([lr.classes_[_np.random.randint(0, len(lr.classes_))] for lr in self.lrs])
        else:
            _s = start.copy()

        #pre-generate random numbers gives very slight speedups for large nsteps
        rnd = _np.random.rand((nsteps-1)*self.nsubsys_).reshape((self.nsubsys_, nsteps - 1))
        
        _states = _np.zeros((self.nsubsys_, nsteps))
        _states[:, 0] = _s.copy()

        idx_ = [lr.classes_.tolist() for lr in self.lrs]

        for n in range(nsteps-1):
            #for every sub-system sample new configuration given current global configuration
            encoded_state = self.encoder.transform(_states[:, n].reshape(-1, 1))
            cmfs = _np.cumsum([lr.predict_proba(encoded_state).ravel() for lr in self.lrs], axis = 1)
            for j in range(self.nsubsys_):
                _states[j, n + 1] = idx_[j][_np.searchsorted(cmfs[j, :], rnd[j, n])]

        return _states.T

    def generate_transition_matrix(self, safemode = True, maxdim = 10000):
        """
            Compute full transition probability matrix of dMRF.

            Arguments:
            --------------------
            safemode (bool=True) enable safemode, checks whether output dimension is below 
                maxdim times maxdim prior to allocating memory.
            maxdim (int=10000) maximum dimension of transtion matrix (used if safemode=True)
        """
        ndims = _np.prod([len(lr.classes_) for lr in self.lrs])
        if ndims > maxdim and safemode:
            raise MemoryError(
                'Maximum safe-mode transition matrix dimension ({:i}x{:i}) exceeded.'.format(maxdim, maxdim))
        idx_ = [lr.classes_.tolist() for lr in self.lrs]

        T = _np.zeros((ndims, ndims))
        
        for i, s in enumerate(_itrt.product(*[lr.classes_ for lr in self.lrs])):
            _se = self.encoder.transform(_np.array([s]))
            # compute transition probabilities for each sub-system state at time t+\tau
            tprobs = [lr.predict_proba(_se) for lr in self.lrs]
            for j, z_ in enumerate(_itrt.product(*[lr.classes_ for lr in self.lrs])):
                # compute product of outcome state
                T[i, j] = _np.prod([tprob[:, idx.index(z)] for idx, tprob, z in zip(idx_, tprobs, list(z_))])
        return T

    def get_subsystem_couplings(self):
        """
            Returns estimated sub-system couplings (J), ndarray
        """
        return _np.vstack([lr.coef_ for lr in self.lrs])

    def get_subsystem_biases(self):
        """
            Returns estimated sub-system biases (h), ndarray
        """
        return _np.concatenate([lr.intercept_ for lr in self.lrs])

    def get_active_subsystems(self):
        """
            Return indices of sub-systems active in dMRF
        """
        return self.active_subsystems_
    def get_subsystem_count(self):
        """
            Returns number of active sub-systems
        """
        return self.nsubsys_
    
def estimate_dMRF(strajs, lag = 1, stride = 1, Encoder = _OneHotEncoder(sparse = False), 
                  logistic_regression_kwargs = {'fit_intercept': False, 
                   'penalty': 'l1', 'C': 1., 'tol': 1e-4, 'solver': 'saga'}):
    """
        Estimate dMRF using logistic (binary sub-systems) or softmax (multinomal sub-systems) regression.

        Arguments:
        --------------------
        strajs (list of ndarrays): state of each subsystem as a function of time.
        lag (int=1): lag-time used in auto-regression
        stride (int=1): data stride prior to model estimation. lag should be devisible by this quantity.
        Encoder (sklearn compatible categorical pre-processor): Encoder for spin-states, usually OneHotEncoder is recommended. 
        logistic_regression_kwargs (dict): dictionary of keyword arguments forwarded to 
            sklearn LogisticRegression. 
            The multi_class kwargs is forced to 'ovr' for binary cases and 'multinomial' for multinomial cases.
        
        returns:
            dMRF instance -- estimated dMRF.
    """
    if stride > lag:
        raise ValueError("Stride exceeds lag. Lag has to be larger or equal to stride.")
    strided_strajs = [t[::stride] for t in strajs]
    P0 = _np.vstack([t[:-lag//stride] for t in strided_strajs])
    Pt = _np.vstack([t[lag//stride:] for t in strided_strajs]) 
    nframes_strided, nsubsys = P0.shape
    
    #find active sub-systems
    active_subsystems_0 = _np.where([len(_np.unique(P0[:, i]))>1 for i in range(nsubsys)])[0]
    active_subsystems_t = _np.where([len(_np.unique(Pt[:, i]))>1 for i in range(nsubsys)])[0]
    active_subsystems = list(set(active_subsystems_0).intersection(active_subsystems_t))
    lrs= []
    
    #remove constant sub-systems
    P0 = P0[:, active_subsystems]
    Pt = Pt[:, active_subsystems]
    
    P0 = Encoder.fit_transform(P0)
    
    for i in range(Pt.shape[1]):
        # if only two categories use one-versus-rest estimation mode
        logistic_regression_kwargs['multi_class'] = 'ovr'
        
        if len(_np.unique(Pt[:, i]))>2:
            # if we have more than 2 states change multiclass flag to multinomial
            logistic_regression_kwargs['multi_class'] = 'multinomial'
        logr = _LogisticRegression(**logistic_regression_kwargs).fit(P0, Pt[:, i])
        lrs.append(logr)

    return dMRF(lrs, active_subsystems, lag = lag, enc = Encoder, estimated = True)