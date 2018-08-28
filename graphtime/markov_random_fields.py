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
                start (list) initial configuration of trajectory. If not given, initial state is randomized.
        """
        # if there is no initial condition generate one
        if not isinstance(start, _np.ndarray): 
            _s = _np.array([lr.classes_[_np.random.randint(0, len(lr.classes_))] for lr in self.lrs])
        
        #pre-generate random numbers gives very slight speedups for large nsteps
        rnd = _np.random.rand((nsteps-1)*self.nsubsys_).reshape((self.nsubsys_, nsteps - 1))
        
        _states = _np.zeros((self.nsubsys_, nsteps))
        _states[:, 0] = _s.copy()

        for n in range(nsteps-1):
            #for every sub-system sample new configuration given current global configuration
            for j,lr in enumerate(self.lrs):
                cmf = _np.cumsum(lr.predict_proba(self.encoder.transform([_states[:, n]])))
                _states[j, n + 1] = _np.searchsorted(cmf, rnd[j, n])

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
    
    
def estimate_dMRF(strajs, lag = 1, stride = 1, Encoder = _OneHotEncoder(sparse=False), 
                  logistic_regression_kwargs = {'fit_intercept': False, 
                   'penalty': 'l1', 'C': 1., 'tol': 1e-4, 'solver': 'saga'}):
    """
        Estimate dMRF using logistic (binary sub-systems) or softmax (multinomal sub-systems) regression.

        Arguments:
        --------------------
        strajs (list of ndarrays): state of each subsystem as a function of time.
        lag (int=1): lag-time used in auto-regression
        stride (int=1): data stride prior to model estimation. lag should be devisible by this quantity.
        logistic_regression_kwargs (dict): dictionary of keyword arguments forwarded to 
            sklearn LogisticRegression. 
            The multi_class kwargs is forced to 'ovr' for binary cases and 'multinomial' for multinomial cases.
        
        returns:
            params_potts (list): coupling parameters for each sub-system
            intercepts_potts (list): intercept parameters for each sub-system
            Encoder: OneHotEncoder to used to featurize
            active_subsystems: indices of non constant subsystems
    """
    Encoder = Encoder

    strided_strajs = [t[::stride] for t in strajs]
    P0 = _np.vstack([t[:-lag//stride] for t in strided_strajs])
    Pt = _np.vstack([t[lag//stride:] for t in strided_strajs]) 
    nframes_strided, nsubsys = P0.shape
    
    #find active sub-systems
    active_subsystems_0 = _np.where([len(_np.unique(P0[:, i]))>1 for i in range(nsubsys)])[0]
    active_subsystems_t = _np.where([len(_np.unique(Pt[:, i]))>1 for i in range(nsubsys)])[0]
    active_subsystems = list(set(active_subsystems_0).intersection(active_subsystems_t))
    lrs= []
    
    #remove constant spins
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
    