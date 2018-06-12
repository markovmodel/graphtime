import numpy as _np

def simulate_MSM(T, N_steps, s0 = 0):
  """
    Fast trajectory generator for Markov state models (from Fabian Paul)
    T: transition matrix
    N_steps: number of steps

  """
  dtraj = _np.zeros(N_steps, dtype = _np.uint16)
  s = s0 
  T_cdf = T.cumsum(axis=1)
  for t in range(N_steps):
    dtraj[t] = s
    s = _np.searchsorted(T_cdf[s, :], _np.random.rand())
  return dtraj
