# graphtime 
A python module for estimation and analysis of dynamic graphical models to encode transition densities.

In particular, `graphtime` implements _dynamic Markov random fields_ (dMRF) or "dynamic Ising models", as a particular case of dynamic graphical models (DGM). DGMs represent molecular configurations using multiple features (_sub-systems_), fx torsion-angles or contacts.  This is in contrast to the single global state, used in for example Markov state models. The advantage of this kind of model is that the number of parameters needed to be estimated is only quadratic in the number of sub-system states, rather than being exponential in the number of meta-stable states. 

The dMRFs models the interactions between the different sub-systems, or more specifically, how a current configuration of the sub-systems encode the distribution of sub-system states at a time $t+\tau$ in the future. The dMRFs are like Markov state models, fully probabilistic. 

Although this library was developed with application to molecular systems in mind, there is currently no functionality to analyse and featurize molecular simulation data within `graphtime` and this is not planned. However, there are several packages that does this including `MDTraj`, `pyEMMA` and `mdanalysis`. `graphtime` depends on a `straj` as input for estimation, which is a list of `numpy` arrays with the dimensions $k\times N$, with $k$ being the number of sub-systems and $N$ being the number of frames a the molecular simulation. 

Further details can be found in the manuscript:

S. Olsson and F. Noé "Dynamic Graphical Models of Molecular Kinetics" _in review._ [pre-print](https://www.biorxiv.org/content/early/2018/11/09/467050)


### Dependencies
The `graphtime` library is minimalisitic and makes extensive use of `numpy` and `sklearn`.

- python >= 3.6.1
- numpy >= 1.3
- scikit-learn >= 0.19.0
- scipy >= 1.1.0
- msmtools >= 1.2.1
- pyemma >= 2.5.2

### Installation

Clone this repository and test
`python setup.py test`

If succesfull install using
`python setup.py install`

### Issues and bugs
If you are having problems using this library or discover any bugs please get in touch through the issues section on the `graphtime` github repository. For bug reports please provide a reproducable example.