# graphtime 
A python module for generation and analysis of undirected graphical models encoding transition densities.

In particular, `graphtime` implements _dynamic Markov random fields_ (dMRF) or "dynamic Ising models". These models use a molecular representation different from 
a single global state, used in for example Markov state models. Instead, a molecular configuration is represented by multiple discrete representations (_sub-systems_), fx torsion-angles or contacts. 
The dMRFs models the interactions between the different sub-systems, or more specifically, how a current configuration of the sub-systems encode the distribution of sub-system states at a time $t+\tau$ in the future. The dMRFs are like Markov state models, fully probabilistic. 

Although this library was developed with application to molecular systems in mind, there is currently no functionality to analyse and featurize molecular simulation data within `graphtime` and this is not planned. However, there are several packages that does this including `MDTraj`, `pyEMMA` and `mdanalysis`. `graphtime` depends on a `straj` as input for estimation, which has the dimensions of $k\times N$, with $k$ being the number of sub-systems and $N$ being the number of frames in the molecular simulation. 

Further details can be found in the manuscript:

S. Olsson and F. Noé "Dynamic Graphical Models of Molecular Kinetics" _in preparation._


### Dependencies
The libary has only been tested Mac OSX.

- python >= 3.6.1
- numpy >= 1.3
- scikit-learn >= 0.19.0
- scipy >= 1.1.0
- msmtools >= 1.2.1
- pyemma >= 2.5.2

### Installation

Clone this repository and install using

`python setup.py install`

to test you installation you can run

`python -m unittest graphtime/test/dmrf_test.py`

### Issues and bugs
If you are having problems unsing this library or discover any bugs please get in touch through the issues section on the `graphtime` github repository. For bug reports please provide a reproducable example.