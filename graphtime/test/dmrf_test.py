from __future__ import absolute_import
import unittest

import numpy as np
import warnings

from graphtime import markov_random_fields
from graphtime import ising_utils
from graphtime import utils as _ut 

from sklearn.preprocessing import OneHotEncoder as OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

class TestdMRFSimple(unittest.TestCase):
    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(0xDEADBEEF)
        self.lag = 1
        self.stride = 1
        self.straj_binary = [np.array([[0,0],[0,1],[1,1],[1,1],[1,0],[1,0],[1,1],[0,1],[0,0],[0,0]])]
        self.straj_multinomial = [np.array([[0,0],[0,1],[1,2],[1,2],[1,0],[2,0],[1,1],[0,1],[2,0],[0,2]])]
        self.straj_bin_mult =  [np.array([[0,0], [0,2], [1,1], [1,2], [1,2], [1,2], [1,1], [0,2], [0,1], [1,1], [0,1], [0,0], [0,2]])]
        
    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_dmrf_binary(self):
        """ self-consistency, estimation and convienence function """
        dmrf_bin = markov_random_fields.estimate_dMRF(self.straj_binary, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = OneHotEncoder(sparse = False))
        self.assertEqual(self.lag, dmrf_bin.lag)
        self.assertTrue(dmrf_bin.estimated_)
        self.assertEqual(len(dmrf_bin.active_subsystems_), 2)
        tmat = dmrf_bin.generate_transition_matrix()
        self.assertEqual(tmat.shape, (4, 4))

    def test_dmrf_multinomial(self):
        """ self-consistency, estimation and convienence function multinomial """
        dmrf_multinom = markov_random_fields.estimate_dMRF(self.straj_multinomial, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = OneHotEncoder(sparse = False))
        self.assertEqual(self.lag, dmrf_multinom.lag)
        self.assertTrue(dmrf_multinom.estimated_)
        self.assertEqual(len(dmrf_multinom.active_subsystems_), 2)
        tmat = dmrf_multinom.generate_transition_matrix()
        self.assertEqual(tmat.shape, (9, 9))

    def test_dmrf_multinomial_binary(self):        
        """ self-consistency, estimation and convienence function, multiple trajectories"""
        dmrf_multinom_bin = markov_random_fields.estimate_dMRF(self.straj_multinomial+self.straj_binary, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = OneHotEncoder(sparse = False))
        self.assertEqual(self.lag, dmrf_multinom_bin.lag)
        self.assertTrue(dmrf_multinom_bin.estimated_)
        self.assertEqual(len(dmrf_multinom_bin.active_subsystems_), 2)
        tmat = dmrf_multinom_bin.generate_transition_matrix()
        self.assertEqual(tmat.shape, (9, 9))

    def test_dmrf_one_multinomial_one_binary(self):        
        """ self-consistency, estimation and convienence function, single trajectory"""
        dmrf_mbin = markov_random_fields.estimate_dMRF(self.straj_bin_mult, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = OneHotEncoder(sparse = False))
        self.assertEqual(self.lag, dmrf_mbin.lag)
        self.assertTrue(dmrf_mbin.estimated_)
        self.assertEqual(len(dmrf_mbin.active_subsystems_), 2)
        tmat = dmrf_mbin.generate_transition_matrix()
        self.assertEqual(tmat.shape, (6, 6))

class TestdMRFIsing(unittest.TestCase):
    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(0xDEADBEEF)
        self.lag = 1
        self.stride = 1
        self.nspins = 3
        self.alpha = 0.10
        self.IsingTmat = ising_utils.Ising_tmatrix(self.nspins, alpha = self.alpha, gamma = 0)
        self.ising_states = ising_utils.all_Ising_states(self.nspins)
        self.Isingdata_state = _ut.simulate_MSM(self.IsingTmat, 500000, s0 = 0)
        self.Isingdata = [np.array(self.ising_states[self.Isingdata_state])]


    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)
    def test_dmrf_Ising_one_spin(self):
        """ estimation with 3 binary uncoupled spins, tests custom encoder """
        IsingDMRF = markov_random_fields.estimate_dMRF(self.Isingdata, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = LabelBinarizer(neg_label = -1, pos_label = 1))
        self_couplings = np.diag(np.vstack([lr.coef_ for lr in IsingDMRF.lrs]))

        # compare estimated sub-system flip rates to analytical values for zero-coupling Ising model
        self.assertTrue(np.allclose(np.ones(self.nspins)*self.alpha*self.lag, -np.log(np.tanh(self_couplings/2.)), rtol = 1e-3, atol = 1e-3))

        # check whether estimated fields are almost 0
        self.assertTrue(np.allclose(np.zeros(self.nspins), np.concatenate([lr.intercept_ for lr in IsingDMRF.lrs])))
    
    def test_dmrf_Ising_simulate_test(self):
        """ estimation with 3 binary uncoupled spins, test simulation with and without initial condition specified """
        IsingDMRF2 = markov_random_fields.estimate_dMRF(self.Isingdata, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = LabelBinarizer(neg_label = -1, pos_label = 1))

        # simulation with specified initial condition
        synth_traj = IsingDMRF2.simulate(nsteps = 2, start = np.ones(self.nspins))
        # simulation without specified initial condition
        synth_traj2 = IsingDMRF2.simulate(nsteps = 2)

        self.assertTrue(np.allclose(synth_traj, np.array([[ 1, 1, 1], 
                                                          [ 1, 1, 1]])))

        self.assertTrue(np.allclose(synth_traj2, np.array([[ -1, 1, -1], 
                                                           [ -1, 1, -1]])))

    def test_dmrf_Ising_strided_estimate(self):
        """ estimation with 3 binary uncoupled spins, compares strided and unstrided self-coupling estimates """

        fact = 2

        IsingDMRF_strided = markov_random_fields.estimate_dMRF(self.Isingdata, 
            lag = self.lag*fact, 
            stride = self.stride*fact, 
            Encoder = LabelBinarizer(neg_label = -1, pos_label = 1))

        IsingDMRF_unstrided = markov_random_fields.estimate_dMRF(self.Isingdata, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = LabelBinarizer(neg_label = -1, pos_label = 1))
        
        sc_strided = np.diag(np.vstack([lr.coef_ for lr in IsingDMRF_strided.lrs]))
        sc_unstrided = np.diag(np.vstack([lr.coef_ for lr in IsingDMRF_unstrided.lrs]))

        # compare strided and unstrided rate estimates
        self.assertTrue(np.allclose(sc_unstrided/2., np.arctanh(np.exp(np.log(np.tanh(sc_strided/2.))/fact)), rtol = 1e-3, atol = 1e-3 ))
