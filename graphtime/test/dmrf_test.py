from __future__ import absolute_import
import unittest

import numpy as np
import warnings

from graphtime import markov_random_fields
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
        self.straj_bin_mult =  [np.array([[0,0], [0,2], [1,1], [1,2], [1,2], [1,2], [1,1], [0,2], [0,1], [1,1]])]
        
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
        """ self-consistency, estimation and convienence function, multiple trajectories"""
        dmrf_mbin = markov_random_fields.estimate_dMRF(self.straj_bin_mult, 
            lag = self.lag, 
            stride = self.stride, 
            Encoder = OneHotEncoder(sparse = False))
        self.assertEqual(self.lag, dmrf_mbin.lag)
        self.assertTrue(dmrf_mbin.estimated_)
        self.assertEqual(len(dmrf_mbin.active_subsystems_), 2)
        tmat = dmrf_mbin.generate_transition_matrix()
        self.assertEqual(tmat.shape, (6, 6))

