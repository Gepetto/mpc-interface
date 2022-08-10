#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:22:35 2022

@author: nvilla
"""



import unittest

import numpy as np
from random import randint

import mpc_interface.tools as use

class ToolsTestCase(unittest.TestCase):
    
    def test_extend_matrices(self):
        
        n = randint(1, 15) # number of states
        m = randint(1, 15) # number of inputs
        N = randint(1, 100) # horizon lenght
        
        A = np.eye(n)
        B = np.ones([n, m])
        
        S, U = use.extend_matrices(N, A, B)
        
        self.assertTrue(S.shape == (N, n, n))
        self.assertIsInstance(U, list)
        self.assertEqual(len(U), m)
        self.assertTrue(U[0].shape == (N, N, n))
        
        
if __name__ == "__main__":
    unittest.main()
    