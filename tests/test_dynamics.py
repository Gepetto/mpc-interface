#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:35:54 2022

@author: nvilla
"""


import unittest

import numpy as np
from collections.abc import Iterable

import gecko.dynamics as dy
import gecko.tools as use

class DynamicsTestCase(unittest.TestCase):
    def setUp(self):
        #### Settings for control systems
        
        inputs = ["u0", "u1", "u2", "u3", "u4", "u5"]
        states = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
        
        A = np.zeros([8, 8])
        A[-1] = np.ones([8])
        A[:-1, 1:] = np.eye(7)
        
        B = np.ones([8, 6])
        
        axes = ["_x", "_y", "_z", "_a", "_b"]
        
        def how_to_update(cntS, **kargs):
            factor = kargs["factor"]
            cntS.B = cntS.A@cntS.B * factor
        
        cnt_sys1 = dy.ControlSystem(inputs, states, A, B, axes, 
                                    time_variant=True,
                                    how_to_update_matrices=how_to_update)
        cnt_sys2 = dy.ControlSystem(inputs, states, A, B, axes, 
                                    time_variant=False,
                                    how_to_update_matrices=how_to_update)
        cnt_sys3 = dy.ControlSystem(inputs, states, A, B, axes, 
                                    time_variant=True)
        
        self.cnt_sys1=cnt_sys1;self.cnt_sys2=cnt_sys2;self.cnt_sys3=cnt_sys3
        self.A=A;self.B=B;self.inputs=inputs;self.states=states;self.axes=axes
        
        #### Setting extended control systems
        horizon_lenght = 20
        
        ext_sys1 = dy.ExtendedSystem.from_cotrol_system(cnt_sys1, 
                                                          "x",
                                                          horizon_lenght)
        ext_sys2 = dy.ExtendedSystem.from_cotrol_system(cnt_sys2, 
                                                          "e",
                                                          horizon_lenght)
        ext_sys3 = dy.ExtendedSystem.from_cotrol_system(cnt_sys3, 
                                                          "r",
                                                          horizon_lenght)
        self.ext_sys1=ext_sys1;self.ext_sys2=ext_sys2;self.ext_sys3=ext_sys3
        self.N = horizon_lenght;
        
        ##### Settings for single variable
        def in_this_way(singVar, **kargs):
            if isinstance(kargs["new_sizes"], Iterable):
                sizes = kargs["new_sizes"]  
            else: sizes = [kargs["new_sizes"]]
            singVar.domain.update({var:sizes[ID] 
                                   for var, ID in singVar.domain_ID.items()})
            
        domVar1 = dy.DomainVariable("n", 20, ["_x", "_y"], time_variant=True,
                                    how_to_update_size=in_this_way)
        domVar2 = dy.DomainVariable("n", 20, ["_x", "_y"], time_variant=False,
                                    how_to_update_size=in_this_way)
        domVar3 = dy.DomainVariable(["n", "H"], [20, 20], ["_x", "_y"])
        
        self.domVar1 = domVar1; self.domVar2 = domVar2; self.domVar3 = domVar3
        
    def test_control_system(self):
        self.assertEqual(self.cnt_sys1.state_names, self.states)
        self.assertEqual(self.cnt_sys2.input_names, self.inputs)
        
        self.assertTrue((self.cnt_sys3.A==self.A).all())
        self.assertTrue((self.cnt_sys1.B==self.B).all())
        
        self.cnt_sys1.update_matrices(factor = 3)
        self.cnt_sys2.update_matrices(factor = 3)
        self.cnt_sys3.update_matrices(factor = 3)
        
        correct_new_B_1 = 3*np.ones([8, 6])
        correct_new_B_1[-1] = 24
    
        self.assertTrue((self.cnt_sys1.B==correct_new_B_1).all())
        self.assertTrue((self.cnt_sys2.B==self.B).all())
        self.assertTrue((self.cnt_sys3.B==self.B).all())
        
        LIP = dy.ControlSystem.from_name("J->CCC", 0.1, 3.5, ["_x", "_y"])
        self.assertEqual(LIP.T, 0.1)
        self.assertEqual(LIP.w, 3.5)
        correctA = use.get_system_matrices("J->CCC")[0](0.1, 3.5)
        correctB = use.get_system_matrices("J->CCC")[1](0.1, 3.5)
        
        self.assertTrue((LIP.A==correctA).all())
        self.assertTrue((LIP.B==correctB).all())
        
    def test_extended_control_system(self):
        n = len(self.states)
        m = len(self.inputs)
        ax = len(self.axes)
        self.assertEqual(len(self.ext_sys1.state_ID), n*ax)
        self.assertEqual(len(self.ext_sys2.domain_ID), m*ax+ax)
        self.assertEqual(len(self.ext_sys3.all_variables), n*ax+m*ax+ax)
        self.assertEqual(len(self.ext_sys1.all_variables),
                         len(self.ext_sys1.definitions.keys()))
        
        ## Composed updates:
        
        self.assertEqual(len(self.ext_sys1.matrices), m+1)
        self.assertEqual(self.ext_sys1.matrices[-1].shape, (self.N, n, n))
        self.assertEqual(self.ext_sys1.matrices[0].shape, (self.N, self.N, n))
        
        original_U0 = self.ext_sys1.matrices[0]
        original_S = self.ext_sys1.matrices[-1]
        
        self.ext_sys1.update(control_system = self.cnt_sys1, factor=7)
        self.ext_sys2.update(control_system = self.cnt_sys2, factor=7)
        self.ext_sys3.update(control_system = self.cnt_sys3, factor=7)
        
        self.assertTrue((self.ext_sys1.matrices[0] != original_U0).any())
        self.assertTrue((self.ext_sys2.matrices[0] == original_U0).all())
        self.assertTrue((self.ext_sys3.matrices[0] == original_U0).all())
        
        self.assertTrue((self.ext_sys1.matrices[-1] == original_S).all())
        self.assertTrue((self.ext_sys2.matrices[-1] == original_S).all())
        self.assertTrue((self.ext_sys3.matrices[-1] == original_S).all())
        
        for state, ID in self.ext_sys2.state_ID.items():
            self.assertEqual(int(state[1]), ID)
            
    def test_domain_variable(self):
        n1 = 1; n2 = 1; n3 = 2
        self.assertEqual(len(self.domVar1.domain), n1*len(self.domVar1.axes))
        self.assertEqual(len(self.domVar2.domain), n2*len(self.domVar2.axes))
        self.assertEqual(len(self.domVar3.domain), n3*len(self.domVar3.axes))
        
        self.assertEqual(len(self.domVar1.definitions),len(self.domVar1.domain))
        self.assertEqual(len(self.domVar2.definitions),len(self.domVar2.domain))
        self.assertEqual(len(self.domVar3.definitions),len(self.domVar3.domain))
        
        # updates
        
        original_dom1 = self.domVar1.domain.copy()
        original_dom2 = self.domVar2.domain.copy()
        original_dom3 = self.domVar3.domain.copy()
        
        self.domVar1.update(new_sizes = 10)
        self.domVar2.update(new_sizes = 10)
        self.domVar3.update(new_sizes = 10)
        
        self.assertNotEqual(self.domVar1.domain, original_dom1)
        self.assertEqual(self.domVar2.domain, original_dom2)
        self.assertEqual(self.domVar3.domain, original_dom3)
        
if __name__ == "__main__":
    unittest.main()
    
    # FOR MANUAL TEST
    o = DynamicsTestCase()
    o.setUp()