#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:09:29 2022

@author: nvilla
"""


import unittest

import numpy as np

from gecko.restrictions import Constraint, Box

class RestrictionsTestCase(unittest.TestCase):
    def setUp(self):
        
        arrowSet = [4,7]
        
        limit1 = Constraint(variable="CoM", extreme=-10)
        limit2 = Constraint(variable="CoM", extreme=-10, axes=["_x"])
        limit3 = Constraint(variable="CoM", extreme=-10, axes=["_x", "_y"],
                            arrow=arrowSet, center=[1,1])
        limit4 = Constraint(variable="CoM", extreme=-10, axes=["_x", "_y"],
                            arrow=arrowSet, L = np.ones([5, 11]))
        limit5 = Constraint(variable="CoM", extreme=-10, axes=["_x", "_y"],
                            arrow=arrowSet, L = np.ones([5, 4]), 
                            schedule=range(3,7))
        limit6 = Constraint(variable="CoM", extreme=-10, axes=["_x", "_y"],
                            arrow=arrowSet, schedule=range(3,7))
        
        self.limit1=limit1;self.limit2=limit2;self.limit3=limit3
        self.limit4=limit4;self.limit5=limit5;self.limit6=limit6
        self.arrowSet = arrowSet
        
    def test_constructor(self):
        
        with self.assertRaises(AssertionError):
            Constraint(variable="CoM",  extreme=10, axes=["_x", "_y"])
        with self.assertRaises(ValueError):
            Constraint(variable="CoM", extreme=10, axes=["_x", "_y"], arrow=5)
            
        self.assertTrue(self.limit1.axes == [""])
        self.assertTrue(self.limit2.axes == ["_x"])
        self.assertTrue(self.limit3.axes == ["_x", "_y"])
        
        self.assertTrue(self.limit1.arrow == -1)
        self.assertTrue(self.limit2.arrow == -1)
        self.assertTrue((self.limit3.arrow == -np.array(self.arrowSet)).all())
        
        self.assertTrue(isinstance(self.limit4.L, list))
        self.assertTrue(len(self.limit5.L) == len(self.limit5.axes))
        
        self.assertTrue(isinstance(self.limit1.matrices, list))
        
        self.assertTrue(len(self.limit1.matrices) == 1)
        self.assertTrue(len(self.limit2.matrices) == 1)
        self.assertTrue(len(self.limit3.matrices) == 2)
        self.assertTrue(len(self.limit4.matrices) == 2)
        self.assertTrue((self.limit5.matrices[0] == 
                        self.limit5.L[0]*self.limit5.arrow[0, 0]).all())
        
        self.assertTrue(self.limit1.nlines is None)
        self.assertTrue(self.limit2.nlines is None)
        self.assertTrue(self.limit3.nlines is None)
        self.assertTrue(self.limit4.nlines == self.limit4.L[0].shape[0])
        self.assertTrue(self.limit5.nlines == self.limit5.L[0].shape[0])
        self.assertTrue(self.limit6.nlines == self.limit6.schedule.stop - 
                                              self.limit6.schedule.start)
        
    def test_bound(self):
        
        self.assertEqual(self.limit1.bound(), self.limit1.extreme)
        self.assertEqual(self.limit2.bound(), self.limit2.extreme)
        self.assertEqual(self.limit3.bound(), -1)
        self.assertEqual(self.limit5.bound(), self.limit5.extreme)
        
    def test_update(self):
        
        self.limit1.update(extreme = [3.9, 2]) # as a result arrow with 2 rows
        self.assertEqual(self.limit1.arrow.shape[0], 2)
        
        self.limit1.update(arrow = 5) # now arrow and extr with 1 row
        self.assertEqual(self.limit1.extreme.shape[0], 1)
        
        self.limit1.update(extreme = [4, 7, 8]) # Now arrow has 3 rows
        self.limit1.update(L=np.zeros([3, 3])) 
        
        self.assertEqual(self.limit1.nlines, 3)
        self.assertEqual(self.limit1.matrices[0].size, self.limit1.L[0].size)
        
        with self.assertRaises(IndexError):
            self.limit1.update(L=[np.zeros([3, 3]), np.eye(3)])
        with self.assertRaises(AssertionError):
            self.limit1.update(L = np.zeros([3, 3]), schedule = range(2))
        
        self.limit1.update(schedule=range(15), L=np.zeros([3, 15]))
        self.assertEqual(self.limit1.nlines, 3)
        self.assertEqual(self.limit1.t, 15)
        self.assertEqual(self.limit1.m, 3)
        self.limit1.update(L=[])
        self.assertEqual(self.limit1.L, [])
        self.assertEqual(self.limit1.m, None)
        self.assertEqual(self.limit1.nlines, self.limit1.t)
        
        self.limit2.update(schedule = range(4))
        self.assertEqual(self.limit2.nlines, 4)
        self.limit2.update(schedule = range(0))
        self.assertEqual(self.limit2.nlines, None)
        
        self.limit4.update(arrow = np.ones([5, 2]))
        self.assertEqual(self.limit4.extreme.shape[0], 5)
        self.limit4.update(L = np.ones([5, 10]))
        
    def test_is_feasible(self):
        
        self.assertTrue(not self.limit1.is_feasible(-20))
        self.assertTrue(self.limit1.is_feasible(9))
        self.assertTrue(self.limit1.is_feasible(-10+1e-6))
        self.assertTrue(not self.limit1.is_feasible(-10-1e-6))
        self.assertTrue(not self.limit1.is_feasible(-10))
        
        points_3 = [np.array([1, 1]), 
                    np.array([1, 0.8]), 
                    np.array([0, (-10+1e-4)/7]),
                    np.array([-1.3, 1])]
        
        self.assertEqual([True, True, False, True], 
                         self.limit3.is_feasible(points_3))
        
        points_4 = [-np.ones([11, 3]), np.zeros([11, 3])]
        
        self.assertTrue(all([[np.zeros([5, 2]).astype(bool), 
                              np.ones([5, 2]).astype(bool)],
                             self.limit4.is_feasible(points_4)]))
        
    def test_box(self):
        
        vertices = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        SSbox = Box.state_space("CoM", vertices, ["_x"])
            
        for limit in SSbox.constraints:
            
            self.assertTrue(np.round(np.linalg.norm(limit.L), 8) == 1)
            self.assertTrue(limit.arrow in [1, -1])
            self.assertTrue(limit.extreme > 0)
            self.assertTrue((limit.center == np.array([0, 0])).all())
        
        SSbox.move_SS_box([1, 2])
        
        true_points = [np.reshape([1e-5, 2], [-1, 1]), 
                       np.reshape([1, 1+1e-5], [-1, 1]),
                       np.reshape([1, 3-1e-5], [-1, 1]),
                       np.reshape([2-1e-5, 2], [-1, 1]),
                       np.reshape([1, 2], [-1, 1])]
        
        false_points = [np.reshape([-1e-5, 2], [-1, 1]), 
                        np.reshape([1, 1-1e-5], [-1, 1]),
                        np.reshape([1, 3+1e-5], [-1, 1]),
                        np.reshape([2+1e-5, 2], [-1, 1]),
                        np.reshape([0, 0], [-1, 1])]
        
        self.assertTrue(all(SSbox.is_feasible(true_points)))
        self.assertTrue(not all(SSbox.is_feasible(false_points)))
        
        for limit in SSbox.constraints:
            self.assertTrue((np.round(np.abs(limit.center), 4) == [2.1213, 0.7071]).any())
        
        SSbox.scale_box(np.sqrt(2))
        for limit in SSbox.constraints:
            self.assertTrue(limit.extreme == 1.)
        
        
if __name__ == "__main__":
    
    unittest.main()
    
    # FOR MANUAL TEST
    o = RestrictionsTestCase()
    o.setUp()