#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:27:32 2022

@author: nvilla
"""




import unittest

import numpy as np

from gecko.goal import Cost

class GoalTestCase(unittest.TestCase):
    def setUp(self):
        
        cost1 = Cost("CoM", 8)
        cost2 = Cost("CoM", 8, aim=[8, 0], axes=["_x", "_y"])
        cost3 = Cost("f", 4, cross="u")
        cost4 = Cost("f", 4, cross="u",axes=["_x", "_y"], cross_L=[5, 2])
        
        self.cost1=cost1;self.cost2=cost2;self.cost3=cost3;self.cost4=cost4
        
        
    def test_constructor(self):
        
        self.assertTrue(self.cost1.variable == "CoM")
        self.assertTrue(self.cost1.aim == 0)
        self.assertTrue(self.cost1.aim is self.cost1.cross_aim)
        self.assertTrue(self.cost1.L == [])
        self.assertTrue(self.cost1.L is self.cost1.cross_L)
        
        self.assertTrue(self.cost2.variable == "CoM")
        self.assertTrue((self.cost2.aim == [8, 0]).all())
        self.assertTrue(self.cost2.t is None)
        self.assertTrue(self.cost2.aim is self.cost2.cross_aim)
        
        self.assertTrue(self.cost3.cross=="u")
        self.assertTrue(self.cost3.variable=="f")
        self.assertTrue(self.cost3.cross_L==[])
        self.assertTrue(self.cost3.aim == 0)
        self.assertTrue(self.cost3.cross_aim == 0)
        self.assertTrue(not self.cost3.aim is self.cost3.cross_aim)
        
        self.assertTrue(self.cost4.cross_L == [5, 2])
        
    def test_update(self):
        
        self.cost1.update(aim = 9)
        self.cost1.update(weight = 0)
        self.cost1.update(L = 4*np.eye(3))
        
        self.assertTrue(self.cost1.cross_aim == 9)
        self.assertTrue(self.cost1.cross_L is self.cost1.L)
        
        with self.assertRaises(KeyError):
            self.cost1.update(cross_aim = 3)
            
        self.assertTrue(self.cost3.cross_aim == self.cost3.aim)
        self.cost3.update(cross_aim = 2)
        self.assertTrue(self.cost3.cross_aim != self.cost3.aim)
        
        self.assertTrue(self.cost4.cross_L and not self.cost4.L)
        self.cost4.update(cross_L = [])
        self.assertTrue(not self.cost4.cross_L and not self.cost4.L)
        self.cost4.update(L = [3, 6])
        self.assertTrue(not self.cost4.cross_L and self.cost4.L)
        

if __name__ == "__main__":
    
    unittest.main()
    