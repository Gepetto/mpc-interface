#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 20:13:46 2022

@author: nvilla
"""
import numpy as np
from behaviors import ControlSystem, ExtendedSystem
from body import Formulation
import tools as use

if __name__ == "__main__":
    
    # configurando pasos
    
    steps = ExtendedSystem(["Ds"], ["s"], "s", 
                            np.ones([9, 1, 1]),
                            [(np.arange(9).reshape([-1,1])>=[2, 5, 8]).astype(int)[:, :, None]],
                            ["_x", "_y"], how_to_update_matrices=use.update_step_matrices, 
                            time_variant=True)
    
    LIP = ControlSystem.from_name("J->CCC", 0.1, 3.5, ["_x", "_y"])
    
    extSys = ExtendedSystem.extend_cotrol_system(LIP, "x", 9)
    
    
    syst = Formulation()
    syst.incorporate_dynamics("steps", steps)
    syst.incorporate_dynamics("LIP", extSys)
    
    # Ranges
    syst.identify_qp_domain(["x0_x", "x0_y", "Ds_x", "Ds_y", 'CoM_dddot_x', 'CoM_dddot_y'])
    syst.update_qp_sizes()
    syst.update_qp_IDs()
    
    # MAtrices
    # Se hace necesario el parametro N de las dinamicas
    Mg1, Mo1 = syst.get_matrices_from_behavior("s0_x")
    syst.make_preview_matrices()
    Mg2, Mo2 = syst.get_matrices_from_definition("s_x")
    
    syst.update(count = 0, regular_time = 2)