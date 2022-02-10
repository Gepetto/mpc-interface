#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:08:02 2022

@author: nvilla
"""
import numpy as np

from gecko.dynamics import ControlSystem, ExtendedSystem, DomainVariable
from gecko.body import Formulation
import gecko.tools as now
from gecko.restrictions import Box
from gecko.goal import Cost
from gecko.combinations import LineCombo

import configuration as conf


class GeckoTalos:
    def __init__(self):
        #### Making dynamics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ##Steps##
        E = now.plan_steps(0, conf.horizon_lenght, 
                           regular_time=conf.step_samples)[:, :, None]
        F = np.ones([conf.horizon_lenght, 1, 1])
        axes = ["_x", "_y"]
        
        steps = ExtendedSystem(["Ds"], ["s"], "s", S = F, U = [E], axes = axes,
                               how_to_update_matrices=now.update_step_matrices, 
                               time_variant=True)
        
                                        ##LIP##
        LIP = ControlSystem.from_name(conf.system_name, conf.dt, conf.w, axes)
        LIP_ext = ExtendedSystem.from_cotrol_system(LIP, "x",
                                                    conf.horizon_lenght)
        
                                    ##Non-Linearity##
        bias = DomainVariable("n", conf.horizon_lenght, ["_x", "_y"])
        
        ##Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        some_defs = {}
        if conf.system_name == "J->CCC":
            for axis in axes:
                LIP_B = LineCombo({"CoM"+axis:1,
                                   "CoM_ddot"+axis:1/conf.w**2})
                BpNmS = LineCombo({"b"+axis : 1,
                                   "n"+axis : 1,
                                   "s"+axis : -1 })
                CmS = LineCombo({"CoM"+axis : 1,
                                 "s"+axis   : -1  })
                DCM = LineCombo({"CoM"+axis     : 1,
                                 "CoM_dot"+axis : 1/conf.w})
                DCMmS = LineCombo({"DCM"+axis : 1,
                                   "s"+axis   : -1})
                B0pN0mS0 = LineCombo({
                        "x0"+axis : np.array([1, 0, 1/conf.w**2]),
                        "n"+axis  : np.hstack([1, np.zeros(conf.horizon_lenght-1)]),# as n is predicted before, the index 0 corresp to the initial
                        "s0"+axis : [-1]
                        })
                C0mS0 = LineCombo({"x0"+axis : np.array([1, 0, 0]),
                                   "s0"+axis : [-1]})
#                s1_swing = LineCombo({"s0"+axis : 1,
#                                      "Ds"+axis : [1, 0, 0, 0]})
                
                some_defs.update({"b"+axis : LIP_B, 
                                  "(b+n-s)"+axis : BpNmS,
                                  "(c-s)"+axis : CmS,
                                  "DCM"+axis : DCM,
                                  "(DCM-s)"+axis : DCMmS,
                                  "(b0+n0-s0)"+axis : B0pN0mS0,
                                  "(c0-s0)"+axis : C0mS0,
#                                  "(s0+Ds)"+axis : s0pDs
                                  })
    
        ##COSTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        relax_ankles = Cost("(b+n-s)", [0, 0], 
                            conf.cost_weights["relax ankles"], axes = axes)
        minimum_jerk = Cost("CoM_dddot", [0, 0],
                            conf.cost_weights["minimize jerk"], axes = axes)
        track_velocity = Cost("CoM_dot", [0.2, 0],
                              conf.cost_weights["track velocity"], axes = axes)
        
                                ##CONSTRAINTS##
        support_vertices = now.make_simetric_vertices(conf.foot_corner)
        support_polygon = Box.task_space("(b+n-s)", support_vertices, axes)                        
                                
        stepping_vertices = now.make_simetric_vertices(conf.stepping_corner)
        stepping_area = Box.task_space("Ds", stepping_vertices, axes, 
                                       how_to_update=now.update_stepping_area,
                                       time_variant=True)
        stepping_area.update(step_count=0, n_next_steps=steps.domain["Ds_x"], 
                             xy_lenght = conf.stepping_center)
        
        reachable_vertices = now.make_simetric_vertices(2*conf.stepping_corner) 
        reachable_steps = Box.task_space("s", reachable_vertices, axes,
                                         schedule = range(8,9), 
                                         how_to_update=now.reduce_by_time,
                                         time_variant = True)
        
        ##Formulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        form = Formulation()
        form.incorporate_dynamics("steps", steps)
        form.incorporate_dynamics("LIP", LIP_ext)
        form.incorporate_dynamics("bias", bias)
        form.incorporate_definitions(some_defs)
        form.incorporate_goal("relax ankles", relax_ankles)
        form.incorporate_goal("minimize jerk", minimum_jerk)
        form.incorporate_goal("track velocity", track_velocity)
        form.incorporate_box("stepping area", stepping_area)
        form.incorporate_box("support_polygon", support_polygon)
        form.incorporate_box("reachable steps", reachable_steps)
        form.identify_qp_domain(conf.optimization_domain)
        form.make_preview_matrices()
        
        self.form = form
        
    def update(self, count, step_count, current_swing_pose, current_ss_time):
        """
        The current implementation requires the following arguments to update 
        the formulation:
            
            for the dynamic of stepes:
                count: current iteration number (discretized time) 
                horizon_lenght: number of samples of the horizon.
                regular_time: number of samples between steps 
                
            for the stepping constraint:
                step_count: current count of steps.
                n_next_steps: number of previewed steps
                stepping_corner : array containing (+) half-lenghts in x and y.
                
            for reachable steps:
                current_swing_pose: current position of the swing foot 
                                    (an homogeneous matrix from mpc_formul)
                current_ss_time: time spent in the current single support.
        """
        
        step_dynamics_keys = dict(count=count,
                                  N=conf.horizon_lenght, 
                                  regular_time=conf.step_samples)
        self.form.dynamics["steps"].update(**step_dynamics_keys)
        
        stepping_constraint_keys = dict(
                step_count=step_count,
                n_next_steps=self.form.dynamics["steps"].domain["Ds_x"],
                xy_lenght=conf.stepping_corner)
        self.form.constraint_boxes["stepping area"].update(**stepping_constraint_keys)
        
        reachable_steps_keys = dict(current_swing_pose=current_swing_pose,
                                    current_ss_time=current_ss_time, 
                                    step_duration=conf.step_duration)
        self.form.constraint_boxes["reachable steps"].update(**reachable_steps_keys)
        
        self.form.update()
        
    def make_the_matrices(self, given_collector):
        
        given = self.form.arrange_given(given_collector)
        
        A, h = self.form.generate_all_qp_constraints(given)
        Q, q = self.form.generate_all_qp_costs(given)
        
        return A, h, Q, q
    

if __name__ == "__main__":
    
    geek = GeckoTalos() 

