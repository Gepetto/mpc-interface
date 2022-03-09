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

import formulation.settings as prop

class GeckoBiped:
    def __init__(self):
        #### Making dynamics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                        ##Steps##
        E = now.plan_steps(0, prop.horizon_lenght, 
                           regular_time=prop.step_samples)[:, :, None]
        F = np.ones([prop.horizon_lenght, 1, 1])
        axes = ["_x", "_y"]
        
        steps = ExtendedSystem(["Ds"], ["s"], "s", S = F, U = [E], axes = axes,
                               how_to_update_matrices=now.update_step_matrices, 
                               time_variant=True)
        
                                        ##LIP##
        LIP = ControlSystem.from_name(prop.system_name, prop.dt, prop.w, axes)
        LIP_ext = ExtendedSystem.from_cotrol_system(LIP, "x",
                                                    prop.horizon_lenght)
        
                                    ##Non-Linearity##
        bias = DomainVariable("n", prop.horizon_lenght, ["_x", "_y"])
        
        ##Definitions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        some_defs = {}
        if prop.system_name == "J->CCC":
            for axis in axes:
                LIP_B = LineCombo({"CoM"+axis:1,
                                   "CoM_ddot"+axis:-1/prop.w**2})
                BpNmS = LineCombo({"b"+axis : 1,
                                   "n"+axis : 1, ##TODO: change it to repeat the last one instead of the first one
                                   "s"+axis : -1 })
                CmS = LineCombo({"CoM"+axis : 1,
                                 "s"+axis   : -1  })
                DCM = LineCombo({"CoM"+axis     : 1,
                                 "CoM_dot"+axis : 1/prop.w})
                DCMmS = LineCombo({"DCM"+axis : 1,
                                   "s"+axis   : -1})
                B0pN0 = LineCombo({
                        "x0"+axis : np.array([1, 0, -1/prop.w**2]),
                        "n"+axis  : np.hstack([1, np.zeros(prop.horizon_lenght-1)])# as n is predicted before, the index 0 corresp to the initial
                        })
                C0 = LineCombo({"x0"+axis : np.array([1, 0, 0])})
                
                some_defs.update({"b"+axis : LIP_B, 
                                  "(b+n-s)"+axis : BpNmS,
                                  "(c-s)"+axis : CmS,
                                  "DCM"+axis : DCM,
                                  "(DCM-s)"+axis : DCMmS,
                                  "(b0+n0)"+axis : B0pN0,
                                  "(c0)"+axis : C0,
                                  })
            
            ##FEEDBACK## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            from cricket import closed_loop_tools as clt
            
            
            T = 0.002 # Check if this is the sampling period for tracking
            A = LIP.get_A(T, LIP.w)
            B = LIP.get_B(T, LIP.w)
            L = np.array([1, 0, -1/LIP.w**2])
            v_act=10
            v_est=np.array([1e-2, 1e-3, 3e-3])
            
            K0 = clt.get_stability_center(A, B),
            K1 = clt.optimal_gain(L, A, B, v_act, v_est, K0=K0,
                                  bound_type="state", method="Nelder-Mead")  
            K = clt.optimal_gain(L, A, B, v_act, v_est, K0=K1,
                                  bound_type="state", method="Powell")  
            self.K = K
            
            G = A+B@K[None, :]
            disturbance = (v_act + np.abs(K) @ np.abs(v_est))/1
            critical_directons = [L,-L]
            Z = clt.approximate_set(G, B, n=1, v=disturbance,
                                    including=critical_directons)
            cop_safety_margin = max(L@Z.T)
            
#            clt.plot_set(Z)
#            print(Z)
#            print(cop_safety_margin)
        ##CONSTRAINTS## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
        initial_constraint_x = Box.state_space("x0", -Z, ["_x"], time_variant=True,
                                               how_to_update=now.recenter_on_real_state_x)
        initial_constraint_y = Box.state_space("x0", -Z, ["_y"], time_variant=True,
                                               how_to_update=now.recenter_on_real_state_y)
        
        support_vertices = now.make_simetric_vertices(prop.foot_corner)
        initial_support = Box.task_space("(b0+n0)", support_vertices, axes,
                                         time_variant=True, 
                                         how_to_update=now.recenter_support)
        
        support_polygon = Box.task_space("(b+n-s)", support_vertices, axes)                        
                                
        stepping_vertices = now.make_simetric_vertices(prop.stepping_corner)
        stepping_area = Box.task_space("Ds", stepping_vertices, axes, 
                                       how_to_update=now.update_stepping_area,
                                       time_variant=True)
        stepping_area.update(step_count=0, n_next_steps=steps.domain["Ds_x"], 
                             xy_lenght = prop.stepping_center)
        
        reachable_vertices = now.make_simetric_vertices(2*prop.stepping_corner) 
        reachable_steps = Box.task_space("s", reachable_vertices, axes,
                                         schedule = range(8,9), 
                                         how_to_update=now.reduce_by_time,
                                         time_variant = True)
        
        terminal_constraint = Box.task_space(
                "(DCM-s)", support_vertices, axes, 
                schedule=range(prop.horizon_lenght-1, prop.horizon_lenght)
                )
        
        support_polygon.set_safety_margin(cop_safety_margin)
        initial_support.set_safety_margin(cop_safety_margin)
        terminal_constraint.set_safety_margin(cop_safety_margin)
        
        ##COSTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        relax_ankles = Cost("(b+n-s)", [0, 0], 
                            prop.cost_weights["relax ankles"], axes = axes)
        minimum_jerk = Cost("CoM_dddot", [0, 0],
                            prop.cost_weights["minimize jerk"], axes = axes)
        track_vel_x = Cost("CoM_dot", prop.target_vel[0],
                            prop.cost_weights["track velocity"], axes = ["_x"])
        track_vel_y = Cost("CoM_dot", prop.target_vel[1],
                           prop.cost_weights["track velocity"]/10, axes = ["_y"])
        
        ##Formulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        form = Formulation()
        form.incorporate_dynamics("steps", steps)
        form.incorporate_dynamics("LIP", LIP_ext)
        form.incorporate_dynamics("bias", bias)
        form.incorporate_definitions(some_defs)
        form.incorporate_goal("relax ankles", relax_ankles)
        form.incorporate_goal("minimize jerk", minimum_jerk)
        form.incorporate_goal("track vel_x", track_vel_x)
        form.incorporate_goal("track vel_y", track_vel_y)
        form.incorporate_box("stepping area", stepping_area)
        form.incorporate_box("support_polygon", support_polygon)
        form.incorporate_box("reachable steps", reachable_steps)
        form.incorporate_box("terminal_Constraint", terminal_constraint)
        form.incorporate_box("initial_Constraint_x", initial_constraint_x)
        form.incorporate_box("initial_Constraint_y", initial_constraint_y)
        form.incorporate_box("initial_support", initial_support)
        form.identify_qp_domain(prop.optimization_domain)
        form.make_preview_matrices()
        
        self.form = form
        
    def update(self, count, step_count, current_swing_pose, current_ss_time, x0, old_s0):
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
                stepping_center : array.
                
            for reachable steps:
                current_swing_pose: current position of the swing foot 
                                    (an homogeneous matrix from mpc_formul)
                current_ss_time: time spent in the current single support.
        """
        
        step_dynamics_keys = dict(count=count,
                                  N=prop.horizon_lenght, 
                                  regular_time=prop.step_samples)
        self.form.dynamics["steps"].update(**step_dynamics_keys)
        
        stepping_constraint_keys = dict(
                step_count=step_count,
                n_next_steps=self.form.dynamics["steps"].domain["Ds_x"],
                xy_lenght=prop.stepping_center)
        self.form.constraint_boxes["stepping area"].update(**stepping_constraint_keys)
        
        reachable_steps_keys = dict(current_swing_pose=current_swing_pose,
                                    current_ss_time=current_ss_time, 
                                    step_duration=prop.step_duration,
                                    landing_advance=prop.landing_advance)
        self.form.constraint_boxes["reachable steps"].update(**reachable_steps_keys)
        
        x0_x = dict(x0_x=x0[:, 0])
        x0_y = dict(x0_y=x0[:, 1])
        self.form.constraint_boxes["initial_Constraint_x"].update(**x0_x)
        self.form.constraint_boxes["initial_Constraint_y"].update(**x0_y)
        
        self.form.constraint_boxes["initial_support"].update(s0 = old_s0)
        
        self.form.update()
        
    def make_the_matrices(self, given_collector):
        
        given = self.form.arrange_given(given_collector)
        
        A, h = self.form.generate_all_qp_constraints(given)
        Q, q = self.form.generate_all_qp_costs(given)
        
        return A, h, Q, q
    
def print_initial_constraint(box):
    
    print("\nin following we show the values for \n"+
          "the initial constraint in each axis \n"+
          "with the form:\n\n"+
          "arrow * (x0 - x0_measured) < extreme\n\n")
    
    print("arrows:")
    for limit in box.constraints:
        print(str(limit.L[0][0])+", "+str(limit.L[0][1])+", "+str(limit.L[0][2]))
    
    print("\nextremes:")
    for i, limit in enumerate(box.constraints):
        print(limit.extreme[0][0])
    
if __name__ == "__main__":
    
    geek = GeckoBiped() 
    
    #### all the following can be removed and is included here just to check that
#    the class do the computations. It is not a proper test:
    
    class hom:
        def __init__(self):
            self.translation = np.array([1, 1, 1])
        def get_translation(self):
            return self.translation
    
    swing = hom()
    
    count = 0
    count_of_steps = 0
    ss_time = 0
    
    given_collector = {var:np.array(rang).reshape([-1, 1]) 
                       for var, rang in geek.form.given_ID.items()}
    
    geek.update(count, count_of_steps, swing, ss_time*prop.dt)
    
    A, h, Q, q = geek.make_the_matrices(given_collector)
