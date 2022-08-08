#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 16:08:02 2022

@author: nvilla
"""
import numpy as np

from mpc_core.dynamics import ControlSystem, ExtendedSystem, DomainVariable
from mpc_core.body import Formulation
import mpc_core.tools as now
from mpc_core.restrictions import Box
from mpc_core.goal import Cost
from mpc_core.combinations import LineCombo
from cricket import closed_loop_tools as clt

import cricket.talos_conf as config

def formulate_biped(conf):
    
    w = conf.omega
    horizon_lenght = conf.horizon_lenght
    step_samples = conf.step_samples
    self_conscious = conf.closed_loop_MPC
    system = conf.system
    #### DYNAMICS AND DOMAIN VARIABLES ~~~~~~~~~~~~~~~~~~~~
    axes = ["_x", "_y"]
                                  ##~Steps~##
    E = now.plan_steps(0, horizon_lenght, 
                       regular_time=step_samples)
    F = np.ones([horizon_lenght, 1])
    
    steps = ExtendedSystem("Ds", "s", "s", S = F, U = E, axes = axes,
                           how_to_update_matrices=now.update_step_matrices, 
                           time_variant=True)
    steps.define_output("stamps", {"s0":1, "Ds":1},
                        time_variant=True, how_to_update=now.adapt_size)

                                  ##~LIP~##
    LIP = ControlSystem.from_name(system, axes, 
                                  tau=conf.mpc_period, omega=w)
    LIP_ext = ExtendedSystem.from_cotrol_system(LIP, "x",
                                                horizon_lenght)
    
                            ##~Non-Linearity~##
    bias = DomainVariable("n", horizon_lenght, axes)
    
    ##EXTRA DEFINITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_coeff = np.diag(np.ones([horizon_lenght-1]), 1)
    
    some_defs = {}
    if system == "J->CCC":
        for axis in axes:
            LIP_B = LineCombo({"CoM"+axis:1,
                               "CoM_ddot"+axis:-1/w**2})
            BpNmS = LineCombo({"b"+axis : 1,
                               "n"+axis : n_coeff, 
                               "s"+axis : -1 })
            CmS = LineCombo({"CoM"+axis : 1,
                             "s"+axis   : -1  })
            DCM = LineCombo({"CoM"+axis     : 1,
                             "CoM_dot"+axis : 1/w})
            DCMmS = LineCombo({"DCM"+axis : 1,
                               "s"+axis   : -1})
            B0pN0 = LineCombo({
                    "x0"+axis : np.array([1, 0, -1/w**2]),
                    "n"+axis  : np.hstack([1, np.zeros(horizon_lenght-1)])# as n is predicted before, the index 0 corresp to the initial
                    })
            C0 = LineCombo({"x0"+axis : np.array([1, 0, 0])})
            
            some_defs.update({"b"+axis : LIP_B, 
                              "(b+n-s)"+axis : BpNmS,
                              "(c-s)"+axis : CmS,
                              "DCM"+axis : DCM,
                              "(DCM-s)"+axis : DCMmS,
                              "(b0+n0)"+axis : B0pN0,
                              "c0"+axis : C0,
                              })
        
        ##FEEDBACK ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        T = conf.tracking_period 
        A = LIP.get_A(tau = T, omega = w)
        B = LIP.get_B(tau = T, omega = w)
        L = np.array([1, 0, -1/w**2])
        K = conf.centGain[0]
        
        G = A+B@K[None, :]
        
        cop_safety_margin = conf.cop_safety_margin
        
        critical_directions = [L, -L]
        Z_uni = clt.approximate_set(G, B, n=1, v=1,
                                    including=critical_directions)
        v_max = cop_safety_margin/np.max(L@Z_uni.T)
        Z = v_max * Z_uni
        
#            clt.plot_set(Z)
#            print(Z)
        print("\n\nMaximum admissible disturbance: ", v_max)
        
    ##CONSTRAINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    support_vertices = now.make_simetric_vertices(conf.foot_corner)
    
    if self_conscious:
        initial_constraint_x = Box.state_space(
                "x0", -Z, ["_x"], time_variant=True,
                how_to_update=now.recenter_on_real_state_x
                )
        initial_constraint_y = Box.state_space(
                "x0", -Z, ["_y"], time_variant=True,
                how_to_update=now.recenter_on_real_state_y
                )
        initial_support = Box.task_space(
                "(b0+n0)", support_vertices, axes, 
                time_variant=True, how_to_update=now.recenter_support
                )
        initial_support.set_safety_margin(cop_safety_margin)
        
    support_polygon = Box.task_space("(b+n-s)", support_vertices, axes)                        
                            
    stepping_vertices = now.make_simetric_vertices(conf.stepping_corner)
    stepping_area = Box.task_space("Ds", stepping_vertices, axes, 
                                   how_to_update=now.update_stepping_area,
                                   time_variant=True)
    stepping_area.update(step_count=0, n_next_steps=steps.domain["Ds_x"], 
                         xy_lenght = conf.stepping_center)
    
    reachable_vertices = now.make_simetric_vertices(2*conf.stepping_corner) 
    reachable_steps = Box.task_space("stamps", reachable_vertices, axes,
                                     how_to_update=now.reduce_by_time,
                                     time_variant = True, schedule=range(1, 2))
    
    terminal_constraint = Box.task_space(
            "(DCM-s)", support_vertices, axes, 
            schedule=range(horizon_lenght-1, horizon_lenght)
            )
    
    support_polygon.set_safety_margin(cop_safety_margin)
    terminal_constraint.set_safety_margin(cop_safety_margin)
    
    ##COSTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    relax_ankles_x = Cost("(b+n-s)", [0], 
                        conf.cost_weights["relax ankles"], axes = ["_x"])
    relax_ankles_y = Cost("(b+n-s)", [0], 
                        conf.cost_weights["relax ankles"], axes = ["_y"])
    minimum_jerk = Cost("CoM_dddot", [0, 0],
                        conf.cost_weights["minimize jerk"], axes = axes)
    track_vel_x = Cost("CoM_dot", conf.target_vel[0],
                        conf.cost_weights["track velocity"], axes = ["_x"])
    track_vel_y = Cost("CoM_dot", conf.target_vel[1],
                       conf.cost_weights["track velocity"], axes = ["_y"])
    
    ##Formulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    form = Formulation()
    form.incorporate_dynamics("steps", steps)
    form.incorporate_dynamics("LIP", LIP_ext)
    form.incorporate_dynamics("bias", bias)
    form.incorporate_definitions(some_defs)
    form.incorporate_goal("relax ankles x", relax_ankles_x)
    form.incorporate_goal("relax ankles y", relax_ankles_y)
    form.incorporate_goal("minimize jerk", minimum_jerk)
    form.incorporate_goal("track vel_x", track_vel_x)
    form.incorporate_goal("track vel_y", track_vel_y)
    form.incorporate_box("stepping area", stepping_area)
    form.incorporate_box("support_polygon", support_polygon)
    form.incorporate_box("reachable steps", reachable_steps)
    if conf.with_terminal_constraint:
        form.incorporate_box("terminal_Constraint", terminal_constraint)
    
    if self_conscious:
        form.incorporate_box("initial_Constraint_x", initial_constraint_x)
        form.incorporate_box("initial_Constraint_y", initial_constraint_y)
        form.incorporate_box("initial_support", initial_support)
    
    form.identify_qp_domain(conf.optimization_domain)
    form.make_preview_matrices()
    
    def update_this_formulation(body, **kargs):
        """
        The current implementation requires the following arguments to update 
        the formulation:
            
            ### for the dynamic of stepes: ###
                Arguments: 
                    count: current iteration number (count of mpc periods) 
                    
                Parameters:
                    horizon_lenght: number of samples of the horizon.
                    regular_time: number of samples between steps. 
                
            ### for the stepping constraint: ###
                Arguments:
                    step_count: current count of steps.
                
                Parameters:
                    n_next_steps: number of previewed steps
                    stepping_center : array from each step place to the center 
                                      of next stepping area.
                
            ### for reachable steps: ###
                Arguments:
                    current_swing_pose: current position of the swing foot 
                                        (an homogeneous matrix from mpc_formul)
                    current_ss_time: time spent in the current single support.
                    
            ### for initial constraints: ###
                Arguments:
                    x0: array witht the initial state as [x0_x, x0_y]
                    
            ### for initial support: ###
                Arguments:
                    s0_old: the last used support place.
            
        """
        step_dynamics_keys = dict(count=kargs["count"], 
                                  N=horizon_lenght, 
                                  regular_time=step_samples)
        body.dynamics["steps"].update(**step_dynamics_keys)
        
        stepping_constraint_keys = dict(
                step_count=kargs["step_count"],
                n_next_steps = body.dynamics["steps"].domain["Ds_x"],
                xy_lenght=conf.stepping_center)
        body.constraint_boxes["stepping area"].update(**stepping_constraint_keys)
        
        reachable_steps_keys = dict(current_swing_pose=kargs["current_swing_pose"],
                                    current_ss_time=kargs["current_ss_time"], 
                                    step_duration=conf.regular_step_duration,
                                    landing_advance=conf.landing_advance)
        body.constraint_boxes["reachable steps"].update(**reachable_steps_keys)
        
        if self_conscious:
            x0_x = dict(x0_x=kargs["x0"][:, 0])
            x0_y = dict(x0_y=kargs["x0"][:, 1])
            body.constraint_boxes["initial_Constraint_x"].update(**x0_x)
            body.constraint_boxes["initial_Constraint_y"].update(**x0_y)
            body.constraint_boxes["initial_support"].update(s0 = kargs["old_s0"])
        
    form.set_updating_rule(update_this_formulation)
    
    return form

if __name__ == "__main__":
    
    bip_form = formulate_biped(config)