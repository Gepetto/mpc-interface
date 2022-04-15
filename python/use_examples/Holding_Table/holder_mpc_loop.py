#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:36:32 2022

@author: nvilla
"""

import numpy as np
import matplotlib.pyplot as plt
import biped_configuration as config
from holder_formulation import formulate_biped
#from self_motivated_steps import formulate_biped ## TODO: MAke a new mpc_loop experimental

from qpsolvers import osqp_solve_qp
import scipy.sparse as sy

class Controller:
    def __init__(self, conf):
        self.N = conf.horizon_lenght
        self.axes = ["_x", "_y"]
        self.form = formulate_biped(conf)
        self.optim = np.zeros([self.form.optim_len, 1])
        self.given = np.zeros([self.form.given_len, 1])
        
        self.motion = {variable:self.form.preview(self.given, self.optim, variable)
                       for variable in self.form.definitions.keys()}
        ### motion initialization:
        self.motion["x0_y"][0] = conf.strt_y
        self.motion["s0_y"][0] = conf.strt_y
        
        ### additional fields:
        self.step_count = 0
        self.n = conf.step_samples
        self.motion["step_times"] = np.array([
                (i+1)*self.n-1 for i in range(self.form.domain["Ds_x"])
                ])
        
    def count_steps(self):
        self.motion["step_times"] -= 1
        if self.motion["step_times"][0] == -1:
            self.motion["step_times"] += self.n
            self.step_count += 1 
        
    def current_state(self):
        return np.hstack([self.motion["x0"+axis] for axis in self.axes])
        
    def decide_actions(self, time):
        self.form.update(step_times = self.motion["step_times"],
                         step_count = self.step_count)
        self.given = self.form.arrange_given(self.motion)

        A, h, Q, q = self.form.generate_all_qp_matrices(self.given)
        Q = sy.csc_matrix(Q)
        A = sy.csc_matrix(A)
        
        self.optim = osqp_solve_qp(P=Q, q=q, G=A, h=h).reshape([-1, 1])

    def preview_all(self):
        
        for variable in self.form.definitions:
            self.motion[variable] = self.form.preview(self.given, 
                                                      self.optim, variable)
        
    def plot_horizon(self, time, axis):
        
        times = time + np.arange(0, self.N)
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(times, self.motion["CoM"+axis])
        ax.plot(times, self.motion["b"+axis])
        ax.plot(times, self.motion["DCM"+axis])
        ax.plot(times, self.motion["s"+axis], "+")
        
    def update_given_collector(self):
        for state, ID in self.form.dynamics["LIP"].state_ID.items():
            axis = state[-2:]
            self.motion["x0"+axis][ID] = self.motion[state][0]
            
        for state, ID in self.form.dynamics["steps"].state_ID.items():
            axis = state[-2:]
            self.motion["s0"+axis][ID] = self.motion[state][1]
            
        "In this simple example, we keep the bias equal to zero"
        for variable, size in self.form.dynamics["bias"].domain.items():
            o.motion[variable] = np.zeros([size, 1])
            
    def save_motion(time, storager):pass
        

if __name__ == "__main__":
    o = Controller(config)
    
    for time in range(100):
        
        o.decide_actions(time)
        o.preview_all()  # o.preview_horizon()
        o.plot_horizon(time, "_y")
        
        ########### ~~~ Change of time ~~~ ###############
        o.update_given_collector()
        o.count_steps()
        
        
     
        
