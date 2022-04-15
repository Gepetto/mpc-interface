#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:16:46 2022

@author: nvilla
"""

import numpy as np

system = "J->CCC"

# timing 

num_steps = 2
mpc_period = 0.1  # [s] nominal sampling period

ss_duration = round(11*mpc_period, 5)
ds_duration = mpc_period

regular_step_duration = round(ss_duration + ds_duration, 5)  # number of nominal periods for step
step_samples = int(round(regular_step_duration / mpc_period ))
horizon_lenght = num_steps * step_samples  # number of iterations in preview horizon


# geometry

feet_sep = 0.08  # [m] minimum separation between feet
foot_corner = np.array([0.1, 0.05])
stepping_corner = np.array([0.3, 0.1])
stepping_center = np.array([0, 2 * foot_corner[1] + stepping_corner[1] + feet_sep])

strt_y = feet_sep/2 + foot_corner[1]
com_height = 0.877  # [m] CoM height

# dynamics
gravity = [0, 0, -9.81]
omega = np.sqrt(-gravity[2] / com_height)

# goal 
cop_safety_margin = 0.02

target_vel = np.array([0.6, 0, 0])

cost_weights = {"minimize jerk": 0.001,
                "track velocity": 0.01,
                "relax ankles": 1,
                "terminal":0}