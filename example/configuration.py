#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:58:40 2021

@author: nvilla
"""
import numpy as np

dt = 0.1
ss_duration = 0.7
ds_duration = dt
step_duration = ss_duration + ds_duration
num_steps = 4

Dt = 1e-5 # period for numerical differentiation.

step_samples = int(round((ss_duration + ds_duration) / dt))
horizon_lenght = num_steps * step_samples  # time horizon
gravity = 9.81

xk = np.array([0, 0, 0]).T
yk = np.array([0.1, 0, 0]).T
zk = np.array([0.87, 0, 0]).T

LF_translation = np.array([0., 0.1, 0]).T
RF_translation = np.array([0, -0.1, 0]).T

LF_rotation = np.eye(3)
RF_rotation = np.eye(3)
foot_rising = 0.1

## DATA for BOXES
foot_corner = np.array([0.1, 0.06])
stepping_corner = np.array([0.5, 0.2])
stepping_center = np.array([0, 0.005+2*foot_corner[1]+2*stepping_corner[1]])


w = np.sqrt(gravity/zk[0])
mu = 1  # static friction coefficient

always_horizontal_ground = True

cost_weights = {"minimize jerk": 0.0001,
                "track velocity": 0.01,
                "relax ankles": 0.1}

target_vel = np.array([.2, 0, 0]).T

with_orientation = False

system_name = "J->CCC"

optimization_domain = ["CoM_dddot_x", "Ds_x", "CoM_dddot_y", "Ds_y"]


