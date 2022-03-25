#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:45:36 2022

@author: nvilla
"""
import numpy as np
import sympy as sy
#from cricket import closed_loop_tools as clt

def extend_matrices(N, A, B):
    n, m = B.shape

    s = np.zeros([n * N, n])
    u = np.zeros([n * N, N, m])

    s[0:n, :] = A
    u[0:n, 0, :] = B

    for i in range(1, N):
        u[n * i : n * (i + 1), 0:i, :] = np.dstack(
            [A.dot(u[n * (i - 1) : n * i, 0:i, j]) for j in range(m)]
        )
        u[n * i : n * (i + 1), i, :] = B

        s[n * i : n * (i + 1), :] = A.dot(s[n * (i - 1) : n * i, :])

    S = np.dstack([s[i : n * N : n, :] for i in range(n)])
    U = [np.dstack([u[i : n * N : n, :, j] for i in range(n)]) for j in range(m)]
    return S, U

def update_step_matrices(extSyst, **kargs):
    """ This function needs
        
        count : int, representing the current time sample number
        
        and one of the following:
        step_times : ndarray or list with next step times.
        
        or 
        
        regular_time : int, to produce steps regularly
        
    """
    N = extSyst.matrices[-1].shape[0]
    if "count" in kargs.keys(): 
        count = kargs["count"]
    else:
        count = 0
    
    if "step_times" in kargs.keys():
        step_times = kargs["step_times"]
        regular_time = None
        
    elif "regular_time" in kargs.keys():
        regular_time = kargs["regular_time"]
        step_times = None
        
        
    else:
        raise KeyError("This funtion needs either 'step_times' or "+
                       "'regular_time', but the kargs "+
                       "introduced are {}".format(kargs.keys()))
        
    U = plan_steps(N, count, step_times, regular_time)
    extSyst.matrices[0] = U[:, :, None]
    
    ### TODO: add if count is None --> count=0 to take step_times that are refered to the present.
def plan_steps(N, count=0, step_times=None, regular_time=None):
    
    preview_times = count + np.arange(N)
    
    if step_times is not None:
        next_steps = step_times[(step_times >= count) * 
                                (step_times < count+N-1)]

    elif regular_time is not None:
        next_steps = np.array([time for time in preview_times
                               if not (time+2)%regular_time
                               and time < count+N-1])
    else:
        msg = "either the step_times or some "+\
        "regular_time for steps must be provided"
        raise AssertionError(msg)
        
    E = (preview_times.reshape([N, 1]) > next_steps).astype(int)
    return E

def update_stepping_area(box, **kargs):
    next_centers = find_step_centers(kargs["step_count"],
                                     kargs["n_next_steps"],
                                     kargs["xy_lenght"])
    
    for limit in box.constraints:
        limit.update(center = next_centers)
                
def find_step_centers(step_count, n_next_steps, xy_lenght):
    side = (-1)**(step_count+1) 
    alterned = np.tile([[1], [-1]],
                       [n_next_steps//2+1, 1])[:n_next_steps]
    
    centers = np.hstack([np.ones([n_next_steps, 1])*xy_lenght[0],
                         side * alterned * xy_lenght[1]])
    return centers

def reduce_by_time(box, **kargs):
    current_swing_pose = kargs["current_swing_pose"]
    current_ss_time = kargs["current_ss_time"]
    step_duration = kargs["step_duration"]
    landing_advance = kargs["landing_advance"]
    
    new_center = current_swing_pose.get_translation()[:2]
    box.recenter_in_TS(new_center)
    
    s = (step_duration-landing_advance-current_ss_time)/(step_duration-landing_advance)
    scale = s if s > 0 else 1e-2 
    
    box.scale_box(scale)
    
def recenter_on_real_state_x(box, **kargs):
    box.move_SS_box(new_center = kargs["x0_x"])
            
def recenter_on_real_state_y(box, **kargs):
    box.move_SS_box(new_center = kargs["x0_y"])
    
def recenter_support(box, **kargs):
    box.move_TS_box(new_center = kargs["s0"])
    
def make_simetric_vertices(xy_corner):
    x_values = np.array([1, -1, -1, 1])[:, None] * xy_corner[0]
    y_values = np.array([1, 1, -1, -1])[:, None] * xy_corner[1]
    
    return np.hstack([x_values, y_values])

def adapt_size(stamps, **kargs):
    dynamics = kargs["extSyst"]
    axis = stamps.variables[0][2:]
    
    Ds_name = "Ds" + axis; x0_name = "s0" + axis
    Ds_coeff_ID = stamps.variables.index(Ds_name)
    s0_coeff_ID = stamps.variables.index(x0_name)
    size = dynamics.domain[Ds_name]
    
    stamps.matrices[Ds_coeff_ID] = np.tril(np.ones([size+1, size]), -1)
    stamps.matrices[s0_coeff_ID] = np.ones([size+1, 1])
              
def get_system_matrices(system):
    """

    Parameters
    ----------
    system : TYPE str
        DESCRIPTION. string containing the name of a valid system formulation

    This method returns the matrices A and B as functions of adequate
    parameters depending on the control 'system' chosen to model the robot
    dynamics.
    """

    T_, w_ = sy.symbols("tau omega")

    if system == "P->CC":
        A = sy.Matrix([[0, 1], [w_ ** 2, 0]])
        B = sy.Matrix([0, -(w_ ** 2)])
        parameters = ("tau=None", "omega=None", "**kwargs")

    elif system == "P->X":
        A = sy.Matrix([w_])
        B = sy.Matrix([-w_])
        parameters = ("tau=None", "omega=None", "**kwargs")

    elif system == "dP->CCC":
        A = sy.Matrix([[0, 1, 0], [0, 0, 1], [0, w_ ** 2, 0]])
        B = sy.Matrix([0, 0, -(w_ ** 2)])
        parameters = ("tau=None", "omega=None", "**kwargs")

    elif system == "dP->CCP":
        A = sy.Matrix([[0, 1, 0], [w_ ** 2, 0, -(w_ ** 2)], [0, 0, 0]])
        B = sy.Matrix([0, 0, 1])
        parameters = ("tau=None", "omega=None", "**kwargs")

    elif system == "J->CCC":
        A = sy.Matrix([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        B = sy.Matrix([0, 0, 1])
        parameters = ("tau=None", "**kwargs")

    A_, B_ = sy.simplify(discretize(A, B))

    get_matrix_A = sy.lambdify(parameters, A_, "numpy")
    get_matrix_B = sy.lambdify(parameters, B_, "numpy")

    return get_matrix_A, get_matrix_B, parameters

def get_system_variables(system):
    """
    indicates the state and input variables for a given system formulation

    Parameters
    ----------
    system : string containing the name of the system formulation between:

        'P->CC', 'Pd->XP', 'P->X', 'Pd->CCP', dP->CCC

    Returns
    -------
    state_variables : list of state variable names
    input_variables : list of input variable names

    """

    if system == "P->CC":
        input_variables = ["cCoP"]
        state_variables = ["CoM", "CoM_dot"]

    elif system == "P->X":
        input_variables = ["cCoP"]
        state_variables = ["DCM"]

    elif system == "dP->CCC":
        input_variables = ["cCoP_dot"]
        state_variables = ["CoM", "CoM_dot", "CoM_ddot"]

    elif system == "dP->CCP":
        input_variables = ["cCoP_dot"]
        state_variables = ["CoM", "CoM_dot", "cCoP"]

    elif system == "J->CCC":
        input_variables = ["CoM_dddot"]
        state_variables = ["CoM", "CoM_dot", "CoM_ddot"]

    return input_variables, state_variables

def discretize(G, *H):
    """
    This function takes the matrices G and H = [h1, h2, ... ] from a
    continuous-time control system of the form:

        dx/dt = Gx + h1u1 + h2u2 + ... ,

    and returns the matrices A and B = [b1, b2, ...] for the exact discretized
    system:

        x^+ = Ax + b1u1 + b2u2 + ... ,

    with some sampling period tau.

    Parameters
    ----------
    G : Sympy.Matrix nXn square matrix representing the continuous time system
    *H : Sympy.Matrix nXm with n rows and m columns for the entry on m inputs

    Returns
    -------
    A : Sympy.Matrix nXn square matrix representing the discrete time system
    B : Sympy.Matrix nXm with n rows and m columns for the entry on m inputs
    """

    T_, t_ = sy.symbols("tau, t")

    A = sy.exp(G * T_)
    B = tuple([sy.integrate(sy.exp(G * t_), (t_, 0, T_), conds="none") * h for h in H])

    return (A, *B)    

def do_not_update(sys, **kargs):
    return None