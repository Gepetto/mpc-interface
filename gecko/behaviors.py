#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:56:04 2022

@author: nvilla
"""
import numpy as np
import tools as use
from combinations import LineCombo

## TODO: Class for solitary domain such as "n"


class ControlSystem:
    def __init__(self, input_names, state_names, A, B, axes=None,
                 time_variant=False, how_to_update_matrices=None):
        
        self.state_names = state_names
        self.input_names = input_names
        self.A = A
        self.B = B
        
        self.axes = axes
        self.time_variant = time_variant
        if how_to_update_matrices is None or time_variant is None:
            self.__figuring_out = (lambda self:None)
        else:
            self.__figuring_out = how_to_update_matrices
            
    def update_matrices(self, **kargs): 
        self.__figuring_out(self, **kargs)
        
    @classmethod
    def from_name(cls, system_name, discretization_period, omega, axes=None,
                  time_variant=False, how_to_update_matrices=None):
        
        T = discretization_period
        w = omega
        input_names, state_names = use.get_system_variables(system_name)
        
        get_A, get_B = use.get_system_matrices(system_name)
        A = get_A(T, w); B = get_B(T, w)
        
        self = cls(input_names, state_names, A, B, axes, 
                   time_variant, how_to_update_matrices)
        
        self.get_A = get_A; self.get_B = get_B
        self.T = T; self.w = w
        
        return self
          
## TODO: hacer tests para todas las clases.
## TODO: (no por ahora) Tambien seria mas simple si hago funciones para actualizar cada definicion por ella misma si es time_variant
## TODO: (no por ahora) Seria preferible introducir directamente variables y matrices en las definiciones.
class ExtendedSystem:
    def __init__(self, input_names, state_names, state_vector_name, S, U,
                 axes=None, time_variant=False, how_to_update_matrices=None):
        
        """ This class works with a dynamics of the form:
            
            x = S*x0 + U*u
            
            where 
            S : ndarray with shape [N, n, n],
            U : list of m ndarrays with shape [N, p_u, n]
            
            with dimentins N horizon lenght, n number of states, m number of inputs
            and p_u the number of actions predicted for each input.
            
        """
        self.matrices = U + [S] # the order in this list is important
        self.axes = axes
        self.state_vector_name = state_vector_name
        
        self.identify_domain(input_names, state_names)
        self.set_sizes()
        
        self.definitions = {}
        self.make_definitions()
        
        self.time_variant = time_variant
        
        if how_to_update_matrices is None:
            self.__figuring_out = (lambda self:None)
        else:
            self.__figuring_out = how_to_update_matrices
            
    @classmethod         
    def extend_cotrol_system(cls, control_system, state_vector_name,
                             horizon_lenght):
        
        S, U = use.extend_matrices(horizon_lenght, control_system.A, 
                               control_system.B)
        if control_system.time_variant:
            def how_to_update_matrices(ext_syst, **kargs):
                """ This function requires a key argument called 
                'control_system' that is an instance of ControlSystem
                """
                ctr_syst = kargs["control_system"]
                ctr_syst.update_matrices(**kargs)
                
                S, U = use.extend_matrices(horizon_lenght, ctr_syst.A, ctr_syst.B)
                ext_syst.matrices = U + [S]
                
        else:
            how_to_update_matrices = None
        
        self = cls(control_system.input_names, control_system.state_names, 
                   state_vector_name, S, U, control_system.axes, 
                   control_system.time_variant, how_to_update_matrices)
        return self
            
    def identify_domain(self, input_names, state_names):
        
        domain_ID = {name:index for index, name in enumerate(input_names)}
        domain_ID.update({self.state_vector_name+"0":len(input_names)})
        state_ID = {name:index for index, name in enumerate(state_names)}
        
        if self.axes is None:
            self.domain_ID = domain_ID
            self.state_ID = state_ID
        else:
            self.domain_ID = {}
            self.state_ID = {}
            for axis in self.axes:
                self.domain_ID.update({
                        name+axis:ID for name, ID in domain_ID.items()
                        })
                self.state_ID.update({
                        name+axis:ID for name, ID in state_ID.items()
                        })
            
    def set_sizes(self):
        state_sizes2 = {state:self.matrices[-1].shape[0] 
                        for state, ID in self.state_ID.items()}
        self.domain = {base:self.matrices[ID].shape[1] 
                       for base, ID in self.domain_ID.items()}
        self.all_variables = {**self.domain, **state_sizes2}
        
    def update_sizes(self):
        for variable, ID in self.domain_ID.items():
            self.domain[variable] = self.matrices[ID].shape[1]

        self.all_variables.update(self.domain)
        # the state sizes (correspondig to the horizon lenght) are fixed.
            
    def make_definitions(self):
        for variable, size in self.domain.items():
            combination = {variable:np.eye(size)}
            self.definitions.update({variable:LineCombo(combination)})
            
        for variable, sID in self.state_ID.items():
            if self.axes is None:
                comb2 = {base:self.matrices[dID][..., sID]
                        for base, dID in self.domain_ID.items()}
            else:
                comb2 = {base:self.matrices[dID][..., sID]
                        for base, dID in self.domain_ID.items()
                        if base[-2:] == variable[-2:]}
            self.definitions.update({variable:LineCombo(comb2)})
            
    def update_definitions(self):
#This should be reformulated: definitions should have a fixed update method that
#receives and sets new matrices. It makes more sense if we set lists [var],[matri] instead of dict when we define the definition.
# IT should be updated only if "time_variant"
#Actually we can keep both forms to update.
        for variable, size in self.domain.items():
            self.definitions[variable].matrices[0] = np.eye(size)
            
        for state in self.state_ID.keys():
            for vID, var in enumerate(self.definitions[state].variables):
                dID = self.domain_ID[var]
                sID = self.state_ID[state]
                self.definitions[state].matrices[vID] = self.matrices[dID][..., sID]
              
    def update_matrices(self, **kargs): 
        self.__figuring_out(self, **kargs)
    
    def update(self, **kargs):
        if self.time_variant:
            self.update_matrices(**kargs)
            self.update_sizes()
            self.update_definitions()
