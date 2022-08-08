#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 19:56:04 2022

@author: nvilla
"""
import numpy as np
import mpc_core.tools as use
from mpc_core.combinations import LineCombo
from collections.abc import Iterable

"""
This module defines two classes
    
    DomainVariable
    ExtendedSystem
    
that will be used as dynamics in the qp formulation.

A third classs is defined that serves as an auxiliar 
class to define ExtendedSystems:
    
    ControlSystem
    
"""

class DomainVariable:
    def __init__(self, names, sizes, axes=None, time_variant=False,
                 how_to_update_size=None):
        
        if isinstance(names, str):
            names = [names]
            
        for name in names: 
            if not isinstance(name, str) : 
                raise TypeError("all variable names must be strings.")
        
        if not isinstance(sizes, Iterable): 
            sizes = [sizes]
            
        if len(names) != len(sizes):
            raise IndexError("'names' and 'sizes' must have the same amount "+
                             "of elements.")
        
        if axes is None:
            self.axes = [""]
        elif not isinstance(axes, Iterable):
            self.axes = [str(axes)]
        elif isinstance(axes, str):
            self.axes = [axes]
        else:
            self.axes = axes
        
        self.identify_domain(names)
        self.set_sizes(names, sizes)
        
        self.outputs = []
        self.definitions = {}
        self.make_definitions()
        
        self.time_variant = time_variant
        if how_to_update_size is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update_size
            
    def identify_domain(self, names):
        
        self.domain_ID = {}
        for axis in self.axes:
            self.domain_ID.update({
                    name+axis:index for index, name in enumerate(names)
                    })
            
    def set_sizes(self, names, sizes):
        self.domain = {name:sizes[ID] for name, ID in self.domain_ID.items()}
        self.all_variables = self.domain
            
    def update_sizes(self, **kargs):
        self.__figuring_out(self, **kargs)

    def make_definitions(self):
        for variable, size in self.domain.items():
            combination = {variable:np.eye(size)}
            self.definitions.update({variable:LineCombo(combination)})
            self.definitions[variable]._coefficients = ["I"]
            
    def define_output(self, name, combination, time_variant=False, 
                       how_to_update=None):
        """ 
        This function incorporates additional (output) definitions related to
        the extended system. 
        
        If needed, the funtion "how_to_update" can only have one karg called
        "domVar" which refers to the extended system where the output is 
        defined.
        """
        for axis in self.axes:
            comb = {var+axis:value for var, value in combination.items()}
            
            self.definitions.update({
                    name+axis:LineCombo(
                            comb, time_variant=time_variant,
                            how_to_update=how_to_update)
                    })
            self.outputs.append(name+axis)
        
    def update_definitions(self):
        for variable, size in self.domain.items():
            self.definitions[variable].matrices[0] = np.eye(size)
        
        for output in self.outputs:
            self.definitions[output].update(domVar = self)
            
    def update(self, **kargs):
        self.update_sizes(**kargs)
        self.update_definitions()              
        
## TODO: Make a form to deal with axis names longer (or shorter) than 2 characters or rise an error when the axes have more (or less) than 2 characters
## TODO: the previous point can be done with variables of hte form tuple(name, axis) which is immutable and we can separate name and axis easily.        
## TODO: Report some how what should be in the **kargs for update functions
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
            
            When the number of inputs is 1 a ndarray is admited for U instead 
            of using a list, And when the number of state variables is 1, 
            2D matrices are admited in S and U.
            
        """
        if isinstance(input_names, str):
            input_names = [input_names]
            
        for name in input_names: 
            if not isinstance(name, str) : 
                raise TypeError("all input names must be strings.")
                
        if isinstance(state_names, str):
            state_names = [state_names]
            
        for name in state_names: 
            if not isinstance(name, str) : 
                raise TypeError("all variable names must be strings.")
        
        if not isinstance(state_vector_name, str):
            raise TypeError("the 'state_vector_name' must be a single string.")
            
        if not isinstance(U, list) and len(input_names) == 1:
            U = [U]
        
        if len(state_names) == 1:
            if len(S.shape) == 2:
                S = S[:, :, None]
            for i, u in enumerate(U):
                if len(u.shape) == 2:
                    U[i] = u[:, :, None]
        
        if axes is None:
            self.axes = [""]
            
        elif not isinstance(axes, Iterable):
            self.axes = [str(axes)]
            
        elif isinstance(axes, str):
            self.axes = [axes]
            
        else:
            self.axes = axes
        
        self.matrices = U + [S] # the order in this list is important
        self.state_vector_name = state_vector_name
        
        self.identify_domain(input_names, state_names)
        self.set_sizes()
        
        self.outputs = []
        self.definitions = {}
        self.make_definitions()
        
        self.time_variant = time_variant
        
        if how_to_update_matrices is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update_matrices
        
    @classmethod         
    def from_cotrol_system(cls, control_system, state_vector_name,
                             horizon_lenght):
        
        S, U = use.extend_matrices(horizon_lenght, control_system.A, 
                                                   control_system.B)
        if control_system.time_variant:
            def how_to_update_matrices(ext_syst, **kargs):
                """ This function requires a key argument called 
                'control_system' that is the instance of ControlSystem
                that was extended to obtain this extended control system
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
        state_sizes = {state:self.matrices[-1].shape[0] 
                        for state, ID in self.state_ID.items()}
        self.domain = {base:self.matrices[ID].shape[1] 
                       for base, ID in self.domain_ID.items()}
        self.all_variables = {**self.domain, **state_sizes}
        
    def update_sizes(self):
        if self.time_variant:
            for variable, ID in self.domain_ID.items():
                self.domain[variable] = self.matrices[ID].shape[1]
    
            self.all_variables.update(self.domain)
        # the state sizes (correspondig to the horizon lenght) are constant.
            
    def make_definitions(self):
        for variable, size in self.domain.items():
            combination = {variable:np.eye(size)}
            self.definitions.update({variable:LineCombo(combination)})
            self.definitions[variable]._coefficients = ["I"]
            
        for variable, sID in self.state_ID.items():
            if self.axes == [""]:
                comb = {base:self.matrices[dID][..., sID]
                        for base, dID in self.domain_ID.items()}
            else:
                comb = {base:self.matrices[dID][..., sID]
                        for base, dID in self.domain_ID.items()
                        if base[-2:] == variable[-2:]}
            self.definitions.update({variable:LineCombo(comb)})
            self.definitions[variable]._coefficients = (len(self.matrices)-1)*["U"]+["S"]
            
    def define_output(self, name, combination, time_variant=False, 
                       how_to_update=None):
        """ 
        This function incorporates additional (output) definitions related to
        the extended system. 
        
        If needed, the funtion "how_to_update" can only have one karg called
        "extSyst" which refers to the extended system where the output is 
        defined.
        """
        for axis in self.axes:
            comb = {var+axis:value for var, value in combination.items()}
            
            self.definitions.update({
                    name+axis:LineCombo(
                            comb, time_variant=time_variant,
                            how_to_update=how_to_update)
                    })
            if time_variant:
                self.definitions[name+axis].update(extSyst = self)
            self.outputs.append(name+axis)
        
    def update_definitions(self):
        for variable, size in self.domain.items():
            self.definitions[variable].matrices[0] = np.eye(size)
            
        for state in self.state_ID.keys():
            for vID, var in enumerate(self.definitions[state].variables):
                dID = self.domain_ID[var]
                sID = self.state_ID[state]
                self.definitions[state].matrices[vID] = self.matrices[dID][..., sID]
                
        for output in self.outputs:
            self.definitions[output].update(extSyst=self)
              
    def update_matrices(self, **kargs): 
        self.__figuring_out(self, **kargs)
    
    def update(self, **kargs):
        if self.time_variant:
            self.update_matrices(**kargs)
            self.update_sizes()
            self.update_definitions()
            
class ControlSystem:
    def __init__(self, input_names, state_names, A, B, axes=None,
                 time_variant=False, how_to_update_matrices=None):
        
        self.state_names = state_names
        self.input_names = input_names
        self.A = A
        self.B = B
        
        self.axes = axes
        self.time_variant = time_variant
        if how_to_update_matrices is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update_matrices
            
    def update_matrices(self, **kargs): 
        self.__figuring_out(self, **kargs)
    
    @staticmethod
    def check_system_parameters(system_name):
        get_A, get_B, parameters = use.get_system_matrices(system_name)
        return parameters
        
    @classmethod
    def from_name(cls, system_name, axes=None, time_variant=False, 
                   how_to_update_matrices=None, **kargs):

        input_names, state_names = use.get_system_variables(system_name)
        
        get_A, get_B, parameters = use.get_system_matrices(system_name)
        try:
            A = get_A(**kargs); B = get_B(**kargs)
        except:
            raise ValueError("This system formulation requires: "+
                             str(parameters))

        self = cls(input_names, state_names, A, B, axes, 
                   time_variant, how_to_update_matrices)
        
        self.get_A = get_A; self.get_B = get_B
        self.parameters = kargs
        self.system_name = system_name
        
        return self
    