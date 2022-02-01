#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 20:07:59 2022

@author: nvilla
"""
import numpy as np


class Formulation:
    def __init__(self):
        self.domain = {}            # list of strings with all domain variables
        self.optim_variables = []
        self.given_variables = []
        self.optim_sizes = []
        self.given_sizes = []
        self.optim_len = 0        # number of elements in the qp_solution   
        self.given_len = 0          # number of elements in the qp_given
        self.optim_ID = None       # range of each variable in the qp_solution
        self.given_ID = None         # range of each variable in the qp_given
        
        self.definitions = {}       # dictionary with all the variable definitions
        self.dynamics = {} # dictionary with named Dynamics
        self.of = {}       # dict mapping each variable with its dynamics name.

        self.constraints = {}  # constraint names related to list of restrictions.
        self.goals = {} # named costs functions

    # INCORPORATIONS
    def incorporate_dynamics(self, name, new_dynamics):
        self.domain.update(new_dynamics.domain)
        self.dynamics.update({name: new_dynamics})
        self.of.update(
            {variable:name for variable in new_dynamics.all_variables.keys()} 
        )
        self.definitions.update(new_dynamics.definitions) # This could go trough the function self.define.
                                                            # Thought it is different, domain variables depend on them selves, they wouldn't pass.

    def define_variable(self, name, new_definition):
        for variable in new_definition.keys():
            assert variable in self.definitions.keys(), "The definition must depend only on defined variables."
        self.definitions.update({name, new_definition})

    def restrict(self, name, new_constraint): pass

    def aim(self, name, new_goal): pass

    # RELATED TO THE OPTIMAL PROBLEM:
    def identify_qp_domain(self, optimization_domain):
        """ optimization_domain is a list of strings representing
         the name of optimal variables """

        self.optim_variables = optimization_domain
        self.given_variables = [variable for variable in self.domain.keys()
                                       if variable not in optimization_domain]

    def update_qp_sizes(self):
        """ It requires an updated self.domain dictionary """
        self.optim_sizes = [self.domain[variable] for variable in self.optim_variables]
        self.given_sizes = [self.domain[variable] for variable in self.given_variables]
        self.optim_len = sum(self.optim_sizes)  
        self.given_len = sum(self.given_sizes)

    def update_qp_IDs(self):
        """ It requires updated qp_sizes """
        optim_ranges = [range(sum(self.optim_sizes[:i]),
                              sum(self.optim_sizes[:i+1]))
                        for i in range(len(self.optim_sizes))]
        
        given_ranges = [range(sum(self.given_sizes[:i]),
                              sum(self.given_sizes[:i+1]))
                        for i in range(len(self.given_sizes))]

        self.optim_ID = dict(zip(self.optim_variables, optim_ranges))
        self.given_ID = dict(zip(self.given_variables, given_ranges))
        
    def update(self, **kargs):
        for dynamics in self.dynamics.values():
            if dynamics.time_variant:
                dynamics.update(**kargs)
            
        for definition in self.definitions.values():
            if definition.time_variant:
                definition.update()
        
        self.update_qp_sizes()
        self.update_qp_IDs()
        
        #### Check if this function is complete
    
    def make_preview_matrices(self):
        self.PM = {}
        
        for variable in self.definitions.keys():
            if variable in self.of.keys():
                self.PM.update({
                        variable:self.get_matrices_from_behavior(variable)
                        })
            else:
                self.PM.update({
                        variable:self.get_matrices_from_definition(variable)
                        })
            
    def get_matrices_from_behavior(self, variable):
        
        behavior = self.dynamics[self.of[variable]]
        Mg = np.zeros([behavior.all_variables[variable], self.given_len])
        Mo = np.zeros([behavior.all_variables[variable], self.optim_len])
        
        for var, matrix in self.definitions[variable].items():
            if var in self.given_variables:
                Mg[:, self.given_ID[var]] = matrix
                
            elif var in self.optim_variables:
                Mo[:, self.optim_ID[var]] = matrix
            
            else:
                raise ValueError("The variable {} ".format(var) +
                                 "in the definition of {} ".format(variable)+
                                 "seems to not be given nor optimal.")
        return Mg, Mo
    
    def get_matrices_from_definition(self, variable):
        
        combination = self.definitions[variable]
        for i, var in enumerate(combination.keys()):
            if i == 0:
                Mg = np.array(combination[var]).dot(self.PM[var][0])
                Mo = np.array(combination[var]).dot(self.PM[var][1])
                
            else:
                Mg += np.array(combination[var]).dot(self.PM[var][0])
                Mo += np.array(combination[var]).dot(self.PM[var][1])
            
        return Mg, Mo