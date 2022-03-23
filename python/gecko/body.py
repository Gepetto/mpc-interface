#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 20:07:59 2022

@author: nvilla
"""
import numpy as np
import gecko.tools as use
#from scipy import sparse

class Formulation:
    def __init__(self):
        self.domain = {}            # list of strings with all domain variables
        self.optim_variables = []
        self.given_variables = []
        self.optim_sizes = []
        self.given_sizes = []
        self.optim_len = 0          # number of elements in the qp_solution   
        self.given_len = 0          # number of elements in the qp_given
        self.optim_ID = {}         # range of each variable in the qp_solution
        self.given_ID = {}         # range of each variable in the qp_given
        self.domain_ID = {"optim_ID":self.optim_ID,
                          "given_ID":self.given_ID}
        
        self.definitions = {}       # dictionary with all the variable definitions
        self.dynamics = {} # dictionary with named Dynamics
        self.of = {}       # dict mapping each variable with its dynamics name.
        
        self.constraint_boxes = {}
        self.constraints = {}  # constraint names related to list of restrictions.
        self.goals = {} # named costs functions
        
        self.update_incorporations = use.do_not_update

    # INCORPORATIONS
    def incorporate_dynamics(self, name, new_dynamics):
        self.domain.update(new_dynamics.domain)
        self.dynamics.update({name: new_dynamics})
        self.of.update(
            {variable:name for variable in new_dynamics.all_variables.keys()} 
        )
        self.definitions.update(new_dynamics.definitions) # This could go trough the function self.define.
                                                            # Thought it is different, domain variables depend on them selves, they wouldn't pass.

    def incorporate_definition(self, name, new_definition):
        for variable in new_definition.keys():
            assert variable in self.definitions.keys(),\
            "The definition must depend only on defined variables, but the "+\
            "this definition depends on "+variable
        self.definitions.update({name:new_definition})
        
    def incorporate_definitions(self, dict_of_defs):
        for name, combination in dict_of_defs.items():
            self.incorporate_definition(name, combination)

    def incorporate_constraint(self, name, new_limits): 
        if not isinstance(new_limits, list) :
            new_limits = [new_limits]
        
        for limit in new_limits:
            for axis in limit.axes:
                assert limit.variable+axis in self.definitions.keys(),\
                "All constrained variables must be previously defined, "+\
                "but '{}' was not defined.".format(limit.variable+axis) 
            
        self.constraints.update({name : new_limits})
        
    def incorporate_box(self, name, new_box):
        
        for limit in new_box.constraints:
            for axis in limit.axes:
                assert limit.variable+axis in self.definitions.keys(),\
                "All constrained variables must be previously defined, "+\
                "but '{}' was not defined.".format(limit.variable+axis) 
        
        self.constraint_boxes.update({name : new_box})
        
    def incorporate_goal(self, name, new_goal):
        
        for axis in new_goal.axes:
            assert new_goal.variable+axis in self.definitions.keys(),\
            "All constrained variables must be previously defined, "+\
            "but '{}' was not defined.".format(new_goal.variable+axis) 
            
        self.goals.update({name : new_goal})

    # RELATED TO THE OPTIMIZATION PROBLEM:
    def identify_qp_domain(self, optimization_domain):
        """ optimization_domain is a list of strings representing
         the name of optimal variables """

        self.optim_variables = optimization_domain
        self.given_variables = [variable for variable in self.domain.keys()
                                       if variable not in optimization_domain]
        
        self.update_qp_sizes()
        self.update_qp_IDs()
        
    def update_qp_domain(self):
        for behavior in self.dynamics.values():
            if behavior.time_variant:
                self.domain.update(behavior.domain)

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

        self.optim_ID.update(dict(zip(self.optim_variables, optim_ranges)))
        self.given_ID.update(dict(zip(self.given_variables, given_ranges)))
    
    def set_updating_rule(self, how_to_update=None):
        
        self.update_incorporations = how_to_update
        
    def update(self, **kargs):
        self.update_incorporations(self, **kargs)
        self.update_qp_domain()
        self.update_qp_sizes()
        self.update_qp_IDs()
        self.make_preview_matrices()
    
    def make_preview_matrices(self):
        self.PM = {}
        
        for variable in self.definitions.keys():
            if variable in self.of.keys():
                self.PM.update({
                        variable:self.get_matrices_from_dynamics(variable)
                        })
            else:
                self.PM.update({
                        variable:self.get_matrices_from_definition(variable)
                        })
            
    def get_matrices_from_dynamics(self, variable):
        
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
                if len(Mg.shape) == 1:
                    Mg = Mg[None, :]; Mo = Mo[None, :]
            else:
                Mg += np.array(combination[var]).dot(self.PM[var][0])
                Mo += np.array(combination[var]).dot(self.PM[var][1])

        return Mg, Mo
    
    def arrange_given(self, collector):
        """The collector must contain the values of at least all the given 
        variables, and maybe for all the qp_domain.
        All values must be provided as ndarrays with one single column"""
        
        if self.given_len:
            given = np.zeros([self.given_len, 1])
            for variable, indices in self.given_ID.items():
                given[indices] = collector[variable]
            
        else:
            given = np.array([])
        return given
    
    def preview(self, given, optim, variable, axes=None): 
        if axes is None:
            Mg, Mo = self.PM[variable] 
            return Mg @ given + Mo @ optim
        
        preview = []
        for axis in axes:
            Mg, Mo = self.PM[variable+axis]
            preview.append(Mg @ given + Mo @ optim)
         
        return np.hstack(preview)
        
    def generate_qp_constraint(self, limit, given):## requires updated limit
        
        rows = self.PM[limit.variable+limit.axes[0]][0].shape[0]
        c_rows = rows if limit.nlines is None else limit.nlines
        
        cMg = np.zeros([c_rows, self.given_len])
        cMo = np.zeros([c_rows, self.optim_len])
        
        schedule = range(rows) if not limit.schedule else limit.schedule
        bound = limit.bound() 
        
        for i, axis in enumerate(limit.axes):
            Mg, Mo = self.PM[limit.variable+axis]
            c = limit.matrices[i]
            
            if limit.L:
                cMg += c @ Mg[schedule]; cMo += c @ Mo[schedule]
                
            else:
                cMg += c * Mg[schedule]; cMo += c * Mo[schedule]

        A = cMo
        h = bound - cMg @ given
        
        return A, h
    
    def generate_qp_cost(self, cost, given):## requires updated limit
        
        Q = np.zeros([self.optim_len, self.optim_len])
        q = np.zeros([self.optim_len, 1])
        
        rows = self.PM[cost.variable+cost.axes[0]][0].shape[0]
        schedule = range(rows) if not cost.schedule else cost.schedule
        
        for i, axis in enumerate(cost.axes):
            Mg, Mo = self.PM[cost.variable+axis]
            c = cost.matrices[i]
            
            if cost.L:
                cMg = c @ Mg[schedule]; cMo = c @ Mo[schedule]
            else:
                cMg = c * Mg[schedule]; cMo = c * Mo[schedule]
            
            Q += cost.weight * cMo.T @ cMo
            q += cost.weight * cMo.T @ (cMg @ given - cost.aim[:, i])
        
        return Q, q
    
    def generate_all_qp_constraints(self, given):
        
        matrices = [self.generate_qp_constraint(limit, given)
                    for area in self.constraints.values()
                    for limit in area]
        matrices += [self.generate_qp_constraint(limit, given)
                     for box in self.constraint_boxes.values()
                     for limit in box.constraints]
        
        A = np.vstack([matrix[0] for matrix in matrices])
        h = np.vstack([matrix[1] for matrix in matrices])
        
        return A, h
    
    def generate_all_qp_costs(self, given):

        matrices = [self.generate_qp_cost(cost, given) 
                    for cost in self.goals.values()]

        Q = np.add.reduce([Q[0] for Q in matrices])
        q = np.add.reduce([q[1] for q in matrices])

        return Q, q
    
    ## TODO: we are missing equality constraints.
        
    def generate_all_qp_matrices(self, given):
        """ 
        Argument:
            given_collector: dictionary containing the values of all given
                             variables as {"variable": column np.array}
                             
            The given variables are reported 
            by asking for `self.given_variables`
            
        """
        
        
        A, h = self.generate_all_qp_constraints(given)
        Q, q = self.generate_all_qp_costs(given)
        
        return A, h, Q, q
    
        
        
        
        
        
        
        
        