import numpy as np

class LineCombo:
    def __init__(self, combination=None, data=None,
                 how_to_update=None, time_variant=False):
        
        if combination is not None:
            variables, matrices = zip(*combination.items());
            self.variables = list(variables)
            self.matrices = list(matrices)
        
        self.data = [] if data is None else data
        self.time_variant = time_variant
        
        if how_to_update is None:
            self.__figuring_out = (lambda self:None)
        else:
            self.__figuring_out = how_to_update
        
    def update(self): self.__figuring_out(self)
    
    def __getitem__(self, variable):
        return self.matrices[self.variables.index(variable)]
    
    def items(self):
        return zip(self.variables, self.matrices)
    
    def keys(self):
        return self.variables
    
    def values(self):
        return self.matrices
    
## TODO: Hace la clase ControlSystem y generar un ExtendedSystem a partir de esta.
## TODO: (no por ahora) Tambien seria mas simple si hago funciones para actualizar cada definicion por ella misma si es time_variant
## TODO: (no por ahora) Seria preferible introducir directamente variables y matrices en las definiciones.
class ExtendedSystem:
    def __init__(self, input_names, state_names, state_vector_name, S, U,
                 axes=None, time_variant = False, how_to_update_matrices=None):
        
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
        
        syst.update_qp_sizes()
        syst.update_qp_IDs()
        
        pass
    
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
        
        
def update_step_matrices(extSyst, **kargs):
    """ This function needs
        
        count : int, representing the current time sample number
        
        and one of the following:
        step_times : ndarray or list with next step times.
        
        or 
        
        regular_time : int, to produce steps regularly
        
    """
    N = extSyst.matrices[-1].shape[0]
    count = kargs["count"]
    
    preview_times = count + np.arange(N)
    
    if "step_times" in kargs.keys():
        step_times = kargs["step_times"]
        next_steps = step_times[(step_times >= count) * 
                                (step_times < count+N-1)]
        
    elif "regular_time" in kargs.keys():
        regular_time = kargs["regular_time"]
        next_steps = np.array([time for time in preview_times
                      if not time%regular_time and time < count+N-1])
        
    else:
        raise KeyError("This funtion needs either 'step_times' or "+
                       "'regular_time', but the kargs "+
                       "introduced are {}".format(kargs.keys()))
        
    U = (preview_times.reshape([N, 1]) > next_steps).astype(int)
    extSyst.matrices[0] = U[:, :, None]

if __name__ == "__main__":
    
    # configurando pasos
    
    steps = ExtendedSystem(["Ds"], ["s"], "s", 
                            np.ones([9, 1, 1]),
                            [(np.arange(9).reshape([-1,1])>=[2, 5, 8]).astype(int)[:, :, None]],
                            ["_x", "_y"], how_to_update_matrices=update_step_matrices, 
                            time_variant=True)
    
    syst = Formulation()
    syst.incorporate_dynamics("steps", steps)
    
    # Ranges
    syst.identify_qp_domain(["Ds_x", "Ds_y"])
    syst.update_qp_sizes()
    syst.update_qp_IDs()
    
    # MAtrices
    # Se hace necesario el parametro N de las dinamicas
    Mg1, Mo1 = syst.get_matrices_from_behavior("s0_x")
    syst.make_preview_matrices()
    Mg2, Mo2 = syst.get_matrices_from_definition("s_x")
