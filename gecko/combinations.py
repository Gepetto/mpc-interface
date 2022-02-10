import gecko.tools as use

class LineCombo:
    def __init__(self, combination=None, data=None,
                 how_to_update=None, time_variant=False):
        
        if combination is not None:
            variables, matrices = zip(*combination.items());
            self.variables = list(variables)
            self.matrices = list(matrices)
        
        self.data = [] if data is None else data
        self.time_variant = time_variant
        
        if how_to_update is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update
        
        self._coefficients = ["C"+str(i) for i in range(len(variables))]
        
    def update(self): self.__figuring_out(self)
    
    def __getitem__(self, variable):
        return self.matrices[self.variables.index(variable)]
    
    def items(self):
        return zip(self.variables, self.matrices)
    
    def keys(self):
        return self.variables
    
    def values(self):
        return self.matrices
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        
        comb_text =" + ".join(coef+" ( "+var+" )" 
                              for coef, var in zip(self._coefficients,
                                                   self.variables))
        return comb_text
