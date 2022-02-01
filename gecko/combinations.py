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
