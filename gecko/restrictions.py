#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:04:15 2021

@author: nvilla
"""
import numpy as np
import scipy.spatial as sp
import gecko.tools as use

## TODO: make some visualization of the constraints graphically

class Constraint:
    def __init__(self, variable, extreme, axes=None, arrow=None,
                 center=None, L=None, schedule=None):
        """ 
        ARGUMENTS:
            
            variable : dict, {name: String, combination: dict }
            
            axes : list of strings ex: ["_x", "_y", "_z"] always starting "_"
            
            arrow :   ndarray with shape [m, len(axes)] or list
            
            extreme : ndarray with shape [m, 1] or single number, or list.
            
            center :  ndarray with shape [m, len(axes)] or list
            
            L : list of [len(axes)] ndarrays each one with shape [m, t]
            
            schedule : range with t elements with t <= horizon_lenght
        
        The null value of L is an empty list [] and the null value of 
        schedule is range(0). These values can be used in the update 
        function to remove them.
        
        Each instance of this class provides instructions to generate 
        m (or t (or horizon_lenght)) lineal constraints of the form:
            
            arrow * ( V - center ) < extreme
            
        where V is:
            
            V = [  Lx @ v_x[schedule], Ly @ v_y[schedule], ... ]
            
        based on the variable v = [v_x, v_y] which is defined by 'variable'.
        """
        self.variable = variable
        self.extreme = np.array(extreme).reshape([-1, 1]).astype(float) 
        
        self.axes = [""] if axes is None else axes
        assert isinstance(self.axes, list), "The axes must be a list of string"
        
        
        self.arrow = arrow
        self.center = center
        
        self.L = [] if L is None else L
        self.schedule = range(0) if schedule is None else schedule
        self.m = None
        self.t = None
        self.axes_len = len(self.axes)
        self.nlines = None
        self.matrices = []
        
        self.arrange_dimentions()
        self.normalize()
        self.compute_matrices()
        
    def arrange_dimentions(self):
        if self.schedule:
            self.t = self.schedule.stop - self.schedule.start
        
        if not isinstance(self.L, list):
            self.L = [self.L]
        
        if self.L:
            if len(self.L) == 1:
                self.L = self.L*self.axes_len
                
            if self.schedule:
                assert self.L[0].shape[1] == self.t,\
                "arrays in L must have '{}' ".format(self.t)+\
                "columns, which is given by the 'schedule'"
            self.m = self.L[0].shape[0]
            self.nlines = self.m
             
        elif self.schedule:
                self.nlines = self.t
        
        if self.arrow is None:    
            if self.axes_len == 1: 
                self.arrow = np.ones(self.extreme.shape)
            else:
                assert self.arrow is not None,"with multiple axes, "+\
                "some normal direction 'arrow' must be provided"
        else:
            self.arrow = np.array(self.arrow).reshape([-1, self.axes_len])
        
        if self.arrow.shape[0] != self.extreme.shape[0]:
            self.arrow = np.resize(self.arrow, [self.extreme.shape[0], 
                                                self.axes_len])
            
        if self.center is None:
            self.center = np.zeros([1, self.axes_len])
        else:
            self.center = np.array(self.center).reshape([-1, self.axes_len])       
    
    def broadcast(self):
        if self.L or self.schedule:
            m = self.t if self.L == [] else self.m
       
            if self.extreme.shape[0] == 1:
                self.extreme = np.resize(self.extreme, [m, 1])
                
            if self.arrow.shape[0] == 1:
                self.arrow = np.resize(self.arrow, [m, self.axes_len])
            
            if self.center.shape[0] == 1:
                self.center = np.resize(self.center, [m, self.axes_len])
        
    def compute_matrices(self):
        if self.L:
            self.matrices = [self.arrow[:,i][:,None]*l 
                             for i, l in enumerate(self.L)]
        else:
            self.matrices = [self.arrow[:,i][:,None] 
                             for i in range(self.axes_len)]
            
    def normalize(self):
        for i, extreme in enumerate(self.extreme):
            # We adapt the arrow to have always positive extreme
            if extreme < 0:
                self.extreme[i] = -self.extreme[i]
                self.arrow[i] = -self.arrow[i]
#            Unit vectors in arrow?

    def bound(self):
        return self.extreme + np.sum(self.arrow * 
                                     self.center, axis=1).reshape([-1, 1])
    
    def update(self, extreme=None, arrow=None, center=None,
               L=None, schedule=None):
        
        if schedule is not None:
            self.schedule = schedule
            
        if L is not None:
            if not isinstance(L, list):
                L = [L]
            if len(L) == 1:
                L = L*self.axes_len
            self.L = L
           
        if L is not None or schedule is not None:
            if self.schedule and self.L:
                self.t = self.schedule.stop - self.schedule.start
                
                assert self.L[0].shape[1] == self.t,\
                "arrays in L must have '{}' ".format(self.t)+\
                "columns, which is given by the 'schedule'"
                
                self.m = self.L[0].shape[0]
                self.nlines = self.m
                
            elif self.L:
                self.m = self.L[0].shape[0]
                self.nlines = self.m; self.t = None
                
            elif self.schedule:
                self.t = self.schedule.stop - self.schedule.start
                self.nlines = self.t; self.m = None; 
                
            else: self.t = None; self.m = None; self.nlines = None
            
        if extreme is not None:
            self.extreme = np.array(extreme).reshape([-1, 1])
        if arrow is not None:
            self.arrow = np.array(arrow).reshape([-1, self.axes_len])
        if center is not None:
            self.center = np.array(center).reshape([-1, self.axes_len])    
            
        if arrow is not None and extreme is None:
            if self.arrow.shape[0] != self.extreme.shape[0]:
                self.extreme = np.resize(self.extreme, [self.arrow.shape[0],
                                                        1])
            self.normalize()     
            
        elif arrow is not None:
            if self.arrow.shape[0] == 1:
                self.arrow = np.resize(self.arrow, [self.extreme.shape[0],
                                                        self.axes_len])
            elif self.extreme.shape[0] == 1:
                self.extreme = np.resize(self.extreme, [self.arrow.shape[0],
                                                        1])
            self.normalize()
            
        elif extreme is not None:
            if self.arrow.shape[0] != self.extreme.shape[0]:
                self.arrow = np.resize(self.arrow, [self.extreme.shape[0],
                                                        self.axes_len])
            self.normalize()        
        
        if arrow is not None or L is not None:
            self.compute_matrices()
    
    def is_feasible(self, points):
        """ Introduce a list of points to verify the feasibility of each
        one separately, or introduce a np.vstack of row points to check 
        feasibility along the time horizon. Or a list of np.stacks if 
        prefered.
        for evaluations along hte horizon, points are considered to ve at the 
        scheduled times.
        """
        if isinstance(points, list):
            return [self._is_feasible(point) for point in points]
        else:
            return self._is_feasible(points)
        
    def _is_feasible(self, point):
        if self.L:
            ts_point = np.vstack([(l@p.T) for l, p in zip(self.L, point.T)]).T
            return self.arrow*(ts_point - self.center) < self.extreme
        return np.sum(self.arrow*(point - self.center), axis=1) < self.extreme
            
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        if self.axes != [""]:
            axes = "_"+"".join(axis[1:] for axis in self.axes)
        else:
            axes = ""
        
        text = "\nvariable: "+ self.variable+axes 
        text+= "\nwith L = " +str(",\n".join(str(l) for l in self.L)) if self.L != [] else ""
        text+= "\n\t\t\t\t\tarrow: "+ (" "*7).join(str(arrow) for arrow in self.arrow)
        text+= "\n\t\t\t\t\tcenter: "+ (" "*8).join(str(center) for center in self.center)
        text+= "\n\t\t\t\t\textreme: "+ (" "*9).join(str(extreme) for extreme in self.extreme)
        text+= "\n"
        return text
    
#    def __repr__(self):
#        if self.axes != [""]:
#            axes = "_"+"".join(axis[1:] for axis in self.axes)
#        else:
#            axes = ""
#        bounds = self.bound()
#        if bounds.shape[0] == 1:
#            timing = "limits of the form:\n"
#            text = timing + "\n".join("\t"+"arrow @ " + self.variable+axes +
#                                      " < "+str(bound[0]) for bound in bounds)   
#        else:
#            text = "\n".join("limit {}:\t\t".format(i)+"arrow @ " + 
#                             self.variable + axes + " < "+str(bound[0]) 
#                             for i, bound in enumerate(bounds))
#        return text
                
## TODO: The axes should also be an atribute of the box
class Box:
    def __init__(self, time_variant=None, how_to_update=None):
        
        self.constraints = []
        
        self.center = np.array([])
        self.rotation = [np.array([])]
        self.scale_factor = np.array([1.])
        self.schedule = range(0)
        self.safety_margin = 0
        
        self.time_variant = time_variant
        
        if how_to_update is None or not time_variant:
            self.__figuring_out = use.do_not_update
        else:
            self.__figuring_out = how_to_update
    
    @classmethod        
    def task_space(cls, variable, vertices, axes=None, L=None, schedule=None,
                   time_variant=None, how_to_update=None):
        
        box = cls(time_variant, how_to_update)
        
        arrows, extremes, center = box_boundaries(vertices)
        for i, extreme in enumerate(extremes):
            box.constraints.append(Constraint(variable, extreme, axes,
                                               arrows[i], center, L, schedule))
            
        return box
    
    @classmethod        
    def state_space(cls, variable, vertices, axes=None, schedule=None,
                   time_variant=None, how_to_update=None):
        
        box = cls(time_variant, how_to_update)
        
        arrows_SS, extremes_SS, center_SS = box_boundaries(vertices)
        center = np.sum(arrows_SS*center_SS, axis=1).reshape([-1, 1])
        
        for i, extreme in enumerate(extremes_SS):
            box.constraints.append(Constraint(variable, extreme, axes=axes,
                                    center=center[i], L=arrows_SS[i],
                                    schedule=schedule))
        
        return box
        
    def move_SS_box(self, new_center):
        assert np.shape(new_center)[-1] == self.constraints[0].L[0].shape[-1] # TODO: Save the dimention of the TS and SS in the box to avoid taking it from the first constraint
        self.center = np.array(new_center)
        for boundary in self.constraints:
            center = np.sum(boundary.L*np.array(new_center), 
                            axis=1).reshape([-1, 1])
            boundary.update(center=center)
    ## TODO: These functions should be "move_in_SS" and "move_in_TS"
    def move_TS_box(self, new_center):
        self.center = new_center
        for boundary in self.constraints:
            boundary.update(center=np.array(new_center))
    
    def rotate_box(self, rotation):
        pass
    
    def scale_box(self, scale_factor):
        
        for boundary in self.constraints:
            boundary.update(extreme = boundary.extreme * 
                                      scale_factor/self.scale_factor)
        self.scale_factor = scale_factor
        
    def set_safety_margin(self, margin):
        
        for boundary in self.constraints:
            boundary.update(extreme = boundary.extreme - 
                            margin*np.linalg.norm(boundary.arrow))
        self.safety_margin = margin
    ## TODO: Make functions"is_feasible_SS" and "is_feasible_TS" to ask for points in any representation
    def is_feasible(self, points):
        if not isinstance(points, list):
            points = [points]
        
        feasible = []
        for point in points:
            feasible.append(all(
                    [limit.is_feasible(point) 
                    for limit in self.constraints]
                    ))
        return feasible
        
    def update(self, **kargs): 
        self.__figuring_out(self, **kargs)
        
def box_boundaries(vertices):
    """ vertices is an ndarray with one vertex per row."""
    vertices = vertices.astype("float64")
    n, dim = vertices.shape
    center = np.sum(vertices, axis=0)/n
    
    if dim == 1:
        simplices = np.array([[0], [1]])
        arrows = np.ones([2, 1])
    
    elif dim == 2:
        simplices = sp.ConvexHull(vertices).simplices
        
        directions = (vertices[simplices[:, 0]] - 
                      vertices[simplices[:, 1]])
        
        arrows = np.hstack([ directions[:, 1].reshape(-1, 1),
                            -directions[:, 0].reshape(-1, 1)])
                
    elif dim == 3:
        simplices = sp.ConvexHull(vertices).simplices
        
        d0 = (vertices[simplices[:, 0]] - 
              vertices[simplices[:, 2]])
        d1 = (vertices[simplices[:, 1]] - 
              vertices[simplices[:, 2]])
        
        arrows = np.hstack([(d0[:, 1]*d1[:, 2]-
                             d0[:, 2]*d1[:, 1]).reshape(-1, 1), 
                            (d0[:, 2]*d1[:, 0]-
                             d0[:, 0]*d1[:, 2]).reshape(-1, 1),
                            (d0[:, 0]*d1[:, 1]-
                             d0[:, 1]*d1[:, 0]).reshape(-1, 1)])

    for i, arrow in enumerate(arrows):
        arrows[i] = arrow/np.linalg.norm(arrow)
        
    furthest_vertex = vertices[simplices[:, 0]]
    extremes = np.sum(arrows * (furthest_vertex - center), 
                      axis=1).reshape([-1, 1])
                              
    return arrows, extremes, center 

    
    
    