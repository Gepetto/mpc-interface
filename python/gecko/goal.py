#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:35:30 2022

@author: nvilla
"""
import numpy as np
#import configuration as conf
class Cost:
    def __init__(self, variable, aim, weight, axes=None,
                 L=None, schedule=None):
        """ 
        ARGUMENTS:
            
            variable : string
            
            aim : ndarray with shape [m, len(axes)] .
            
            weight : positive number.
            
            axes : list of strings of the form ["_x", "_y"]. Defaults to [""]
            
            L : list of ndarray with shapes [m, t]. Defaults to []
            
            schedule : range with t elements <= horizon_lenght. Defaults to range(0)
        
        Each instance of this class provides instructions to bring V
        towards the aimed values aim with some weight.
            
             V --> aim
            
        where V is:
            
            V = L @ v[schedule]
            
        based on the variable v = [v_x, v_y, v_...] which is defined by 
        'variable' and 'axes'.
        """
        
        self.variable = variable 
        self.axes = [""] if axes is None else axes
        assert isinstance(self.axes, list), "The axes must be a list of string"
        self.axes_len = len(self.axes)
        
        self.aim = np.array(aim).reshape([-1, self.axes_len])
        
        self.weight = weight
        self.L = [] if L is None else L
        self.schedule = range(0) if schedule is None else schedule
        
        self.arrange_dimentions()
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
             
    def update(self, aim=None, weight=None, L=None, schedule=None):
        
        if weight is not None:
            self.weight = weight
        
        if aim is not None:
            self.aim = np.array(aim).reshape([-1, self.axes_len])
        
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
                self.m = self.L[0].shape[0]
                
                assert self.L[0].shape[1] == self.t,\
                "arrays in L must have '{}' ".format(self.t)+\
                "columns, which is given by the 'schedule'"
                
            elif self.L:
                self.m = self.L[0].shape[0]
                self.t = None
                
            elif self.schedule:
                self.t = self.schedule.stop - self.schedule.start
                self.m = None; 
                
            else: self.t = None; self.m = None; self.nlines = None
            
        if L is not None:
            self.compute_matrices()
            
    def compute_matrices(self):
        if self.L:
            self.matrices = self.L
        else:
            self.matrices = [1]*self.axes_len

    def broadcast(self):
        if self.L or self.schedule:
            m = self.t if self.L == [] else self.m
       
            if self.aim.shape[0] == 1:
                self.aim = np.resize(self.aim, [m, self.axes_len])
            
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        text = ""
        for i, axis in enumerate(self.axes):
            text += "\n"+self.variable+axis
            spaces = len(self.variable+axis)
            for j, goal in enumerate(self.aim[:, i]):
                if j == 0:
                    text += " --> " + str(goal)+"\t\t~ weight = {} ~\n".format(self.weight)
                else:
                    text += " "*spaces +" --> " + str(self.aim[j, i]) +"\n"
        return text+"\n"
