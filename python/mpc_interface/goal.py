#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:35:30 2022

@author: nvilla
"""
import numpy as np


class Cost:
    def __init__(
        self,
        variable,
        weight,
        aim=None,
        axes=None,
        L=None,
        schedule=None,
        cross=None,
        cross_aim=None,
        cross_L=None,
    ):
        """
        ARGUMENTS:

            variable : str
            weight : float positive
            aim : ndarray with shape [m, len(axes)] .
            axes : list of strings of the form ["_x", "_y"]. Defaults to [""]
            L : list of ndarray with shapes [m, t]. Defaults to []
            schedule : range with t elements <= horizon_lenght. Defaults to range(0)
            cross: str
            cross_aim: ndarray with shape [m, len(axes)]
            cross_L: list of ndarray with shapes [m, t]. Defaults to []

        Each instance of this class provides instructions to bring V
        towards the aimed values aim with some weight.

             V --> aim

        where V is:

            V = L @ v[schedule]

        based on the variable v = [v_x, v_y, v_...] which is defined by
        'variable' and 'axes'.

        When 'cross' is provided, the cost works with the crossed produc
        making

            ( V - aim ) * ( C - cross_aim ) --> 0

        where C is:

            C = cross_L @ c [schedule]

        where the variable c = [c_x, c_y, c_...] is defined by the 'cross' and
        the 'axes'.

        """
        self.axes = axes if axes else [""]
        if not isinstance(self.axes, list):
            raise TypeError("The axes must be a list of strings")
        self.axes_len = len(self.axes)

        self.variable = variable
        aim = np.zeros([1, self.axes_len]) if aim is None else aim
        self.aim = np.array(aim).reshape([-1, self.axes_len])
        self.weight = weight

        self.schedule = schedule if schedule else range(0)
        self.L = [] if L is None else self.arrange_L(L)

        if cross:
            self.cross = cross
            self.crossed = True
            cross_aim = cross_aim if cross_aim else np.zeros([1, self.axes_len])
            self.cross_aim = np.array(cross_aim).reshape([-1, self.axes_len])
            self.cross_L = [] if cross_L is None else self.arrange_L(cross_L)

        else:
            self.cross = self.variable
            self.crossed = False
            self.cross_aim = self.aim
            self.cross_L = self.L

    def arrange_L(self, L):
        """This function requires an updated schedule."""
        arranged_L = L if isinstance(L, list) else [L]

        if len(arranged_L) == 1:
            arranged_L = arranged_L * self.axes_len
        elif len(arranged_L) not in (self.axes_len, 0):
            raise IndexError(
                "'L' must have 0, 1 or len(axes) = {} "
                + "elements".format(self.axes_len)
            )

        if self.schedule and np.any([l.shape[-1] != self.t for l in arranged_L]):
            raise ValueError(
                "arrays in L must have {} ".format(self.t)
                + "columns, which is given by the 'schedule'."
            )
        return arranged_L

    def update(
        self, aim=None, weight=None, L=None, schedule=None, cross_aim=None, cross_L=None
    ):

        if schedule is not None:
            self.schedule = schedule

        if L is not None:
            self.L = self.arrange_L(L)
            if not self.crossed:
                self.cross_L = self.L

        if cross_L is not None:
            if self.crossed:
                self.cross_L = self.arrange_L(cross_L)
            else:
                raise KeyError("Trying to set cross_aim in a non-crossed cost")

        if aim is not None:
            self.aim = np.array(aim).reshape([-1, self.axes_len])
            if not self.crossed:
                self.cross_aim = self.aim

        if weight is not None:
            self.weight = weight

        if cross_aim is not None:
            if self.crossed:
                self.cross_aim = cross_aim
            else:
                raise KeyError("Trying to set cross_aim in a non-crossed cost")

    @property
    def t(self):
        if self.schedule:
            return self.schedule.stop - self.schedule.start
        return None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        text = ""
        for i, axis in enumerate(self.axes):
            text += "\n" + self.variable + axis
            spaces = len(self.variable + axis)
            for j, goal in enumerate(self.aim[:, i]):
                if j == 0:
                    text += (
                        " --> "
                        + str(goal)
                        + "\t\t~ weight = {} ~\n".format(self.weight)
                    )
                else:
                    text += " " * spaces + " --> " + str(self.aim[j, i]) + "\n"
        return text + "\n"
