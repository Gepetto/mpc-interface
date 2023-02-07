#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 16:21:03 2022

@author: nvilla
"""


import unittest

import numpy as np
from mpc_interface.dynamics import ControlSystem, ExtendedSystem, DomainVariable
from mpc_interface.body import Formulation
import mpc_interface.tools as use
from mpc_interface.restrictions import Constraint, Box
from mpc_interface.goal import Cost
from mpc_interface.combinations import LineCombo


class BodyTestCase(unittest.TestCase):
    def setUp(self):
        LIP = ControlSystem.from_name("J->CCC", tau=0.1, omega=3.5, axes=["_x", "_y"])
        LIP_ext = ExtendedSystem.from_cotrol_system(LIP, "x", 9)
        LIP_ext.define_output("DCM", {"CoM": 1, "CoM_dot": 1 / 3.5})

        E = use.plan_steps(9, 1, step_times=np.array([2, 5, 8]))[:, :, None]
        F = np.ones([9, 1, 1])

        steps = ExtendedSystem(
            ["Ds"],
            ["s"],
            "s",
            F,
            [E],
            ["_x", "_y"],
            how_to_update_matrices=use.update_step_matrices,
            time_variant=True,
        )

        bias = DomainVariable("n", 9, ["_x", "_y"])

        LIP_outputs = {
            "DCM_x": LineCombo({"CoM_x": 1, "CoM_dot_x": 1 / 3.5}),
            "DCM_y": LineCombo({"CoM_y": 1, "CoM_dot_y": 1 / 3.5}),
        }

        limit = Constraint("CoM_x", 4)
        limit2 = Constraint("s", 10, arrow=[1, 1], axes=["_x", "_y"], L=np.eye(9))
        cost1 = Cost("CoM_dot", aim=[1, 2], weight=10, axes=["_x", "_y"])
        cost2 = Cost("DCM_x", aim=2, weight=1)
        cost3 = Cost(
            "CoM", aim=[50, 50], weight=100, axes=["_x", "_y"], schedule=range(8, 9)
        )
        cost4 = Cost("DCM_x", aim=2, weight=1, cross="s_y", cross_L=np.eye(9))

        vertices = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        SSbox = Box.task_space("CoM", vertices, ["_x", "_y"])

        self.LIP_ext = LIP_ext
        self.steps = steps
        self.bias = bias
        self.limit = limit
        self.limit2 = limit2
        self.LIP_outputs = LIP_outputs
        self.cost1 = cost1
        self.cost2 = cost2
        self.cost3 = cost3
        self.cost4 = cost4
        self.SSbox = SSbox

        self.optimization_domain = [
            "x0_x",
            "x0_y",
            "Ds_x",
            "Ds_y",
            "CoM_dddot_x",
            "CoM_dddot_y",
        ]

        body = Formulation()
        body.incorporate_dynamics("steps", self.steps)
        body.incorporate_dynamics("LIP", self.LIP_ext)
        body.incorporate_dynamics("n", self.bias)
        body.incorporate_definitions(LIP_outputs)

        body.incorporate_constraint("kinematics", self.limit)
        body.incorporate_constraint("steppingArea", self.limit2)

        body.incorporate_goal("velocity", cost1)
        body.incorporate_goal("stability", cost2)
        body.incorporate_goal("terminal", cost3)
        body.incorporate_goal("crossed", cost4)

        body.incorporate_box("kine", self.SSbox)

        self.body = body

    def test_incorporations(self):
        self.assertTrue("steps" in self.body.dynamics.keys())
        self.assertTrue(isinstance(self.body.dynamics["steps"], ExtendedSystem))

        self.assertTrue("LIP" in self.body.dynamics.keys())
        self.assertTrue(isinstance(self.body.dynamics["LIP"], ExtendedSystem))

        self.assertTrue("n" in self.body.dynamics.keys())
        self.assertTrue(isinstance(self.body.dynamics["n"], DomainVariable))

        self.assertTrue("kinematics" in self.body.constraints.keys())
        self.assertTrue(isinstance(self.body.constraints["kinematics"], list))
        self.assertTrue(isinstance(self.body.constraints["kinematics"][0], Constraint))

        self.assertTrue("DCM_x" in self.body.definitions.keys())
        self.assertTrue(isinstance(self.body.definitions["DCM_y"], LineCombo))

    def test_qp_problem(self):
        self.body.identify_qp_domain(self.optimization_domain)

        self.assertEqual(self.body.optim_variables, self.optimization_domain)
        self.assertEqual(
            set(self.body.domain.keys()),
            set(self.body.optim_variables + self.body.given_variables),
        )

        self.body.make_preview_matrices()
        self.assertEqual(len(self.body.definitions), len(self.body.PM))
        self.assertEqual(self.body.PM["DCM_y"][0].shape, self.body.PM["CoM_y"][0].shape)

        collector = {
            var: np.array(list(rang)).reshape([-1, 1])
            for var, rang in self.body.given_ID.items()
        }
        given = self.body.arrange_given(collector)

        A, h = self.body.generate_qp_constraint(self.limit, given)

        self.assertEqual(h.shape, (9, 1))
        # given doesn't affect this inequality
        self.assertTrue((h == self.limit.bound()).all())
        self.assertEqual(A.shape, (9, self.body.optim_len))

        A2, h2 = self.body.generate_all_qp_constraints(given)
        self.assertEqual(h2.shape, (54, 1))
        self.assertEqual(A2.shape, (54, self.body.optim_len))

        # ## COSTS
        self.assertTrue(self.body.goals)

        Q1, q1 = self.body.generate_qp_cost(self.cost1, given)
        self.assertEqual(Q1.shape, (self.body.optim_len, self.body.optim_len))
        self.assertEqual(q1.shape, (self.body.optim_len, 1))

        cQ, cq = self.body.generate_qp_cost(self.cost4, given)
        self.assertEqual(cQ.shape, (self.body.optim_len, self.body.optim_len))
        self.assertEqual(cq.shape, (self.body.optim_len, 1))

        Q, q = self.body.generate_all_qp_costs(given)
        self.assertEqual(Q.shape, (self.body.optim_len, self.body.optim_len))
        self.assertEqual(q.shape, (self.body.optim_len, 1))


if __name__ == "__main__":
    unittest.main()

    # FOR MANUAL TEST
    o = BodyTestCase()
    o.setUp()
    o.test_incorporations()
    o.test_qp_problem()
