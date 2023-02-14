import unittest
import numpy as np
from Link_budget import link_budget
from input import *
steps = 100
range_float = 1200.0E3
range_string = 'range'

ranges = np.array([1200.0E3,1100.0E3,1000.0E3,900.0E3])
heights_SC = np.array([550.0E3, 550.0E3, 550.0E3, 550.0E3])
heights_AC = np.array([10.0E3, 10.0E3, 10.0E3, 10.0E3])

heights = np.zeros((len(heights_SC), 100))
for i in range(len(heights_SC)):
    heights[i] = np.linspace(heights_AC[i], heights_SC[i], 100)

Dt = 0.1
Dr = 0.1
radial_pos = np.linspace(-Dt/2, Dt/2, steps)


class Testlink_budget(unittest.TestCase):
    def setUp(self):
        self.link_budget = link_budget(P_t=P_ac, range=ranges, D_t=Dt, D_r=Dr)

    def test_exceptions(self):
        # self.link_budget_exception = link_budget(P_t=P_ac, range=range_float, D_t=Dt, D_r=Dr)
        self.assertRaises(TypeError, link_budget(P_t=P_ac, range=range_float, D_t=Dt, D_r=Dr))

    def test_p_r(self):
        P_r = self.link_budget.P_r_0_func()
        self.assertEqual(len(P_r), len(ranges))
        for i in P_r:
            self.assertIsInstance(i, float)

    def test_i_t(self):
        I_t = self.link_budget.I_r_0_func()
        self.assertEqual(len(I_t), len(ranges))
        for i in I_t:
            self.assertIsInstance(i, float)

    def test_variables(self):
        self.assertEqual(len(self.link_budget.T_fs), len(ranges))
        self.assertIsInstance(self.link_budget.T_fs, np.ndarray)



        # self.assertTrue(np.argmin(self.I_t) == 0 or np.argmin(self.I_t) == -1)
        # self.assertTrue(self.link_budget.I_t_0 > 0.95*I_t[np.argmax(I_t)] and self.link_budget.I_t_0 < 1.05*self.I_t[np.argmax(self.I_t)])
        # self.assertTrue(self.I_t[0]/self.link_budget.I_t_0 == 1/2.718281828459045**2)

    # def test_i_r(self):
    #     self.fail()
    #
    # def test_link_margin(self):
    #     self.fail()
    #
    # def test_beam_spread(self):
    #     self.fail()


if __name__ == '__main__':
    unittest.main()