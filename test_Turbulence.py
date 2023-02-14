import unittest
from unittest import TestCase

import numpy as np
from constants import *
from Atmosphere import turbulence

steps = 100
heights = np.linspace(h_AC, h_SC, steps)
zenith_angles = np.flip(np.arange(30.0, 60.0, np.deg2rad(10.0)))

# zenith_angles_dim1 = np.array([zenith_max, 30])

ranges = np.sqrt(
    (h_SC - h_AC) ** 2 + 2 * h_SC * R_earth + R_earth ** 2 * np.cos(zenith_angles) ** 2) \
         - (R_earth + h_SC) * np.cos(zenith_angles)
slew_rate = 1 / np.sqrt((R_earth + h_SC) ** 3 / mu_earth)

Vg = 250.0

w0 = D_sc/2

samples = 1000

class Testturbulence(unittest.TestCase):
    def setUp(self):
        self.turb = turbulence(range=ranges)
        self.windspeed = self.turb.windspeed(slew_rate, Vg)
        self.Cn = self.turb.Cn_func()
        self.r0 = self.turb.r0_func(zenith_angles)


        # self.h_scint = self.turb.PDF(ranges, w0, zenith_angles=0.0, PDF_type="lognormal", steps=samples, effect="scintillation")


    def test_windspeed(self):
        self.assertTrue(np.ndim(self.windspeed) == 0)
        self.assertIsInstance(self.windspeed, float)
        # self.assertTrue(self.turb.Cn()[0] < 1.0E-14)

    def test_cn(self):
        # self.assertEqual(len(self.Cn), len())
        self.assertTrue(self.Cn[0] < 1.0E-14)
        self.assertTrue(self.Cn[0] > 1.0E-18)

    def test_r0(self):
        self.assertTrue(len(self.r0) == len(zenith_angles))
        self.assertIsInstance(self.r0, np.ndarray)
        print(self.r0)
        print(zenith_angles)
        for i in range(1, len(self.r0)):
            self.assertIsInstance(self.r0[i], float)
            # self.assertGreater(self.r0[i], self.r0[i-1])
            self.assertTrue(self.r0[i] < 0.1)
            self.assertGreater(self.r0[i], 0.01)


    # def PDF(self):
        # self.assertEqual(len(self.h_scint[0, :]), steps)
        # self.assertEqual(len(self.h_scint[:, 0]), len(zenith_angles))


# suite = unittest.TestLoader().loadTestsFromTestCase(Testturbulence)
# unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    unittest.main()