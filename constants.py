import numpy as np


def W2dB(x):
    return 10 * np.log10(x)

def W2dBm(x):
    return 10 * np.log10(x) + 30

def dB2W(x):
    return 10**(x/10)


R_earth = 6367.0E3
speed_of_light = 2.99792458E8
q = 1.602176634*10**(-19)
h = 6.62607015*10**(-34)
k = 1.38*10**(-23)
mu_earth = 3.986004418E14
wavelength = 1.55E-6
v = speed_of_light/wavelength
Ep = h * v
t_day = 24*3600.0
Omega_t = 2*np.pi / t_day


# Simple attenuation analysis
def Beer(a):
    T_att = np.exp(-a * (1.5))
    return T_att

# print(W2dB(Beer(0.15)))
# print(W2dB(Beer(40)))

# Simple uncertainty analysis
range1 = 500.0E3
range2 = range1 - 1.0E3
T_fs1 = wavelength / (4 * np.pi * range1) ** 2
T_fs2 = wavelength / (4 * np.pi * range2) ** 2

# print(T_fs2, T_fs1, 1-T_fs1/T_fs2)


# Simple PPB analysis
data_rate = 2.5E9
Pr = -58.43
# print(dB2W(Pr))
Ep = h * v
PPB = Pr / (Ep * data_rate)
# print(PPB)

R = 2574.73E3
a = R + 1500.0E3
m = 1.3452E23
G = 6.67430E-11
mu = m * G

T = 2*np.pi * np.sqrt(a**3/mu) / 3600

alpha = np.arccos(R/a)
delta_T = (np.pi - 2*alpha) / (2*np.pi) * T

