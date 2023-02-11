import numpy as np
from tudatpy.kernel import constants as cons_tudat
from scipy.special import erfc, erf, erfinv, erfcinv

# from pint import UnitRegistry, set_application_registry
# ureg = UnitRegistry()
# set_application_registry(ureg)
# Q = ureg.Quantity
#
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     Q([])

def W2dB(x):
    return 10 * np.log10(x)

def W2dBm(x):
    return 10 * np.log10(x) + 30

def dB2W(x):
    return 10**(x/10)




#------------------------------------------------------------------------
#-------------------------------CONSTANTS--------------------------------
#------------------------------------------------------------------------

R_earth = 6367.0E3 #* ureg.meters
speed_of_light = 2.99792458E8 #* ureg.meter / ureg.second
q = 1.602176634*10**(-19) #* ureg.coulomb
h = 6.62607015*10**(-34) #* ureg.joule / ureg.hertz
k = 1.38*10**(-23) #* ureg.meter**2 * ureg.kg / (ureg.second**2 * ureg.kelvin)
mu_earth = 3.986004418E14 #* ureg.meter**3 / (ureg.second**2)
t_day = 24.0 #* ureg.days
Omega_t = 2*np.pi #* ureg.rad / t_day

#------------------------------------------------------------------------
#---------------------------DESIGN-VARIABLES-----------------------------
#------------------------------------------------------------------------

# Aircraft LCT
P_ac = 1.0 #* ureg.watt
eff_ac = 0.7
D_ac = 0.15 #* ureg.meter
Ar_ac = 1/4*np.pi*D_ac**2
angle_pe_ac = 3.6E-6 #* ureg.microrad
var_pj_ac = 3.3E-6 #* ureg.microrad
eff_quantum_ac = 0.7
T_s_ac = 70 #* ureg.kelvin
eff_coupling_ac = 0.8
FOV_ac = 1.0E-8 #* ureg.steradian  # 6.794E-5

detection_ac = "ADP" #PIN, Preamp, coherent (to be added)
mod_ac = "OOK-NRZ" #DPSK, PPM, QPSK
M_ac = 100.0 # AMPLIFICATION GAIN (For ADP & Preamp)
F_ac = 5.0 # NOISE FACTOR (For ADP)
BW_ac = 2.0E9 # 0.8 * data_rate (Given by Airbus)
delta_wavelength_ac = 5.0E-9 #* ureg.nm

# Spacecraft LCT
P_sc = 1.0 #* ureg.watt
D_sc = 0.15 #* ureg.meter
Ar_sc = 1/4*np.pi*D_sc**2
angle_pe_sc = 3.6E-6 #* ureg.microrad
var_pj_sc = 3.3E-6 #* ureg.microrad #0.5 / self.G
eff_sc = 0.5
eff_quantum_sc = 0.7
T_s_sc = 70 #* ureg.kelvin
eff_coupling_sc = 0.8
FOV_sc = 1.0E-8 #* ureg.steradian # 6.794E-5

detection_sc = "ADP" #PIN, Preamp, coherent (to be added)
mod_sc = "OOK-NRZ" #DPSK, PPM, QPSK
M_sc = 100.0 # AMPLIFICATION GAIN (For ADP & Preamp)
F_sc = 5.0 # NOISE FACTOR (For ADP)
BW_sc = 2.0E9 #* ureg.hertz #0.8 * data_rate
delta_wavelength_sc = 5.0E-9 #* ureg.nm
N_symb = 4


# Atmosphere
att_coeff = 0.025
I_sun = 0.2 #* ureg.watt / ureg.centimeter**2 / ureg.micrometer / ureg.steradian
I_sky = 0.2 #* ureg.watt / ureg.centimeter**2 / ureg.micrometer / ureg.steradian
n_index = 1.002

# Performance
BER_thres = 1.0E-6
latency = 10.0 #* ureg.second
h_tracking = 0.9 # During communications, about 10 percent of incoming light is reserved for tracking control

# Constellation architecture
constellation_type = "LEO_cons"
h_SC = 550.0E3 #* ureg.meter
inc_SC = 55.98 #* ureg.degree
number_of_planes = 10
number_sats_per_plane = 10

# Geometric constraints
method_AC = "straight"
method_SC = "tudat"
h_AC = 10.0E3 #* ureg.meters
vel_AC = np.array([0.0, 150.0, 0.0])
speed_AC = np.sqrt(vel_AC[0]**2 + vel_AC[1]**2 + vel_AC[2]**2)
lat_init_AC = 0.0 #* ureg.degrees
lon_init_AC = 0.0 #* ureg.degrees
elevation_min = np.deg2rad(20.0) #* ureg.degrees)
zenith_max = np.pi/2 - elevation_min

# Turbulence
PDF_type = "lognormal" #gamma-gamma (to be added)
turbulence_model="HVB"
wind_model_type = "Bufton"

# laser
wavelength = 1550.0E-9 #* ureg.nm
data_rate = 10.0E9  # 1 - 10 Gb/s (Given by Airbus)
v = (speed_of_light / wavelength)#.to('Hz').to_compact()
Ep = h * v
k_number = 2 * np.pi / wavelength

diff_limit_ac = wavelength / D_ac
diff_limit_sc = wavelength / D_sc
# angle_div_ac = 10.0E-6
# angle_div_sc = 10.0E-6
w0_ac = D_ac/2
w0_sc = D_sc/2
angle_div_ac  = 2.44 * diff_limit_ac #rad
angle_div_sc  = 2.44 * diff_limit_sc #rad
# w0_ac = np.pi * angle_div_ac / wavelength
# w0_sc = np.pi * angle_div_sc / wavelength

# Time scales
start_time = 0.0
end_time = 3600.0 * 6.0
step_size_dim1 = 1 / data_rate
step_size_dim2 = 10.0
















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

T = 2*np.pi * np.sqrt(a**3/mu) / 60

alpha = np.arccos(R/a)
delta_T = (np.pi - 2*alpha) / (2*np.pi) * T

print(T, np.rad2deg(np.pi-2*alpha), delta_T, T-delta_T)

R = 0.875
M = 100
F = 5

P0 = dB2W(-47.31)
P_norm = 0.5

P = P0 * P_norm

# noise_sh = 1.0408392967179929e-14
# noise_th = 6.624e-13
# noise_bg = 9.910742308650457e-20


noise_sh_norm = 2 * q * R * P_norm * BW_sc
noise_sh      = 2 * q * R * P * BW_sc
noise_th = 6.624e-13
noise_bg = 9.910742308650457e-20


# noise_th = 0.0
# noise_bg = 0.0

# PIN
SNR_norm = np.sqrt( (R*P_norm)**2 /noise_th )
SNR = np.sqrt( (R*P)**2 /noise_th )

#APD
# SNR_norm = np.sqrt( (R*P_norm*M)**2 /(M**2*F*(noise_sh_norm+noise_bg)+noise_th) )
# SNR = np.sqrt( (R*P*M)**2 /(M**2*F*(noise_sh+noise_bg)+noise_th) )

# print(M**2*F*noise_sh / (M**2*F*(noise_sh+noise_bg)+noise_th))
# print(noise_sh)
# print(SNR_ADP, SNR_PIN)


BER = 1/2 * erfc( np.sqrt(SNR)/np.sqrt(2) )
BER_norm = 1/2 * erfc( np.sqrt(SNR_norm)/np.sqrt(2) )



# print(SNR, SNR_norm, SNR_norm*P0)
# print(BER, BER_norm, BER_norm * erfc(np.sqrt(P0)))







