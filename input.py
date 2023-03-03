import json
import numpy as np
from tudatpy.kernel import constants as cons_tudat
from matplotlib import pyplot as plt
from scipy.special import erfc, erf, erfinv, erfcinv


def W2dB(x):
    return 10 * np.log10(x)
def dB2W(x):
    return 10**(x/10)
def W2dBm(x):
    return 10 * np.log10(x) + 30
def dBm2W(x):
    return 10**((x-30)/10)



# class input_parameters:
#     def __init__(self):

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
#--------------------------LCT-&-LASER-PARAMETERS------------------------
#------------------------------------------------------------------------

# laser
#----------------------------
wavelength = 1550.0E-9                          # wavelength of laser (in m), REF: given by Airbus
data_rate = 10.0E9                              # Data rate of the link (1 - 10 Gbit/s), REF: given by Airbus
v = (speed_of_light / wavelength)               # frequency of laser (in Hz)
Ep = h * v                                      # Energy per photon (in J/photon)
k_number = 2 * np.pi / wavelength               # wave number of laser (in rad/m)

# Coding
#----------------------------
N, K = 255, 223                                 # N is the total number of symbols per RS codeword, K is the total number of information bits per RS codeword
symbol_length = 8                               # Symbol length is the total number of bits within one symbol. The default is set to 8 bits (1 byte)

# Aircraft LCT
#----------------------------
P_ac = 10.0                                     # Optical power at aircraft LCT (in W), REF: Remco, ranges are typically between 5-10 W
clipping_ratio_ac = 2                           # Clipping ratio of aircraft telescope, REF: Excel sheet from Remco
obscuration_ratio_ac = 0.1                      # Obscuration ratio of aircraft telescope, REF: Coupling efficiency tech note from Airbus
angle_div_ac = 25.0E-6                          # Diffraction limited divergence angle
w0_ac = wavelength / (np.pi * angle_div_ac)     # Beam waist (1/e^2), REF: R.Saathof AE4880 slides I, P.19
D_ac = w0_ac * clipping_ratio_ac * 2            # Aperture diameter of aircraft LCT (in m), REF: Excel sheet from Remco
focal_length_ac = 0.05                          # Focal length of the aircraft LCT lens (in m), REF: Coupling efficiency tech note from Airbus

angle_pe_ac = 3.6E-6                            # Pointing error of aircraft LCT (in rad)
std_pj_ac = 3.3E-6                              # pointing jitter variance of aircraft LCT (in rad)
std_pj_spot_ac = 25.0E-6                        # Pointing jitter variance for spot of aircraft LCT (in rad), REF: Zephyr LCT, HAPS-TN-ADSN-0008, P.17
eff_quantum_ac = 1.0                            # Quantum efficiency of photon detector aircraft LCT (no unit)
T_s_ac = 70                                     # Circuit temperature of aircraft LCT (in Kelvin)
FOV_ac = 1.0E-8                                 # Field of View of aircraft LCT (in steradian), REF: Rudolf

eff_ac = 0.7                                    # Overall efficiency of the optical system of the aircraft LCT, REF:
h_splitting_ac = 0.9                            # Tracking efficiency of aircraft LCT, REF: Remco
eff_coupling_ac = 0.81 * 0.74                   # One-mode fiber coupling efficiency of the receiving LCT, REF: Zephyr LCT, HAPS-TN-ADSN-0008, P.33

detection_ac = "APD"                            # Detection method of aircraft LCT                         #PIN, Preamp, coherent (to be added)
mod_ac = "OOK-NRZ"                              # Modulation method  of aircraft LCT                         #DPSK, PPM, QPSK
symbol_length_ac = 8                            # Length of the data symbol that is modulated and coded at aircraft LCT
M_ac = 30.0                                     # Amplification gain of of aircraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
F_ac = 5.0                                      # Noise factor of the aircraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
BW_ac = 0.8 * data_rate                         # Optical bandwidth at aircraft LCT (in Hz) # 0.8 * data_rate (Given by Airbus)
delta_wavelength_ac = 5.0E-9                    # Optical filter at aircraft LCT (in m)
R_L_ac = 50.0                                   # Load resistor of aircraft LCT (in case of PIN detection)
sensitivity_acquisition_ac = dB2W(-95)          # Sensitivity of acquisition system of aircraft LCT, REF: Remco


# Spacecraft LCT
#----------------------------
P_sc = 10.0                                     # Optical power at spacecraft LCT (in W), REF: Remco, ranges are typically between 5-10 W
clipping_ratio_sc = 2                           # Clipping ratio of spacecraft telescope, REF: Excel sheet from Remco
obscuration_ratio_sc = 0.1                      # Obscuration ratio of spacecraft telescope, REF: Coupling efficiency tech note from Airbus
angle_div_sc = 25.0E-6                          # Diffraction limited divergence angle
w0_sc = wavelength / (np.pi * angle_div_sc)     # Beam waist (1/e^2), REF: R.Saathof AE4880 slides I, P.19
D_sc = w0_ac * clipping_ratio_sc * 2            # Aperture diameter of spacecraft LCT (in m), REF: Excel sheet from Remco
focal_length_sc = 0.05                          # Focal length of the spacecraft LCT lens (in m), REF: Coupling efficiency tech note from Airbus

angle_pe_sc = 3.6E-6                            # pointing error of spacecraft LCT (in rad)
std_pj_sc = 3.3E-6                              # pointing jitter variance of spacecraft LCT (in rad)
std_pj_spot_sc = 25.0E-6                        # Pointing jitter variance for spot of spacecraft LCT (in rad), REF: Zephyr LCT, HAPS-TN-ADSN-0008, P.17
eff_quantum_sc = 1.0                            # Quantum efficiency of photon detector spacecraft LCT (no unit)
T_s_sc = 70                                     # Circuit temperature of spacecraft LCT (in Kelvin)
FOV_sc = 1.0E-8                                 # Field of View of aircraft LCT (in steradian), REF: Rudolf
# During communications, about 10 percent of incoming light is reserved for tracking control

eff_sc = 0.7                                    # Overall efficiency of the optical system of the spacecraft LCT, REF:
h_splitting_sc = 0.9                            # Tracking efficiency of spacecraft LCT, REF: Remco
eff_coupling_sc = 0.81 * 0.74                   # One-mode fiber coupling efficiency of the receiving LCT, REF: Zephyr LCT, HAPS-TN-ADSN-0008, P.33

detection_sc = "APD"                            # Detection method of spacecraft LCT                         #PIN, Preamp, coherent (to be added)
mod_sc = "BPSK"                                 # Modulation method  of spacecraft LCT                         #DPSK, PPM, QPSK
symbol_length_sc = 8                            # Length of the data symbol that is modulated and coded at aircraft LCT
M_sc = 30.0                                     # Amplification gain of of spacecraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
F_sc = 5.0                                      # Noise factor of the spacecraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
BW_sc = 0.8 * data_rate                         # Optical bandwidth at spacecraft LCT (in Hz) # 0.8 * data_rate (Given by Airbus)
delta_wavelength_sc = 5.0E-9                    # Optical filter at spacecraft LCT (in m)
R_L_sc = 50.0                                   # Load resistor of aircraft LCT (in case of PIN detection)
sensitivity_acquisition_sc = dB2W(-95)          # Sensitivity of acquisition system of spacecraft LCT, REF: Remco

#------------------------------------------------------------------------
#-----------------------GEOMETRIC-PARAMETERS-----------------------------
#------------------------------------------------------------------------

# Constellation architecture (only in case of 'tudat' method)
#----------------------------
constellation_type = "LEO_cons"                 # Type of constellation (1 sat in LEO, 1 sat in GEO, LEO constellation)
h_SC = 550.0E3                                  # Initial altitude of the satellite(s)
inc_SC = 55.98 #* ureg.degree                   # Initial inclination of the satellite(s)
number_of_planes = 10                           # Number of planes within the constellation (if 1 sat: number_of_planes = 1)
number_sats_per_plane = 1                       # Number of satellites per plane within the constellation (if 1 sat: number_sats_per_plane = 1)

# Geometric constraints (only in case of 'straight flight' method)
#----------------------------
h_AC = 10.0E3                                   # Cruise altitude of aircraft (in m)
vel_AC = np.array([0.0, 150.0, 0.0])            # velocity vector of aircraft (in m/sec)
speed_AC = np.sqrt(vel_AC[0]**2 +
                   vel_AC[1]**2 +
                   vel_AC[2]**2)                # speed magnitude of aircraft (in m/sec)
lat_init_AC = 0.0                               # initial latitude (in degrees)
lon_init_AC = 0.0                               # initial longitude (in degrees)
elevation_min = np.deg2rad(20.0)                # minimum elevation angle between aircrat and spacecraft for an active link (in rad)
zenith_max = np.pi/2 - elevation_min            # minimum zenith angle between aircrat and spacecraft for an active link (in rad)

# Time scales
#----------------------------
start_time = 0.0                                # Start epoch of the simulation
end_time = 3600.0 * 6.0                         # End epoch of the simulation
step_size_dim1 = 1.0E-4                         # timestep of 0.1 msec for dimension 1 (turbulence simulation)
step_size_dim2 = 10.0                           # timestep of 10 sec (ac/sc dynamics simulation)

#------------------------------------------------------------------------
#-----------------------ATMOSPHERIC-PARAMETERS---------------------------
#------------------------------------------------------------------------

att_coeff = 0.025                               # Attenuation coefficient of the molecules and aerosols, REF:
I_sun = 0.5                                     # Solar irradiance (in W/cm^2/microm^2/steradian), REF: Hemmati
I_sky = 0.0                                     # Sky irradiance (in W/cm^2/microm^2/steradian), REF: Hemmati
n_index = 1.002                                 # Refractive index of atmosphere, REF: Google

#------------------------------------------------------------------------
#------------------------PERFORMANCE-METRICS-----------------------------
#------------------------------------------------------------------------

BER_thres = [1.0E-9, 1.0E-6, 1.0E-3]            # Minimum required Bit Error Rate
latency = 10.0                                  # Total required latency (in sec)

#------------------------------------------------------------------------
#-------------------------------METHODS----------------------------------
#------------------------------------------------------------------------

# Geometric methods
#----------------------------
method_AC = "straight"                          # Opensky (to be added)
method_SC = "tudat"
integrator = "Runge Kutta 78"

# Turbulence methods
#----------------------------
PDF_scintillation = "lognormal"                 # Gamma-Gamma (to be added)
PDF_beam_wander = "rayleigh"
PDF_AoA = "rayleigh"
turbulence_model="HVB"
wind_model_type = "bufton"

# Attenuation methods
#----------------------------
method_att = 'standard_atmosphere'

# LCT methods
#----------------------------
PDF_pointing = 'rice'

#------------------------------------------------------------------------
#------------------------UPLINK-&-DOWNLINK-PARAMETERS--------------------
#------------------------------------------------------------------------

# Here, the link parameters are sorted in uplink parameters and downlink parameters.
# These will be used in the simulation of dimension 1 and dimension 2

link = "up"

if link == "up":
    P_t = P_ac
    clipping_ratio = clipping_ratio_ac
    obscuration_ratio = obscuration_ratio_ac
    eff_t = eff_ac
    eff_r = eff_sc
    eff_coupling = eff_coupling_sc
    h_splitting = h_splitting_sc
    D_t = D_ac
    D_r = D_sc
    w0 = w0_ac
    angle_pe_t = angle_pe_ac
    angle_pe_r = angle_pe_sc
    std_pj_t = std_pj_ac
    std_pj_r = std_pj_sc
    angle_div = angle_div_ac
    eff_quantum = eff_quantum_sc
    T_s = T_s_sc
    FOV_t = FOV_ac
    FOV_r = FOV_sc
    detection = detection_sc
    modulation = mod_sc
    symbol_length = symbol_length_sc
    M = M_sc
    noise_factor = F_sc
    BW = BW_sc
    delta_wavelength = delta_wavelength_sc
    R_L = R_L_sc
    sensitivity_acquisition = sensitivity_acquisition_sc
    focal_length = focal_length_sc


elif link == "down":
    P_t = P_sc
    clipping_ratio = clipping_ratio_sc
    obscuration_ratio = obscuration_ratio_sc
    eff_t = eff_sc
    eff_r = eff_ac
    eff_coupling = eff_coupling_ac
    h_splitting = h_splitting_ac
    D_t = D_ac
    D_r = D_sc
    w0 = w0_sc
    angle_pe_t = angle_pe_sc
    angle_pe_r = angle_pe_ac
    std_pj_t = std_pj_sc
    std_pj_r = std_pj_ac
    angle_div = angle_div_sc
    eff_quantum = eff_quantum_ac
    T_s = T_s_ac
    FOV_t = FOV_sc
    FOV_r = FOV_ac
    detection = detection_ac
    modulation = mod_ac
    symbol_length = symbol_length_ac
    M = M_ac
    noise_factor = F_ac
    BW = BW_ac
    delta_wavelength = delta_wavelength_ac
    R_L = R_L_sc
    sensitivity_acquisition = sensitivity_acquisition_ac
    focal_length = focal_length_ac




dict_LCT_ac = {
    "wavelength": 1550E-9,
    "frequency": speed_of_light / wavelength,
    "k number": 2 * np.pi / wavelength,
    "P_t": 1.0,
    "clipping ratio": 2,
    "obscuration ratio": 0.1,
    "divergence": 25.0E-6,
    "w0": w0_ac,
    "D": w0_ac * clipping_ratio_ac * 2,

}

with open('LCT_ac.json', 'w') as outfile:
    json.dump(dict_LCT_ac, outfile)


with open('LCT_ac.json', 'r') as openfile:
    LCT_ac = json.load(openfile)

# for key, value in LCT_ac.items():
#     print(key, value)
# print(LCT_ac)





# w0 = 0.1
# print(w0)
# r0 = 0.1
# windspeed_rms = 180.0
# # h_AC = 10.0
# heights = np.linspace(h_AC, h_SC, 10000)
# Cn =  5.94E-53 * (windspeed_rms / 27) ** 2 * heights ** 10 * np.exp(-heights/ 1000.0) + \
#                   2.75E-16 * np.exp(-heights / 1500.0)
#
# L = 500000
#
# z_r = np.pi * w0 ** 2 * n_index / wavelength
# z_r1 = k_number * w0**2 / 2 * (1 + k_number * w0**2 / (2 * L))**(-1)
#
#
# # r0_down = (0.423 * k_number ** 2 * np.trapz(Cn, x=heights)) ** (-3 / 5)
# r0 = (0.423 * k_number ** 2 * np.trapz(Cn * ((heights[-1] - heights) / heights[-1]) ** (5/3), x=heights)) ** (-3 / 5)
# # sigma = 2.25 * k_number ** (7 / 6) * \
# #                          np.trapz(Cn * (heights - h_AC) ** (5/6) * ((heights[-1] - heights) / heights[-1]) ** (5/6), x=heights)
# sigma = 2.25 * k_number ** (7 / 6) * np.trapz(Cn * (heights - h_AC) ** (5/6), x=heights)
#
# heights_zr_lower = heights[heights<(h_AC+z_r)]
# Cn_zr_lower = Cn[heights<(h_AC+z_r)]
# heights_zr_higher = heights[heights>(h_AC+z_r)]
# Cn_zr_higher = Cn[heights>(h_AC+z_r)]
# r0_zr_higher = (0.423 * k_number ** 2 * np.trapz(Cn_zr_higher * ((h_SC - heights_zr_higher) / h_SC) ** (5/3), x=heights_zr_higher)) ** (-3 / 5)
# r0_zr_lower  = (0.423 * k_number ** 2 * np.trapz(Cn_zr_lower  * ((h_SC - heights_zr_lower)  / h_SC) ** (5/3), x=heights_zr_lower))  ** (-3 / 5)
# # print(heights_zr_lower[-1], heights_zr_higher[0], Cn_zr_lower[0], Cn_zr_higher[0])
# # print(r0_down, r0_up, r0_down/r0_up)
# # print(r0, r0_zr_lower, r0_zr_higher)
#
#
# D = 2**(3/2) * w0
# WFE = 1.03 * (D / r0) ** (5 / 3)
# S_phase = np.exp(-WFE)
# S_phase_NF = (1 + (5.56 - (4.84 / (1 + 0.04 * (w0/r0_zr_lower)**(5/3)))) * (w0/r0_zr_lower)**(5/3))**(-6/5)
# print(S_phase, W2dB(S_phase))
# print(S_phase_NF, W2dB(S_phase_NF))
#
#
#
#
#
# sigma_FF = 2.25 * k_number ** (7 / 6) * np.trapz(Cn_zr_higher * (heights_zr_higher - h_AC) ** (5/6) *
#                                                  ((h_SC - heights_zr_higher) / h_SC) ** (5/6), x=heights_zr_higher)
# sigma_NF = (0.77 * (w0/r0_zr_lower)**(-5/3) + np.exp(-5*(w0/r0_zr_lower)**(-5/3)))**(-1)
# sigma_phase = 1.03 * (D / r0) ** (5 / 3)
# print(w0, r0, r0_zr_lower, w0/r0, w0/r0_zr_lower, w0/r0_zr_higher)
# print(sigma_FF, sigma, sigma_NF, sigma_phase)
#
# S_phase_NF = (1 + 5.56*(w0/r0_zr_lower)**(5/3))**(-6/5)
# print(S_phase, S_phase_NF)


















# BEAM WANDER

# var_bw = 5.09 / (k_number**2 * r0**(5/3) * w0**(1/3))
# kappa0 = 30.0
# var_bw1 = 0.54 * (h_SC - h_AC)**2 * (wavelength / (2*w0))**2 * (2*w0 / r0)**(5/3) * \
#                       (1 - ((kappa0**2 * w0**2) / (1 + kappa0**2 * w0**2))**(1/6))

# print(var_bw*L, var_bw1, r0, w0)

# I = np.exp(-30**2/37**2)
# print(W2dB(I))