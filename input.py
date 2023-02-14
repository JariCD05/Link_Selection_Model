import numpy as np
from tudatpy.kernel import constants as cons_tudat
from matplotlib import pyplot as plt
from scipy.special import erfc, erf, erfinv, erfcinv


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
#--------------------------LCT-&-LASER-PARAMETERS------------------------
#------------------------------------------------------------------------

# laser
#----------------------------
wavelength = 1550.0E-9 # wavelength of laser (in m), REF: given by Airbus
data_rate = 10.0E9  # Data rate of the link (1 - 10 Gbit/s), REF: given by Airbus
v = (speed_of_light / wavelength) # frequency of laser (in Hz)
Ep = h * v # Energy per photon (in J/photon)
k_number = 2 * np.pi / wavelength # wave number of laser (in rad/m)

# Aircraft LCT
#----------------------------
P_ac = 1.0 # transmitter power (in Watt), REF: JSAT prototype
eff_ac = 0.7 # Overall efficiency of the optical system of the aircraft LCT, REF:
D_ac = 0.15 # Aperture diameter of aircraft LCT (in m)
angle_pe_ac = 3.6E-6 # pointing error of aircraft LCT (in rad)
var_pj_ac = 3.3E-6  # pointing jitter variance of aircraft LCT (in rad)
eff_quantum_ac = 0.7 # Quantum efficiency of photon detector aircraft LCT (no unit)
T_s_ac = 70 # Circuit temperature of aircraft LCT (in Kelvin)
eff_coupling_ac = 0.7 # One-mode fiber coupling efficiency of the receiving LCT, REF: Zephyr LCT, HAPS-TN-ADSN-0008, P.33
FOV_ac = 1.0E-8 # Field of View of aircraft LCT (in steradian), REF: Rudolf
h_tracking_ac = 0.9 # Tracking efficiency of aircraft LCT, REF: Remco

detection_ac = "ADP" # Detection method of aircraft LCT                         #PIN, Preamp, coherent (to be added)
mod_ac = "OOK-NRZ" # Modulation method  of aircraft LCT                         #DPSK, PPM, QPSK
M_ac = 100.0 # Amplification gain of of aircraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
F_ac = 5.0 # Noise factor of the aircraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
BW_ac = 0.8 * data_rate  # Beam width at receiver (in Hz) # 0.8 * data_rate (Given by Airbus)
delta_wavelength_ac = 5.0E-9  # Optical filter at aircraft LCT (in m)
R_L_ac = 50.0 # Load resistor of aircraft LCT (in case of PIN detection)
sensitivity_acquisition_ac = dB2W(-95) # Sensitivity of acquisition system of aircraft LCT, REF: Remco

# REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 13
diff_limit_ac = wavelength / D_ac # Diffraction limited divergence angle of aircraft LCT (in rad)
angle_div_ac  = 2.44 * diff_limit_ac # Real divergence angle of aircraft LCT (in rad)
w0_ac = D_ac/2 # beam waist of aircraft LCT (in m)
# # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021
# w0_ac = np.pi * angle_div_ac / wavelength

# Spacecraft LCT
#----------------------------
P_sc = 1.5
D_sc = 0.15 # Aperture diameter of spacecraft LCT (in m)
angle_pe_sc = 3.6E-6 # pointing error of spacecraft LCT (in rad)
var_pj_sc = 3.3E-6 # pointing jitter variance of spacecraft LCT (in rad)
eff_sc = 0.5 # Overall efficiency of the optical system of the spacecraft LCT, REF:
eff_quantum_sc = 0.7 # Quantum efficiency of photon detector spacecraft LCT (no unit)
T_s_sc = 70 # Circuit temperature of spacecraft LCT (in Kelvin)
eff_coupling_sc = 0.7 # One-mode fiber coupling efficiency of the receiving LCT, REF: Zephyr LCT, HAPS-TN-ADSN-0008, P.33
FOV_sc = 1.0E-8 # Field of View of aircraft LCT (in steradian), REF: Rudolf
h_tracking_sc = 0.9 # Tracking efficiency of spacecraft LCT, REF: Remco
# During communications, about 10 percent of incoming light is reserved for tracking control

detection_sc = "ADP" # Detection method of spacecraft LCT                         #PIN, Preamp, coherent (to be added)
mod_sc = "OOK-NRZ" # Modulation method  of spacecraft LCT                         #DPSK, PPM, QPSK
M_sc = 100.0 # Amplification gain of of spacecraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
F_sc = 5.0  # Noise factor of the spacecraft LCT (For ADP & Preamp), REF: Avalanche Photodiodes: A User's Guide, p.6
BW_sc = 2.0E9 #* ureg.hertz #0.8 * data_rate
delta_wavelength_sc = 5.0E-9 # Optical filter at spacecraft LCT (in m)
R_L_sc = 50.0 # Load resistor of aircraft LCT (in case of PIN detection)
sensitivity_acquisition_sc = dB2W(-95) # Sensitivity of acquisition system of spacecraft LCT, REF: Remco

# REF: AE4880 LASER SATELLITE COMMUNICATIONS I, R.SAATHOF, 2021, SLIDE 13
diff_limit_sc = wavelength / D_sc # Diffraction limited divergence angle of spacecraft LCT (in rad)
angle_div_sc  = 2.44 * diff_limit_sc # Real divergence angle of spacecraft LCT (in rad)
w0_sc = D_sc/2 # beam waist of spacecraft LCT (in m)
# # REF: AE4880 LASER SATELLITE COMMUNICATIONS II, R.SAATHOF, 2021
# w0_sc = np.pi * angle_div_sc / wavelength

#------------------------------------------------------------------------
#-----------------------GEOMETRIC-PARAMETERS-----------------------------
#------------------------------------------------------------------------

# Constellation architecture (only in case of 'tudat' method)
#----------------------------
constellation_type = "LEO_cons"
h_SC = 550.0E3 #* ureg.meter
inc_SC = 55.98 #* ureg.degree
number_of_planes = 10
number_sats_per_plane = 10

# Geometric constraints (only in case of 'straight flight' method)
#----------------------------
h_AC = 10.0E3 # Cruise altitude of aircraft (in m)
vel_AC = np.array([0.0, 150.0, 0.0]) # velocity vector of aircraft (in m/sec)
speed_AC = np.sqrt(vel_AC[0]**2 + vel_AC[1]**2 + vel_AC[2]**2) # speed magnitude of aircraft (in m/sec)
lat_init_AC = 0.0 # initial latitude (in degrees)
lon_init_AC = 0.0 # initial longitude (in degrees)
elevation_min = np.deg2rad(20.0) # minimum elevation angle between aircrat and spacecraft for an active link (in rad)
zenith_max = np.pi/2 - elevation_min # minimum zenith angle between aircrat and spacecraft for an active link (in rad)

# Time scales
#----------------------------
start_time = 0.0 # Start epoch of the simulation
end_time = 3600.0 * 6.0 # End epoch of the simulation
step_size_dim1 = 1.0e-1 # timestep of 0.1 msec for dimension 1 (turbulence simulation)
step_size_dim2 = 10.0 # timestep of 10 sec (ac/sc dynamics simulation)

#------------------------------------------------------------------------
#-----------------------ATMOSPHERIC-PARAMETERS---------------------------
#------------------------------------------------------------------------

att_coeff = 0.025 # Attenuation coefficient of the molecules and aerosols, REF:
I_sun = 0.2 #Solar irradiance (in W/cm^2/microm^2/steradian), REF: Hemmati
I_sky = 0.2 #Sky irradiance (in W/cm^2/microm^2/steradian), REF: Hemmati
n_index = 1.002 # Refractive index of atmosphere, REF: Google

#------------------------------------------------------------------------
#------------------------PERFORMANCE-METRICS-----------------------------
#------------------------------------------------------------------------

BER_thres = 1.0E-6
latency = 10.0 # Total required latency (in sec)

#------------------------------------------------------------------------
#-------------------------------METHODS----------------------------------
#------------------------------------------------------------------------

# Geometric methods
#----------------------------
method_AC = "straight" #Opensky (to be added)
method_SC = "tudat"

# Turbulence methods
#----------------------------
PDF_scintillation = "lognormal" #gamma-gamma (to be added)
PDF_beam_wander = "gaussian"
turbulence_model="HVB"
wind_model_type = "bufton"

# Attenuation methods
#----------------------------
method_att = 'standard_atmosphere'

# LCT methods
#----------------------------
PDF_pointing = 'gaussian'

#------------------------------------------------------------------------
#------------------------UPLINK-&-DOWNLINK-PARAMETERS--------------------
#------------------------------------------------------------------------

# Here, the link parameters are sorted in uplink parameters and downlink parameters.
# These will be used in the simulation of dimension 1 and dimension 2

link = "up"

if link == "up":
    P_t = P_ac
    eff_t = eff_ac
    eff_r = eff_sc
    eff_coupling = eff_coupling_sc
    h_tracking = h_tracking_sc
    D_t = D_sc
    D_r = D_ac
    w0 = w0_ac
    angle_pe_t = angle_pe_ac
    angle_pe_r = angle_pe_sc
    var_pj_t = var_pj_ac
    var_pj_r = var_pj_sc
    angle_div = angle_div_ac
    eff_quantum = eff_quantum_sc
    T_s = T_s_sc
    FOV_t = FOV_ac
    FOV_r = FOV_sc
    detection = detection_sc
    modulation = mod_sc
    M = M_sc
    noise_factor = F_sc
    BW = BW_sc
    delta_wavelength = delta_wavelength_sc
    R_L = R_L_sc
    sensitivity_acquisition = sensitivity_acquisition_sc


elif link == "down":
    P_t = P_sc
    eff_t = eff_sc
    eff_r = eff_ac
    eff_coupling = eff_coupling_ac
    h_tracking = h_tracking_ac
    D_t = D_ac
    D_r = D_sc
    w0 = w0_sc
    angle_pe_t = angle_pe_sc
    angle_pe_r = angle_pe_ac
    var_pj_t = var_pj_sc
    var_pj_r = var_pj_ac
    angle_div = angle_div_sc
    eff_quantum = eff_quantum_ac
    T_s = T_s_ac
    FOV_t = FOV_sc
    FOV_r = FOV_ac
    detection = detection_ac
    modulation = mod_ac
    M = M_ac
    noise_factor = F_ac
    BW = BW_ac
    delta_wavelength = delta_wavelength_ac
    R_L = R_L_sc
    sensitivity_acquisition = sensitivity_acquisition_ac













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







