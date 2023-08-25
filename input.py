import json
import numpy as np
from datetime import datetime
import time
import warnings
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")


#----------------------------------------------------------------------------------------------------
#-------------------------------------------------CONSTANTS------------------------------------------
#----------------------------------------------------------------------------------------------------

R_earth = 6367.0E3 #* ureg.meters
speed_of_light = 2.99792458E8 #* ureg.meter / ureg.second
q = 1.602176634*10**(-19) #* ureg.coulomb
h = 6.62607015*10**(-34) #* ureg.joule / ureg.hertz
k = 1.38*10**(-23) #* ureg.meter**2 * ureg.kg / (ureg.second**2 * ureg.kelvin)
mu_earth = 3.986004418E14 #* ureg.meter**3 / (ureg.second**2)
t_day = 24.0 #* ureg.days
Omega_t = 2*np.pi #* ureg.rad / t_day
omega_earth = 2 * np.pi / 86400.0


#----------------------------------------------------------------------------------------------------
#--------------------------------------NUMERICAL-SIMULATION-SETUP------------------------------------
#----------------------------------------------------------------------------------------------------

# Set-up of Macro-scale model
#----------------------------
start_time = 0.0                                  # Start epoch of the simulation
end_time = 3600 * 6 #3600.0   / 2                 # End epoch of the simulation
step_size_link = 6.0                              # Step size at which the macro-scale model is simulated
step_size_SC = 7.0                                # Numerical propagation time step of all SPACECRAFT in the constellation
step_size_AC = step_size_link                     # Numerical propagation time step of the AIRCRAFT
integrator = "Runge Kutta 4"

# Set-up of Micro-scale model
#----------------------------
step_size_channel_level = 1.0E-4                  # Sample size for the Monte Carlo time simulation of the micro-scale effects. Default is 0.1ms resolution
interval_channel_level = 5.0                      # Interval of the Monte Carlo time simulation. Default is 10s (verified for stability)
frequency_filter_order = 2


analysis    = 'total' # 'total' or 'time step specific'
link_number = 1 # If 'all': model simulates all links

ac_LCT = 'general' # 'general' or 'Zephyr'
link   = "up" # 'up' or 'down'


#----------------------------------------------------------------------------------------------------
#----------------------------------------LCT-&-LASER-PARAMETERS--------------------------------------
#----------------------------------------------------------------------------------------------------
if ac_LCT == 'general':
    # Aircraft LCT (GENERAL)
    #----------------------------
    wavelength_ac = 1553.0E-9                       # wavelength of laser (in m), REF: given by Airbus
    data_rate_ac = 2.5E9                            # Data rate of the link (1 - 10 Gbit/s), REF: given by Airbus
    P_ac = 20.0                                     # Optical power at aircraft LCT (in W)                                  REF: Remco, ranges are typically between 5-10 W
    D_ac = 0.08                                     # Aperture diameter of aircraft LCT (in m)                              REF: Zephyr link budget
    clipping_ratio_ac = 2.0                         # Clipping ratio of aircraft telescope                                  REF: Zephyr link budget
    obscuration_ratio_ac = 0.1                      # Obscuration ratio of aircraft telescope                               REF: Zephyr, HAPS-TN-ADSN-0008
    M2_defocus_ac = 1.0                             # M2 booster, defocusing of the beam at TX                              REF: Zephyr link budget
    M2_defocus_acquisition_ac = 11.66               # M2 booster, defocusing of the beam at TX, during acquistion           REF: Zephyr link budget
    angle_div_ac_acquisition = 300E-6               # Diffraction limited divergence angle during acquisition               REF: R.Saathof AE4880 slides I, P.19

    focal_length_ac = 0.12 #187mm (Zephyr)          # Focal length of the aircraft LCT lens (in m)                          REF: Zephyr, HAPS-TN-ADSN-0008
    angle_pe_ac = 4.0E-6 # 6.8E-6 (Zephyr)          # Pointing error of aircraft LCT (in rad)                               REF: Technical note Zephyr
    std_pj_ac = 3.4E-6   # 4.0E-6 (Zephyr)          # pointing jitter variance of aircraft LCT (in rad)                     REF: Technical note Zephyr
    std_pj_spot_ac = 25.0E-6                        # Pointing jitter variance for spot of aircraft LCT (in rad)            REF: Zephyr, HAPS-TN-ADSN-0008, P.17
    eff_quantum_ac = 1.0                            # Quantum efficiency of photon detector aircraft LCT (no unit)          REF: Commonly used
    T_s_ac = 300                                    # Circuit temperature of aircraft LCT (in Kelvin)                       REF: Commonly used
    FOV_ac = 1.0E-8 #3.5E-6 (Zephyr)                # Field of View of aircraft LCT (in steradian)
    FOV_ac_acquisition = 6.5E-3 #(Zephyr)           # Field of View of aircraft LCT (in steradian)                          REF: Rudolf

    eff_transmission_ac = 0.8 #1.0 dB (Zephyr)      # One-mode fiber nominal coupling efficiency of the receiving LCT       REF: Zephyr link budget
    WFE_static_ac = 100.0E-9                        # Static Wave front error loss in the fiber (without turbulence)        REF: Zephyr link budget
    WFE_static_acquisition_ac = 150.0E-9            # Static Wave front error in case of acquisition                        REF: Zephyr link budget
    h_splitting_ac = 0.9                            # Tracking efficiency of aircraft LCT                                   REF: Remco (10% of incoming light is assumed for tracking control)

    detection_ac = "Preamp"                         # Detection method of aircraft LCT ('PIN', 'Preamp', 'APD')             REF: Majumdar 2008
    mod_ac = "OOK-NRZ"                              # Modulation method  of aircraft LCT ('BDPSK', 'M-PPM', 'OOK-NRZ')      REF: Majumdar 2008
    M_ac = 150.0                                    # Amplification gain of of aircraft LCT (For ADP & Preamp)              REF: Avalanche Photodiodes: A User's Guide, p.6
    F_ac = 4.0                                      # Noise factor of the aircraft LCT (For ADP & Preamp)                   REF: Avalanche Photodiodes: A User's Guide, p.6
    BW_ac = 0.8 * data_rate_ac                      # Optical bandwidth at aircraft LCT (in Hz) # 0.8 * data_rate           REF: Commonly used
    Be_ac = 0.5 * BW_ac                             # Electrical bandwidth at aircraft LCT (in Hz) # 0.5 * BW_sc            REF: Commonly used
    delta_wavelength_ac = 5.0E-9                    # Optical filter at aircraft LCT (in m)                                 REF: Commonly used
    R_L_ac = 50.0                                   # Load resistor of aircraft LCT (in case of PIN detection)              REF: Commonly used
    sensitivity_acquisition_ac = 3.16227766016e-10  # Sensitivity of acquisition system of aircraft LCT                     REF: Remco (-95 dB -> 3.16e-10 W)

elif ac_LCT == 'Zephyr':
    # Aircraft LCT (Zephyr)
    #----------------------------
    wavelength_ac = 1553.0E-9                       # wavelength of laser (in m), REF: given by Airbus
    data_rate_ac = 2.5E9                            # Data rate of the link (1 - 10 Gbit/s), REF: given by Airbus
    P_ac = 1.2                                      # Optical power at aircraft LCT (in W)                                  REF: Remco, ranges are typically between 5-10 W
    D_ac = 0.04                                     # Aperture diameter of aircraft LCT (in m)                              REF: Zephyr link budget
    clipping_ratio_ac = 1.6                         # Clipping ratio of aircraft telescope                                  REF: Zephyr link budget
    obscuration_ratio_ac = 0.0                      # Obscuration ratio of aircraft telescope                               REF: Zephyr, HAPS-TN-ADSN-0008
    M2_defocus_ac = 1.1                             # M2 booster, defocusing of the beam at TX                              REF: Zephyr link budget
    M2_defocus_acquisition_ac = 7.56                # M2 booster, defocusing of the beam at TX, during acquistion           REF: Zephyr link budget
    # angle_div_ac_acquisition = 300E-6             # Diffraction limited divergence angle during acquisition               REF: R.Saathof AE4880 slides I, P.19

    focal_length_ac = 0.187                         # Focal length of the aircraft LCT lens (in m)                          REF: Zephyr, HAPS-TN-ADSN-0008
    angle_pe_ac = 6.8E-6                            # Pointing error of aircraft LCT (in rad)                               REF: Technical note Zephyr
    std_pj_ac = 4.0E-6                              # pointing jitter variance of aircraft LCT (in rad)                     REF: Technical note Zephyr
    std_pj_spot_ac = 25.0E-6                        # Pointing jitter variance for spot of aircraft LCT (in rad)            REF: Zephyr, HAPS-TN-ADSN-0008, P.17
    eff_quantum_ac = 1.0                            # Quantum efficiency of photon detector aircraft LCT (no unit)          REF: Commonly used
    T_s_ac = 300                                    # Circuit temperature of aircraft LCT (in Kelvin)                       REF: Commonly used
    FOV_ac = 3.5E-6                                 # Field of View of aircraft LCT (in steradian)
    FOV_ac_acquisition = 6.5E-3 #(Zephyr)           # Field of View of aircraft LCT (in steradian)                          REF: Rudolf

    eff_transmission_ac = 0.8 #1.0 dB (Zephyr)      # One-mode fiber nominal coupling efficiency of the receiving LCT       REF: Zephyr link budget
    WFE_static_ac = 100.0E-9                        # Static Wave front error loss in the fiber (without turbulence)        REF: Zephyr link budget
    WFE_static_acquisition_ac = 150.0E-9            # Static Wave front error in case of acquisition                        REF: Zephyr link budget
    h_splitting_ac = 0.9                            # Tracking efficiency of aircraft LCT                                   REF: Remco (10% of incoming light is assumed for tracking control)

    detection_ac = "Preamp"                         # Detection method of aircraft LCT ('PIN', 'Preamp', 'APD')             REF: Majumdar 2008
    mod_ac = "OOK-NRZ"                              # Modulation method  of aircraft LCT ('BDPSK', 'M-PPM', 'OOK-NRZ')      REF: Majumdar 2008
    M_ac = 350.0                                    # Amplification gain of of aircraft LCT (For ADP & Preamp)              REF: Avalanche Photodiodes: A User's Guide, p.6
    F_ac = 3.0                                      # Noise factor of the aircraft LCT (For ADP & Preamp)                   REF: Avalanche Photodiodes: A User's Guide, p.6
    BW_ac = 0.5 * data_rate_ac                      # Optical bandwidth at aircraft LCT (in Hz) # 0.8 * data_rate           REF: Commonly used
    Be_ac = 0.5 * BW_ac                             # Electrical bandwidth at aircraft LCT (in Hz) # 0.5 * BW_sc            REF: Commonly used
    delta_wavelength_ac = 5.0E-9                    # Optical filter at aircraft LCT (in m)                                 REF: Commonly used
    R_L_ac = 50.0                                   # Load resistor of aircraft LCT (in case of PIN detection)              REF: Commonly used
    sensitivity_acquisition_ac = 3.16227766016e-10  # Sensitivity of acquisition system of aircraft LCT                     REF: Remco (-95 dB -> 3.16e-10 W)
    #Single-fiber, low-noise pre-amplifier
    # Sensitivity: -47 for 2.5Gb/s (TESAT), -43 for 1.25Gb/s

# Spacecraft LCT
#----------------------------
wavelength_sc = 1553E-9 #or 1536E-9             # wavelength of laser (in m)                                            REF: SDA standard document
data_rate_sc  = 2.5E9                           # Data rate of the link (1 - 10 Gbit/s)                                 REF: SDA data sheet
P_sc = 20.0                                     # Optical power at spacecraft LCT (in W)
D_sc = 0.08                                     # Aperture diameter of aircraft LCT (in m)                              REF: Zephyr link budget
clipping_ratio_sc = 2.0                         # Clipping ratio of aircraft telescope                                  REF: Zephyr link budget
obscuration_ratio_sc = 0.1                      # Obscuration ratio of aircraft telescope                               REF: Zephyr, HAPS-TN-ADSN-0008
M2_defocus_sc = 1.0                             # M2 booster, defocusing of the beam at TX                              REF: Zephyr link budget
M2_defocus_acquisition_sc = 11.66               # M2 booster, defocusing of the beam at TX, during acquistion           REF: Zephyr link budget
angle_div_sc = 25.0E-6 #36.0E-6 (Zephyr)        # Diffraction limited divergence angle
angle_div_sc_acquisition = 300E-6               # REF: R.Saathof AE4880 slides I, P.19
focal_length_sc = 0.12                          # Focal length of the spacecraft LCT lens (in m)

angle_pe_sc = 3.6E-6                            # pointing error of spacecraft LCT (in rad)
std_pj_sc = 3.3E-6                              # pointing jitter varFiance of spacecraft LCT (in rad)
std_pj_spot_sc = 25.0E-6                        # Pointing jitter variance for spot of spacecraft LCT (in rad)
eff_quantum_sc = 0.7                            # Quantum efficiency of photon detector spacecraft LCT (no unit)
T_s_sc = 300                                    # Circuit temperature of spacecraft LCT (in Kelvin)
FOV_sc = 1.0E-8 #3.5E-3 (Zephyr)                # Field of View of aircraft LCT (in steradian)
FOV_sc_acquisition = 6.5E-3                     # Field of View of aircraft LCT (in steradian)

eff_transmission_sc = 0.8                       # One-mode fiber nominal coupling efficiency of the receiving LCT       REF: Zephyr link budget
WFE_static_sc = 100.0E-9                        # Static Wave front error loss in the fiber (without turbulence)        REF: Zephyr link budget
h_splitting_sc = 0.9                            # Tracking efficiency of aircraft LCT                                   REF: Remco (10% of incoming light is assumed for tracking control)

detection_sc = "Preamp"                         # Detection method of aircraft LCT ('PIN', 'Preamp', 'APD')
mod_sc = "BPSK"                                 # Modulation method  of aircraft LCT ('BPSK', 'M-PPM', 'OOK-NRZ')
M_sc  = 285.0                                   # Amplification gain of of spacecraft LCT (For ADP & Preamp)
F_sc  = 2.0                                     # Noise factor of the spacecraft LCT (For ADP & Preamp)
BW_sc = 2.5E9                                   # Optical bandwidth at spacecraft LCT (in Hz) # 0.8 * data_rate
Be_sc = 0.5 * BW_sc                             # Electrical bandwidth at spacecraft LCT (in Hz) # 0.5 * BW_sc
delta_wavelength_sc = 5.0E-9                    # Optical filter at spacecraft LCT (in m)
R_L_sc = 50.0                                   # Load resistor of aircraft LCT (in case of PIN detection)
sensitivity_acquisition_sc = 3.16227766016e-10  # Sensitivity of acquisition system of spacecraft LCT



#----------------------------------------------------------------------------------------------------
#--------------------------------------AIRCRAFT-PARAMETERS-------------------------------------------
#----------------------------------------------------------------------------------------------------
method_AC = "opensky" #"straight" or "opensky" # Choice of AIRCRAFT propagation, load from database (="opensky") or propagate a straight trajectory (="straight")
#--------------------In case of 'straight' method-----------------
h_AC   = 10.0E3                                # Cruise altitude of aircraft (in m)
vel_AC = np.array([0.0, 220.0, 0.0])           # velocity vector of aircraft (in m/sec)
speed_AC = np.sqrt(vel_AC[0]**2 +
                   vel_AC[1]**2 +
                   vel_AC[2]**2)                # speed magnitude of aircraft (in m/sec)
lat_init_AC = 78.5094                           # initial latitude (in degrees)
lon_init_AC = 60.54131                          # initial longitude (in degrees)
#--------------------In case of 'opensky' method-----------------
aircraft_filename_load = r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\traffic_trajectories\OSL_ENEV.csv"
aircraft_filename_save = r'C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\traffic_trajectories\SDA_30min_7sec_dt.json'



#----------------------------------------------------------------------------------------------------
#------------------------------------CONSTELLATION-PARAMETERS----------------------------------------
#----------------------------------------------------------------------------------------------------
constellation_data = 'NONE' #'NONE' or 'SAVE' or 'LOAD'
# In case of constellation_data='SAVE': These are the architecture parameters of the constellation to be saved
#---------------------------------------------
method_SC = "tudat"  # "TLE"
# Option to LOAD an existing json file with positional SC data (SC_filename_load)
# Or to propagate a new constellation and SAVE to a new json file (SC_filename_save)
SC_filename_load  = r'C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\constellation_states\SDA_30min_7sec_dt.json'
SC_filename_save  = r'C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\constellation_states\SDA_30min_7sec_dt.json'
#--------------------In case of 'tudat' method-----------------
constellation_type = "LEO_cons"                 # Type of constellation (1 sat in LEO, 1 sat in GEO, LEO constellation)
h_SC = 1200.0E3 #(SDA) or 550.0E3 (Starlink)       # Initial altitude of the satellite(s)
inc_SC = 85.0 #55.98 (Starlink) or 0.0 (GEO) or 80.0 (SDA)  # Initial inclination of the satellite(s)
number_of_planes = 2                            # Number of planes within the constellation (if 1 sat: number_of_planes = 1)
number_sats_per_plane = 14                      # Number of satellites per plane within the constellation (if 1 sat: number_sats_per_plane = 1)
#---------------------In case of 'TLE' method------------------
TLE_filename_load = r'C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\constellation_TLE_data\oneweb_tle.json'



#----------------------------------------------------------------------------------------------------
#------------------------------------LINK-SELECTION-PARAMETERS---------------------------------------
#----------------------------------------------------------------------------------------------------
# LINK SELECTION constraints
elevation_min = np.deg2rad(0.0)                 # minimum elevation angle between aircraft and spacecraft for start of an active link (in rad)
elevation_thres = np.deg2rad(5.0)              # maximum elevation angle between aircraft and spacecraft for start of an active link (in rad)
zenith_max = np.pi/2 - elevation_min            # minimum zenith angle between aircraft and spacecraft for start of an active link (in rad)
acquisition_time = 50.0  # seconds



#----------------------------------------------------------------------------------------------------
#-------------------------------------ATMOSPHERIC-PARAMETERS-----------------------------------------
#----------------------------------------------------------------------------------------------------

scale_height = 6600.0                           # Scale height of the atmosphere                                        REF: Vasylev 2019, Satellite-mediated quantum ..., Eq.12
att_coeff = 0.005                               # Attenuation coefficient of the molecules and aerosols                 REF: Vasylev 2019, Satellite-mediated quantum ..., Eq.12
I_sun = 0.5                                     # Solar irradiance (in W/cm^2/microm^2/steradian)                       REF: https://en.wikipedia.org/wiki/Solar_irradiance
I_sky = 0.0                                     # Sky irradiance (in W/cm^2/microm^2/steradian)                         REF: Hemmati
n_index = 1.002                                 # Refractive index of atmosphere                                        REF: Wikipedia

#----------------------------------------------------------------------------------------------------
#--------------------------------------------METHODS-&-CHOICES---------------------------------------
#----------------------------------------------------------------------------------------------------



# LCT model choices
#----------------------------
desired_frac_fade_time = 0.01
BER_thres = [1.0E-9, 1.0E-6, 1.0E-3]            # Minimum required Bit Error Rate, defined for an acceptable link
coding = 'no' # 'yes' or 'no'
# if coding = 'yes'
latency_interleaving = 1.0E-1                   # Interleaver length of coded bitframes (in seconds)
N, K = 255, 223                                 # N is the total number of symbols per RS codeword, K is the total number of information bits per RS codeword
symbol_length = 8                               # Symbol length is the total number of bits within one symbol. The default is set to 8 bits (1 byte)


# Turbulence model choices
#----------------------------
turbulence_model="Hufnagel-Valley"
wind_model_type = "Bufton"
turbulence_freq_lowpass = 1000.0                 # Defined cut-off frequency of refractive index fluctuations, estimated at 1 ms intervals. REF: Giggenbach 2018, PVGeT
jitter_freq_lowpass = 100.0                      # Defined cut-off frequency of platform microvibrations. REF: Zephyr technical note
jitter_freq2 = [100.0, 300.0]                    # Defined frequency peak of platform microvibrations. REF: Zephyr technical note
jitter_freq1 = [900.0, 1100.0]                   # Defined frequency peak of platform microvibrations. REF: Zephyr technical note

# Atmospheric model choises
#----------------------------
method_att = 'ISA profile'
method_clouds = 'static'

# Distributions
#----------------------------
dist_scintillation = "lognormal"                  # Distribution of the Monte Carlo sampling of scintillation fluctuations
dist_beam_wander = "rayleigh"                     # Distribution of the Monte Carlo sampling of angular displacement fluctuations of beam wander
dist_AoA = "rayleigh"                             # Distribution of the Monte Carlo sampling of angular displacement fluctuations of angle-of-arrival fluctuations
dist_pointing = "rayleigh"                        # Distribution of the Monte Carlo sampling of angular displacement fluctuations of mechanical jitter (TX and RX)




#------------------------------------------------------------------------
#------------------------UPLINK-&-DOWNLINK-PARAMETERS--------------------
#------------------------------------------------------------------------

# Here, the link parameters are sorted in uplink parameters and downlink parameters.
# These will be used in the simulation of dimension 1 and dimension 2

if link == "up":
    P_t                     = P_ac
    wavelength              = wavelength_ac
    data_rate               = data_rate_ac
    clipping_ratio          = clipping_ratio_ac
    obscuration_ratio       = obscuration_ratio_ac
    eff_transmission_t      = eff_transmission_ac
    eff_transmission_r      = eff_transmission_sc
    WFE_static_t            = WFE_static_ac
    WFE_static_r            = WFE_static_sc
    M2_defocus              = M2_defocus_ac
    M2_defocus_acq          = M2_defocus_acquisition_ac
    h_splitting             = h_splitting_sc
    D_t                     = D_ac
    D_r                     = D_sc
    angle_pe_t              = angle_pe_ac
    angle_pe_r              = angle_pe_sc
    std_pj_t                = std_pj_ac
    std_pj_r                = std_pj_sc
    eff_quantum             = eff_quantum_sc
    T_s                     = T_s_sc
    FOV_t                   = FOV_ac
    FOV_r                   = FOV_sc
    detection               = detection_sc
    modulation              = mod_sc
    M                       = M_sc
    noise_factor            = F_sc
    BW                      = BW_sc
    Be                      = Be_sc
    delta_wavelength        = delta_wavelength_sc
    R_L                     = R_L_sc
    sensitivity_acquisition = sensitivity_acquisition_sc
    focal_length            = focal_length_sc

elif link == "down":
    P_t                     = P_sc
    data_rate               = data_rate_sc
    wavelength              = wavelength_sc
    clipping_ratio          = clipping_ratio_sc
    obscuration_ratio       = obscuration_ratio_sc
    eff_transmission_t      = eff_transmission_sc
    eff_transmission_r      = eff_transmission_ac
    WFE_static_t            = WFE_static_sc
    WFE_static_r            = WFE_static_ac
    M2_defocus              = M2_defocus_sc
    M2_defocus_acq          = M2_defocus_acquisition_sc
    h_splitting             = h_splitting_ac
    D_t                     = D_sc
    D_r                     = D_ac
    angle_pe_t              = angle_pe_sc
    angle_pe_r              = angle_pe_ac
    std_pj_t                = std_pj_sc
    std_pj_r                = std_pj_ac
    eff_quantum             = eff_quantum_ac
    T_s                     = T_s_ac
    FOV_t                   = FOV_sc
    FOV_r                   = FOV_ac
    detection               = detection_ac
    modulation              = mod_ac
    M                       = M_ac
    noise_factor            = F_ac
    BW                      = BW_ac
    Be                      = Be_ac
    delta_wavelength        = delta_wavelength_ac
    R_L                     = R_L_sc
    sensitivity_acquisition = sensitivity_acquisition_ac
    focal_length            = focal_length_ac


#------------------------------------------------------------------------
#----------------------------DEPENDENT-PARAMETERS------------------------
#------------------------------------------------------------------------

v           = speed_of_light / wavelength         # frequency of laser (in Hz)                                                  REF: Wikipedia
k_number  = 2 * np.pi / wavelength                # wave number of laser (in rad/m)                                             REF: Wikipedia
w0        = D_t / clipping_ratio / 2              # Beam waist (1/e^2)                                                          REF: R.Saathof AE4880 slides I, P.19
angle_div = wavelength / ( np.pi * w0)            # Diffraction limited divergence angle during communication                   REF: R.Saathof AE4880 slides I, P.19
R         = eff_quantum * q / (h * v)             # Responsivity of the detector (conversion from power to electrica current)   REF: Wikipedia
orbital_period = 2 * np.pi * np.sqrt((R_earth+h_SC)**3 / mu_earth)
v_zenith  = 2 * np.pi * (R_earth+h_SC) / orbital_period
slew_rate_zenith = np.rad2deg(v_zenith / (h_SC - h_AC))