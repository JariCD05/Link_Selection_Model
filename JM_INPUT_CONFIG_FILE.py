import json
import numpy as np
from datetime import datetime
import time
import warnings
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import json

# Load the JSON config file
with open('config_files/USER_CASE_2 SDA_50sec_00000_28sat.json', 'r') as file:
    config = json.load(file)

# Access the configuration settings

# Constants
R_earth = config['constants']['R_earth']
speed_of_light = config['constants']['speed_of_light']
q = config['constants']['q']
h = config['constants']['h']
k = config['constants']['k']
mu_earth = config['constants']['mu_earth']
t_day = config['constants']['t_day']
Omega_t = config['constants']['Omega_t']
omega_earth = config['constants']['omega_earth']

# Numerical Simulation Setup
start_time = config['numerical_simulation_setup']['start_time']
end_time = config['numerical_simulation_setup']['end_time']
step_size_link = config['numerical_simulation_setup']['step_size_link']
step_size_SC = config['numerical_simulation_setup']['step_size_SC']
integrator = config['numerical_simulation_setup']['integrator']
step_size_channel_level = config['numerical_simulation_setup']['step_size_channel_level']
interval_channel_level = config['numerical_simulation_setup']['interval_channel_level']
frequency_filter_order = config['numerical_simulation_setup']['frequency_filter_order']

# Link Analysis Parameters
analysis = config['link_analysis_parameters']['analysis']
link_number = config['link_analysis_parameters']['link_number']
ac_LCT = config['link_analysis_parameters']['ac_LCT']
link = config['link_analysis_parameters']['link']

# LCT Laser Parameters
ac_LCT_general = config['LCT_laser_parameters']['ac_LCT_general']
wavelength_ac = ac_LCT_general['wavelength_ac']
data_rate_ac = ac_LCT_general['data_rate_ac']
P_ac = ac_LCT_general['P_ac']
D_ac = ac_LCT_general['D_ac']
clipping_ratio_ac = ac_LCT_general['clipping_ratio_ac']
obscuration_ratio_ac = ac_LCT_general['obscuration_ratio_ac']
M2_defocus_ac = ac_LCT_general['M2_defocus_ac']
M2_defocus_acquisition_ac = ac_LCT_general['M2_defocus_acquisition_ac']
angle_div_ac_acquisition = ac_LCT_general['angle_div_ac_acquisition']
focal_length_ac = ac_LCT_general['focal_length_ac']
angle_pe_ac = ac_LCT_general['angle_pe_ac']
std_pj_ac = ac_LCT_general['std_pj_ac']
std_pj_spot_ac = ac_LCT_general['std_pj_spot_ac']
eff_quantum_ac = ac_LCT_general['eff_quantum_ac']
T_s_ac = ac_LCT_general['T_s_ac']
FOV_ac = ac_LCT_general['FOV_ac']
FOV_ac_acquisition = ac_LCT_general['FOV_ac_acquisition']
eff_transmission_ac = ac_LCT_general['eff_transmission_ac']
WFE_static_ac = ac_LCT_general['WFE_static_ac']
WFE_static_acquisition_ac = ac_LCT_general['WFE_static_acquisition_ac']
h_splitting_ac = ac_LCT_general['h_splitting_ac']
detection_ac = ac_LCT_general['detection_ac']
mod_ac = ac_LCT_general['mod_ac']
M_ac = ac_LCT_general['M_ac']
F_ac = ac_LCT_general['F_ac']
delta_wavelength_ac = ac_LCT_general['delta_wavelength_ac']
R_L_ac = ac_LCT_general['R_L_ac']
sensitivity_acquisition_ac = ac_LCT_general['sensitivity_acquisition_ac']

sc_LCT = config['LCT_laser_parameters']['sc_LCT']
wavelength_sc = sc_LCT['wavelength_sc']
data_rate_sc = sc_LCT['data_rate_sc']
P_sc = sc_LCT['P_sc']
D_sc = sc_LCT['D_sc']
clipping_ratio_sc = sc_LCT['clipping_ratio_sc']
obscuration_ratio_sc = sc_LCT['obscuration_ratio_sc']
M2_defocus_sc = sc_LCT['M2_defocus_sc']
M2_defocus_acquisition_sc = sc_LCT['M2_defocus_acquisition_sc']
angle_div_sc = sc_LCT['angle_div_sc']
angle_div_sc_acquisition = sc_LCT['angle_div_sc_acquisition']
focal_length_sc = sc_LCT['focal_length_sc']
angle_pe_sc = sc_LCT['angle_pe_sc']
std_pj_sc = sc_LCT['std_pj_sc']
std_pj_spot_sc = sc_LCT['std_pj_spot_sc']
eff_quantum_sc = sc_LCT['eff_quantum_sc']
T_s_sc = sc_LCT['T_s_sc']
FOV_sc = sc_LCT['FOV_sc']
FOV_sc_acquisition = sc_LCT['FOV_sc_acquisition']
eff_transmission_sc = sc_LCT['eff_transmission_sc']
WFE_static_sc = sc_LCT['WFE_static_sc']
h_splitting_sc = sc_LCT['h_splitting_sc']
detection_sc = sc_LCT['detection_sc']
mod_sc = sc_LCT['mod_sc']
M_sc = sc_LCT['M_sc']
F_sc = sc_LCT['F_sc']
BW_sc = sc_LCT['BW_sc']
delta_wavelength_sc = sc_LCT['delta_wavelength_sc']
R_L_sc = sc_LCT['R_L_sc']
sensitivity_acquisition_sc = sc_LCT['sensitivity_acquisition_sc']

num_opt_head = config['LCT_laser_parameters']['num_opt_head']

# Aircraft Parameters
method_AC = config['aircraft_parameters']['method_AC']
h_AC = config['aircraft_parameters']['h_AC']
vel_AC = config['aircraft_parameters']['vel_AC']
lat_init_AC = config['aircraft_parameters']['lat_init_AC']
lon_init_AC = config['aircraft_parameters']['lon_init_AC']
aircraft_filename_load = config['aircraft_parameters']['aircraft_filename_load']
aircraft_filename_save = config['aircraft_parameters']['aircraft_filename_save']

# Constellation Parameters
constellation_data = config['constellation_parameters']['constellation_data']
method_SC = config['constellation_parameters']['method_SC']
SC_filename_load = config['constellation_parameters']['SC_filename_load']
SC_filename_save = config['constellation_parameters']['SC_filename_save']
constellation_type = config['constellation_parameters']['constellation_type']
h_SC = config['constellation_parameters']['h_SC']
inc_SC = config['constellation_parameters']['inc_SC']
number_of_planes = config['constellation_parameters']['number_of_planes']
number_sats_per_plane = config['constellation_parameters']['number_sats_per_plane']
variable_link_cost_const1 = config['constellation_parameters']['variable_link_cost_const1']
fixed_link_cost_const1 = config['constellation_parameters']['fixed_link_cost_const1']
TLE_filename_load = config['constellation_parameters']['TLE_filename_load']

# Link Selection Parameters
elevation_min_angle = config['link_selection_parameters']['elevation_min_angle']
elevation_threshold = config['link_selection_parameters']['elevation_threshold']
acquisition_time = config['link_selection_parameters']['acquisition_time']

# Atmospheric Parameters
scale_height = config['atmospheric_parameters']['scale_height']
att_coeff = config['atmospheric_parameters']['att_coeff']
I_sun = config['atmospheric_parameters']['I_sun']
I_sky = config['atmospheric_parameters']['I_sky']
n_index = config['atmospheric_parameters']['n_index']

# Methods Choices
margin_buffer = config['methods_choices']['margin_buffer']
desired_frac_fade_time = config['methods_choices']['desired_frac_fade_time']
BER_thres = config['methods_choices']['BER_thres']
coding = config['methods_choices']['coding']
latency_interleaving = config['methods_choices']['latency_interleaving']
N = config['methods_choices']['N']
K = config['methods_choices']['K']
symbol_length = config['methods_choices']['symbol_length']
turbulence_model = config['methods_choices']['turbulence_model']
wind_model_type = config['methods_choices']['wind_model_type']
turbulence_freq_lowpass = config['methods_choices']['turbulence_freq_lowpass']
jitter_freq_lowpass = config['methods_choices']['jitter_freq_lowpass']
jitter_freq2 = config['methods_choices']['jitter_freq2']
jitter_freq1 = config['methods_choices']['jitter_freq1']
method_att = config['methods_choices']['method_att']
method_clouds = config['methods_choices']['method_clouds']
dist_scintillation = config['methods_choices']['dist_scintillation']
dist_beam_wander = config['methods_choices']['dist_beam_wander']
dist_AoA = config['methods_choices']['dist_AoA']
dist_pointing = config['methods_choices']['dist_pointing']

# Performance Parameters
client_input_availability = config['performance_parameters']['client_input_availability']
client_input_BER = config['performance_parameters']['client_input_BER']
client_input_cost = config['performance_parameters']['client_input_cost']
client_input_data_transfer_latency = config['performance_parameters']['client_input_data_transfer_latency']
client_input_propagation_latency = config['performance_parameters']['client_input_propagation_latency']
client_input_throughput = config['performance_parameters']['client_input_throughput']

# Throughput Decay Rate
decay_rate = config['throughput_decay_rate']['decay_rate']

# Visualization
folder_path = config['visualization']['folder_path']

# Acquisition
T_acq = config['acquisition']['T_acq']

# Print out the loaded configurations to verify
#print(f"R_earth: {R_earth}")
#print(f"speed_of_light: {speed_of_light}")
#print(f"q: {q}")
#print(f"h: {h}")
#print(f"k: {k}")
#print(f"mu_earth: {mu_earth}")
#print(f"t_day: {t_day}")
#print(f"Omega_t: {Omega_t}")
#print(f"omega_earth: {omega_earth}")
#
#print(f"start_time: {start_time}")
#print(f"end_time: {end_time}")
#print(f"step_size_link: {step_size_link}")
#print(f"step_size_SC: {step_size_SC}")
#print(f"step_size_AC: {step_size_AC}")
#print(f"integrator: {integrator}")
#print(f"step_size_channel_level: {step_size_channel_level}")
#print(f"interval_channel_level: {interval_channel_level}")
#print(f"frequency_filter_order: {frequency_filter_order}")
#
#print(f"analysis: {analysis}")
#print(f"link_number: {link_number}")
#print(f"ac_LCT: {ac_LCT}")
#print(f"link: {link}")
#
#print(f"wavelength_ac: {wavelength_ac}")
#print(f"data_rate_ac: {data_rate_ac}")
#print(f"P_ac: {P_ac}")
#print(f"D_ac: {D_ac}")
#print(f"clipping_ratio_ac: {clipping_ratio_ac}")
#print(f"obscuration_ratio_ac: {obscuration_ratio_ac}")
#print(f"M2_defocus_ac: {M2_defocus_ac}")
#print(f"M2_defocus_acquisition_ac: {M2_defocus_acquisition_ac}")
#print(f"angle_div_ac_acquisition: {angle_div_ac_acquisition}")
#print(f"focal_length_ac: {focal_length_ac}")
#print(f"angle_pe_ac: {angle_pe_ac}")
#print(f"std_pj_ac: {std_pj_ac}")
#print(f"std_pj_spot_ac: {std_pj_spot_ac}")
#print(f"eff_quantum_ac: {eff_quantum_ac}")
#print(f"T_s_ac: {T_s_ac}")
#print(f"FOV_ac: {FOV_ac}")
#print(f"FOV_ac_acquisition: {FOV_ac_acquisition}")
#print(f"eff_transmission_ac: {eff_transmission_ac}")
#print(f"WFE_static_ac: {WFE_static_ac}")
#print(f"WFE_static_acquisition_ac: {WFE_static_acquisition_ac}")
#print(f"h_splitting_ac: {h_splitting_ac}")
#print(f"detection_ac: {detection_ac}")
#print(f"mod_ac: {mod_ac}")
#print(f"M_ac: {M_ac}")
#print(f"F_ac: {F_ac}")
#print(f"delta_wavelength_ac: {delta_wavelength_ac}")
#print(f"R_L_ac: {R_L_ac}")
#print(f"sensitivity_acquisition_ac: {sensitivity_acquisition_ac}")
#
#print(f"wavelength_sc: {wavelength_sc}")
#print(f"data_rate_sc: {data_rate_sc}")
#print(f"P_sc: {P_sc}")
#print(f"D_sc: {D_sc}")
#print(f"clipping_ratio_sc: {clipping_ratio_sc}")
#print(f"obscuration_ratio_sc: {obscuration_ratio_sc}")
#print(f"M2_defocus_sc: {M2_defocus_sc}")
#print(f"M2_defocus_acquisition_sc: {M2_defocus_acquisition_sc}")
#print(f"angle_div_sc: {angle_div_sc}")
#print(f"angle_div_sc_acquisition: {angle_div_sc_acquisition}")
#print(f"focal_length_sc: {focal_length_sc}")
#print(f"angle_pe_sc: {angle_pe_sc}")
#print(f"std_pj_sc: {std_pj_sc}")
#print(f"std_pj_spot_sc: {std_pj_spot_sc}")
#print(f"eff_quantum_sc: {eff_quantum_sc}")
#print(f"T_s_sc: {T_s_sc}")
#print(f"FOV_sc: {FOV_sc}")
#print(f"FOV_sc_acquisition: {FOV_sc_acquisition}")
#print(f"eff_transmission_sc: {eff_transmission_sc}")
#print(f"WFE_static_sc: {WFE_static_sc}")
#print(f"h_splitting_sc: {h_splitting_sc}")
#print(f"detection_sc: {detection_sc}")
#print(f"mod_sc: {mod_sc}")
#print(f"M_sc: {M_sc}")
#print(f"F_sc: {F_sc}")
#print(f"BW_sc: {BW_sc}")
#print(f"delta_wavelength_sc: {delta_wavelength_sc}")
#print(f"R_L_sc: {R_L_sc}")
#print(f"sensitivity_acquisition_sc: {sensitivity_acquisition_sc}")
#
#print(f"num_opt_head: {num_opt_head}")
#
#print(f"method_AC: {method_AC}")
#print(f"h_AC: {h_AC}")
#print(f"vel_AC: {vel_AC}")
#print(f"lat_init_AC: {lat_init_AC}")
#print(f"lon_init_AC: {lon_init_AC}")
#print(f"aircraft_filename_load: {aircraft_filename_load}")
#print(f"aircraft_filename_save: {aircraft_filename_save}")
#
#print(f"constellation_data: {constellation_data}")
#print(f"method_SC: {method_SC}")
#print(f"SC_filename_load: {SC_filename_load}")
#print(f"SC_filename_save: {SC_filename_save}")
#print(f"constellation_type: {constellation_type}")
#print(f"h_SC: {h_SC}")
#print(f"inc_SC: {inc_SC}")
#print(f"number_of_planes: {number_of_planes}")
#print(f"number_sats_per_plane: {number_sats_per_plane}")
#print(f"variable_link_cost_const1: {variable_link_cost_const1}")
#print(f"fixed_link_cost_const1: {fixed_link_cost_const1}")
#print(f"TLE_filename_load: {TLE_filename_load}")
#
#print(f"elevation_min_angle: {elevation_min_angle}")
#print(f"elevation_threshold: {elevation_threshold}")
#print(f"acquisition_time: {acquisition_time}")
#
#print(f"scale_height: {scale_height}")
#print(f"att_coeff: {att_coeff}")
#print(f"I_sun: {I_sun}")
#print(f"I_sky: {I_sky}")
#print(f"n_index: {n_index}")
#
#print(f"margin_buffer: {margin_buffer}")
#print(f"desired_frac_fade_time: {desired_frac_fade_time}")
#print(f"BER_thres: {BER_thres}")
#print(f"coding: {coding}")
#print(f"latency_interleaving: {latency_interleaving}")
#print(f"N: {N}")
#print(f"K: {K}")
#print(f"symbol_length: {symbol_length}")
#print(f"turbulence_model: {turbulence_model}")
#print(f"wind_model_type: {wind_model_type}")
#print(f"turbulence_freq_lowpass: {turbulence_freq_lowpass}")
#print(f"jitter_freq_lowpass: {jitter_freq_lowpass}")
#print(f"jitter_freq2: {jitter_freq2}")
#print(f"jitter_freq1: {jitter_freq1}")
#print(f"method_att: {method_att}")
#print(f"method_clouds: {method_clouds}")
#print(f"dist_scintillation: {dist_scintillation}")
#print(f"dist_beam_wander: {dist_beam_wander}")
#print(f"dist_AoA: {dist_AoA}")
#print(f"dist_pointing: {dist_pointing}")
#
#print(f"client_input_availability: {client_input_availability}")
#print(f"client_input_BER: {client_input_BER}")
#print(f"client_input_cost: {client_input_cost}")
#print(f"client_input_data_transfer_latency: {client_input_data_transfer_latency}")
#print(f"client_input_propagation_latency: {client_input_propagation_latency}")
#print(f"client_input_throughput: {client_input_throughput}")
#
#print(f"decay_rate: {decay_rate}")
#
#print(f"folder_path: {folder_path}")
#
#print(f"T_acq: {T_acq}")




#----------------------------------------------------------------------------------------------------

if ac_LCT == 'general':
    BW_ac = 0.8 * data_rate_ac                      # Optical bandwidth at aircraft LCT (in Hz) # 0.8 * data_rate           REF: Commonly used
    Be_ac = 0.5 * BW_ac                             # Electrical bandwidth at aircraft LCT (in Hz) # 0.5 * BW_sc            REF: Commonly used

# Spacecraft LCT
Be_sc = 0.5 * BW_sc                             # Electrical bandwidth at spacecraft LCT (in Hz) # 0.5 * BW_sc


#----------------------------------------------------------------------------------------------------
#--------------------------------------AIRCRAFT-PARAMETERS-------------------------------------------
#----------------------------------------------------------------------------------------------------

speed_AC = np.sqrt(vel_AC[0]**2 +
                   vel_AC[1]**2 +
                   vel_AC[2]**2)                # speed magnitude of aircraft (in m/sec)


step_size_AC = step_size_link     
#----------------------------------------------------------------------------------------------------
#------------------------------------CONSTELLATION-PARAMETERS----------------------------------------
#----------------------------------------------------------------------------------------------------

num_satellites = number_of_planes*number_sats_per_plane #defined to make sure this is the leading value while propagating the links
max_satellites = num_satellites
constellation_variable_link_cost = [variable_link_cost_const1] # the combined set of all variable link costs
constellation_fixed_link_cost = [fixed_link_cost_const1]



#----------------------------------------------------------------------------------------------------
#------------------------------------LINK-SELECTION-PARAMETERS---------------------------------------
#----------------------------------------------------------------------------------------------------
# LINK SELECTION constraints
elevation_min = np.deg2rad(elevation_min_angle)                 # minimum elevation angle between aircraft and spacecraft for start of an active link (in rad)
elevation_thres = np.deg2rad(elevation_threshold)                # maximum elevation angle between aircraft and spacecraft for start of an active link (in rad)
zenith_max = np.pi/2 - elevation_min            # minimum zenith angle between aircraft and spacecraft for start of an active link (in rad)
acquisition_time_steps = int(acquisition_time/step_size_link)

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


#-------perf param--------

weights = [client_input_availability, client_input_BER, client_input_cost,  client_input_data_transfer_latency, client_input_propagation_latency, client_input_throughput]

