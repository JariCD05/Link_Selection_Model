
import Link_budget as LB
import Turbulence as turb
import Terminal as term
import Attenuation as atm
import constants as cons
import Constellation as SC
from Link_geometry import link_geometry

import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel import constants as cons_tudat


#------------------------------------------------------------------------
#---------------------------DESIGN-VARIABLES-----------------------------
#------------------------------------------------------------------------

# Aircraft LCT
P_ac = 0.1
eff_ac = 0.7
D_ac = 0.15 #m
Ar_ac = 1/4*np.pi*D_ac**2
angle_jit = 5E-6
eff_quantum_ac = 0.6
T_s_ac = 70
eff_coupling_ac = 0.8

detection_ac = "ADP" #PIN, Preamp, coherent (to be added)
mod_ac = "OOK-NRZ" #DPSK, PPM, QPSK
M_ac = 100.0 # AMPLIFICATION GAIN (For ADP & Preamp)
F_ac = 5.0 # NOISE FACTOR (For ADP)
BW_ac = 10.0E9 # 0.8 * data_rate (Given by Airbus)
delta_wavelength_ac = 5.0

# Spacecraft LCT
P_sc = 0.1
D_sc = 0.15 #m
Ar_sc = 1/4*np.pi*D_sc**2
eff_sc = 0.7
eff_quantum_sc = 0.85
T_s_sc = 70
eff_coupling_sc = 0.8

detection_sc = "ADP" #PIN, Preamp, coherent (to be added)
mod_sc = "OOK-NRZ" #DPSK, PPM, QPSK
M_sc = 100.0 # AMPLIFICATION GAIN (For ADP & Preamp)
F_sc = 5.0 # NOISE FACTOR (For ADP)
BW_sc = 10.0E9 #0.8 * data_rate
delta_wavelength_sc = 5.0
N_symb = 4


# Atmosphere
att_coeff = 0.025
I_sun = 0.02
I_sky = 0.02

# Performance
BER = 1.0E-6
latency = 10.0

# Geometric constraints
sat_setup = "LEO_1"
h_AC = 10.0E3
vel_AC = np.array([-250.0, 0.0, 0.0]) #m/s
speed_AC = np.linalg.norm(vel_AC)
lat_init_AC = 0.0
lon_init_AC = 0.0
elevation_min = 5.0

# Turbulence
PDF_type = "lognormal" #gamma-gamma (to be added)
turbulence_model="HVB"
wind_model_type = "Bufton"

# laser
wavelength = 1550E-9  # m
data_rate = 10.0E9  # 1 - 10 Gb/s (Given by Airbus)
v = cons.speed_of_light / wavelength
diff_limit_ac = wavelength / D_sc
diff_limit_sc = wavelength / D_ac

# Time scales
start_time = 0.0
end_time = cons_tudat.JULIAN_DAY
step_size_dim1 = 1 / data_rate
step_size_dim2 = 10.0

#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------

link_geometry = link_geometry(vel_AC = vel_AC,
                               lat_init_AC = lat_init_AC,
                               lon_init_AC = lon_init_AC,
                               simulation_start_epoch = start_time,
                               simulation_end_epoch= end_time,
                               sat_setup = sat_setup,
                               fixed_step_size= step_size_dim2,
                               elevation_min= elevation_min,
                               height_init_AC = h_AC)


#------------------------------------------------------------------------
#--------------------------HAND-OVER-STRATEGY----------------------------
#------------------------------------------------------------------------
# PICK A PLANE AND A SATELLITE WITHIN THAT PLANE
plane = 0
sat = 0
index = 0
phase = "Acquisition"

if phase == "Communication":
    P_ac = 0.1
    P_sc = 0.1
    FOV_ac = 1.0E-8 #6.794E-5
    FOV_sc = 1.0E-8  # 6.794E-5
    angle_pe = 0.0
    angle_div_ac = 2.44 * diff_limit_ac  # rad
    angle_div_sc = 2.44 * diff_limit_sc  # rad
elif phase == "Acquisition":
    FOV_ac = 1.0E-8  # 6.794E-5
    FOV_sc = 1.0E-8  # 6.794E-5
    angle_pe  = 0.0
    angle_div_ac = 2.44 * diff_limit_ac  # rad
    angle_div_sc = 2.44 * diff_limit_sc  # rad
    Gc = 0.0


# Initiate data of current aircraft and satellite
time = link_geometry.time_filtered[plane][sat]
ranges = link_geometry.ranges_filtered[plane][sat]
heights_AC = link_geometry.heights_AC[:len(time)]
heights_SC = link_geometry.heights_SC_filtered[plane][sat]
zenith_angles = link_geometry.zenith_filtered[plane][sat]
elevation_angles = link_geometry.elevation_filtered[plane][sat]
slew_rates = link_geometry.slew_rates_filtered[plane][sat]

heights = np.zeros((len(heights_SC), 100))
for i in range(len(heights_SC)):
    heights[i] = np.linspace(heights_AC[i], heights_SC[i], 100)

# index = np.argmin(ranges)
# range_current = ranges[index]
# zenith_current =zenith_angles[index]
# elev_current = elevation_angles[index]
# slew_current = slew_rates[index]
# heights_current = heights[index]

# link_geometry.plot("angles")
link_geometry.print(plane, sat, index)
#------------------------------------------------------------------------
#---------------------------INITIATE-TERMINALS---------------------------
#------------------------------------------------------------------------

terminal_ac = term.terminal_properties(D = D_ac,
                                       data_rate = data_rate,
                                       eff_quantum = eff_quantum_ac,
                                       BW = BW_ac,
                                       wavelength = wavelength,
                                       modulator = mod_ac,
                                       detection= detection_ac,
                                       FOV = FOV_ac,
                                       delta_wavelength=delta_wavelength_ac,
                                       M = M_ac,
                                       F = F_ac,
                                       BER_thres=BER,
                                       N_symb=N_symb
                                       )

terminal_sc = term.terminal_properties(D = D_sc,
                                       data_rate = data_rate,
                                       eff_quantum = eff_quantum_sc,
                                       BW = BW_sc,
                                       wavelength = wavelength,
                                       modulator = mod_sc,
                                       detection = detection_sc,
                                       FOV = FOV_sc,
                                       delta_wavelength=delta_wavelength_sc,
                                       M = M_sc,
                                       F = F_sc,
                                       BER_thres=BER
                                       )

# Threshold SNR at receiver
SNR_thres_sc, Q_thres_sc, Np_quantum_limit_sc = terminal_sc.threshold()

terminal_sc.print(type="terminal")
#------------------------------------------------------------------------
#--------------------------INITIATE-TURBULENCE---------------------------
#------------------------------------------------------------------------

# SET UP TURBULENCE MODEL
turbulence_uplink = turb.turbulence(
     range=ranges,
     heights=heights,
     wavelength=wavelength,
     D_r=D_sc,
     D_t=D_ac,
     angle_pe=angle_jit,
     att_coeff= att_coeff,
     link='up',
     h_cruise=heights_AC,
     h_sc = heights_SC,
     eff_coupling=eff_coupling_sc)

windspeed_rms = turbulence_uplink.windspeed(slew=slew_rates, Vg=speed_AC, wind_model_type=wind_model_type)
Cn_HVB    = turbulence_uplink.Cn(turbulence_model=turbulence_model)
r0 = turbulence_uplink.r0(zenith_angles)
T_WFE = turbulence_uplink.WFE(tip_tilt="YES")

# ------------------------------------------------------------------------
# --------------------------ATTENUATION-ANALYSIS--------------------------
# ------------------------------------------------------------------------

att = atm.attenuation(range_link=ranges, heights=heights, zenith_angles=zenith_angles, h_sc=heights_SC, h_ac=heights_AC)
T_ext = att.std_atm()

# att.print()
#------------------------------------------------------------------------
#-------------------------LINK-BUDGET-ANALYSIS---------------------------
#------------------------------------------------------------------------

# SET UP LINK BUDGET (POWER)
link_budget = LB.link_budget(ranges,
                         heights,
                         P_t=P_ac,
                         eff_t=eff_ac,
                         eff_r=eff_sc,
                         eff_coupling=eff_coupling_sc,
                         D_t=D_sc,
                         D_r=D_ac,
                         wavelength=wavelength,
                         angle_jit_t=angle_jit,
                         angle_jit_r=angle_jit,
                         att_coeff=att_coeff,
                         # r0=r0,
                         T_att=T_ext)

# SET UP LINK BUDGET (POWER)
P_r = link_budget.P_r()

# INCLUDE BEAM SPREAD EFFECT
link_budget.beam_spread(r0=r0)

# SET UP LINK BUDGET (GAUSSIAN BEAM INTENSITY)
radial_pos_t = np.linspace(-D_sc*3, D_sc*3, 50)
radial_pos_r = np.linspace(-link_budget.w_r*3, link_budget.w_r*3, 50)
I_t = link_budget.I_t(radial_pos_t)
I_r = link_budget.I_r(radial_pos_r, T_WFE=1.0, T_att=T_ext, beam_spread="YES")


link_budget.print(data_rate=data_rate, elevation=elevation_angles)
link_budget.plot(t=time, ranges=ranges)
# plt.show()
#---------------------------------------------------------------------------------------------
#--------------------------------------DIMENSION-1-SIMULATION---------------------------------
#----------------------------TURBULENCE----POINTING-ERROR----BER/SNR--------------------------
#---------------------------------------------------------------------------------------------

print('------------------------------------------------')
print('DIMENSION 1 (~GHz)')
print('------------------------------------------------')

# SET TIME INTERVAL OF THE SIMULATION OF DIMENSION 1
duration = 100.0
dt = 1.0E-4
t = np.arange(0.0, duration, dt)
steps = len(t)

N = 100
t_dim0 = np.linspace(0.0, 100000, N)
steps_dim0 = len(t_dim0)
print(steps)
print(steps_dim0)
print(1/0)

# SET UP LOOP VARIABLES
dim_1_results = {}

height_SC = 550.0E3
height_AC = 10.0E3
heights_dim1 = np.linspace(height_AC, height_SC, 100)
zenith_angles = np.deg2rad(np.arange(1.0, 90.0, 9))
zenith_angles_a = np.arcsin( 1/ 1.00027 * np.sin(zenith_angles) )
ranges_dim1          = np.sqrt((height_SC-height_AC)**2 + 2*height_SC*cons.R_earth  + cons.R_earth**2 * np.cos(zenith_angles)**2) \
                     - (cons.R_earth+height_AC)*np.cos(zenith_angles)
refractions     = np.array((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.01, 1.03, 1.05, 1.3))
slew_rate = 1 / np.sqrt((cons.R_earth+height_SC)**3 / cons.mu_earth)

P_r_dim1 = cons.dB2W(-50.0)
#------------------------------------------------------------------------
#--------------------------TURBULENCE-ANALYSIS---------------------------
#------------------------------------------------------------------------

# SET UP TURBULENCE MODEL
turbulence_uplink = turb.turbulence(
     range=ranges_dim1,
     heights=heights_dim1,
     wavelength=wavelength,
     D_r=D_sc,
     D_t=D_ac,
     angle_pe=angle_jit,
     att_coeff= att_coeff,
     link='up',
     h_cruise=height_AC,
     h_sc = height_SC,
     eff_coupling=eff_coupling_sc)

windspeed_rms = turbulence_uplink.windspeed(slew=slew_rate, Vg=speed_AC, wind_model_type=wind_model_type, dim=1)
Cn_HVB    = turbulence_uplink.Cn(turbulence_model=turbulence_model, dim=1)
r0 = turbulence_uplink.r0(zenith_angles, dim = 1)
T_WFE = turbulence_uplink.WFE(tip_tilt="YES")

var_rytov = turbulence_uplink.var_rytov(zenith_angles_a)
var_scint = turbulence_uplink.var_scint(PDF_type=PDF_type)

# Power and intensity at receiver
r_norm = turbulence_uplink.PDF(PDF_type=PDF_type, steps=steps, zenith_angles=zenith_angles_a)
P_turb = r_norm * P_r_dim1

turbulence_uplink.plot(t = t, plot="scint pdf")

turbulence_uplink.print(P_r = P_r_dim1)
#------------------------------------------------------------------------
#-------------------------POINTING-ERROR-ANALYSIS------------------------
#------------------------------------------------------------------------

# Pointing error distribution
var_pointing = 0.5 #0.5 / self.G
r_pe, T_pe = terminal_sc.pointing(boresight_error=0.0, dist_type="rayleigh", steps=steps)

#------------------------------------------------------------------------
#-----------------------------NOISE-BUDGET-------------------------------
#------------------------------------------------------------------------

noise_sh = terminal_sc.noise(noise_type="shot", P_r=P_turb)
noise_th = terminal_sc.noise(noise_type="thermal")
noise_bg = terminal_sc.noise(noise_type="background", P_r=P_turb, I_sun=I_sun)


terminal_sc.print(type="noise")
#------------------------------------------------------------------------
#-----------------------FLUCTUATING-SNR-&-BER----------------------------
#------------------------------------------------------------------------

# SNR & BER at receiver
SNR_turb = terminal_sc.SNR(P_r=P_turb, noise_sh=terminal_sc.noise_sh, noise_th=terminal_sc.noise_th, noise_bg=terminal_sc.noise_bg, F=F_sc, M=M_sc)
BER_turb = terminal_sc.BER()
# Photon count at receiver
Np_turb = terminal_sc.Np(P_r=P_turb)

terminal_sc.print(type="BER/SNR")
#------------------------------------------------------------------------
#------------------------------LINK-MARGIN-------------------------------
#------------------------------------------------------------------------

# Threshold at receiver
P_r_thres_sc = terminal_sc.P_r_thres()
Np_thres_sc = terminal_sc.Np_thres()

# Link Margin
link_margin = link_budget.link_margin(P_r_thres_sc, P_turb)
L_scint = np.mean(P_turb)/P_r_dim1

link_budget.print(type="link margin", L_scint=L_scint, P_turb=P_turb, Np_turb=Np_turb, P_r_thres=P_r_thres_sc, Np_thres=Np_thres_sc, P_r = P_r_dim1)
#------------------------------------------------------------------------
#---------------------SAVE-DIM-1-RESULTS-IN-DICTIONARY-------------------
#------------------------------------------------------------------------

for i in range(len(zenith_angles)):
    # print("Zenith angle          :", np.rad2deg(zenith_angle))
    # print("Refracted zenith angle: ", np.rad2deg(zenith_angle_a))
    # print("Range                 :", range_link/1000)
    # print("Refracted range       : ", range_link_r/1000)
    # print("Refraction            :", refractions[i])
    # print("BER                   :", np.mean(BER_turb[i]))
    # print("Link margin           :", np.mean(link_margin[i]))
    # print("---------------------------------------------------- ")

    dim_1_results[i] = [zenith_angles_a[i], Np_turb[i], P_turb[i]]
# ------------------------------------------------------------------------
#----------------------------------PLOT----------------------------------
#------------------------------------------------------------------------

link_geometry.plot()
# link_budget.plot(D_ac, D_ac, radial_pos_t, radial_pos_r)

turbulence_uplink.plot(t = t, plot="scint pdf")
turbulence_uplink.plot(t = t, plot="scint vs. zenith", zenith=zenith_angles_a)
terminal_sc.plot(t = t, plot="pointing")
terminal_sc.plot(t = t, plot="BER & SNR")

plt.show()