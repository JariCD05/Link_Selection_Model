import numpy as np
import csv
import sqlite3

from input import *
from helper_functions import *
import Atmosphere as atm
import LCT
import Link_budget as LB
from matplotlib import pyplot as plt
from PDF import distributions


#---------------------------------------------------------------------------------------------
#---------------------------------DIMENSION-1-SIMULATION-SETUP--------------------------------
#---------------------------------------------------------------------------------------------

# Here, the time interval is set for the simulation of dimension 1.
duration = 100.0
dt = 1.0e-4
t = np.arange(0.0, duration, dt)
# This is the number of samples (1.000.000)
steps = len(t)

# Here, the time interval is set for the simulation of dimension 0. This is not yet implemented at this point.
# steps = 100
# N = 100
# t_dim0 = np.linspace(0.0, 100000, N)
# steps_dim0 = len(t_dim0)

# A sequence of elevation and zenith angles are generated here.
# These angles will be looped, where for each angle a Monte Carlo simulation is done with 1.000.000 samples.
zenith_angles_dim1 = np.flip(np.arange(0.0,zenith_max, np.deg2rad(1.0)))
elevation_angles_dim1 = np.pi/2 - zenith_angles_dim1
# These angles are corrected with a refractive index.
zenith_angles_dim1 = np.arcsin( 1/ n_index * np.sin(zenith_angles_dim1) )
# The range is computed, based on the aircraft height, spacecraft height and the zenith angles.
ranges_dim1 = np.sqrt((h_SC-h_AC)**2 + 2*h_SC*R_earth  + R_earth**2 * np.cos(zenith_angles_dim1)**2) \
                     - (R_earth+h_AC)*np.cos(zenith_angles_dim1)
# The slew rate is computed for computation of the Cn^2 model, inside the turbulence class.
slew_rate_dim1 = 1 / np.sqrt((R_earth+h_SC)**3 / mu_earth)

#------------------------------------------------------------------------
#-----------------------------------LCT----------------------------------
#------------------------------------------------------------------------
#
LCT = LCT.terminal_properties()

#------------------------------------------------------------------------
#-------------------------------TURBULENCE-------------------------------
#------------------------------------------------------------------------

# The turbulence class is initiated here. Inside the turbulence class, there are multiple functions that are run.
turbulence_dim1 = atm.turbulence(range=ranges_dim1, link=link)

# Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
# With Cn^2 and r0, the variances for scintillation and beam wander are computed
turbulence_dim1.windspeed(slew=slew_rate_dim1, Vg=speed_AC, wind_model_type=wind_model_type)
turbulence_dim1.Cn_func(turbulence_model=turbulence_model)
r0 = turbulence_dim1.r0_func(zenith_angles_dim1)
turbulence_dim1.var_rytov_func(zenith_angles_dim1)
turbulence_dim1.var_scint(PDF_type="lognormal", zenith_angles=zenith_angles_dim1)
turbulence_dim1.var_bw(D_t = D_ac, zenith_angles=zenith_angles_dim1)

# ------------------------------------------------------------------------
# -----------------------------ATTENUATION--------------------------------
# ------------------------------------------------------------------------
# The atmospheric attenutuation/extinction class is set up here.
attenuation = atm.attenuation(range_link=ranges_dim1)
h_att = attenuation.T_ext(zenith_angles=zenith_angles_dim1, method=method_att, steps=steps)


# ------------------------------------------------------------------------
# --------------------------STATIC-LINK-BUDGET----------------------------
# ------------------------------------------------------------------------
# The link budget class is initiated here.
link_budget_dim1 = LB.link_budget(ranges_dim1)

# The loss due to the turbulence beam spread effect is computed and taken as a static loss T_beamspread.
# This is then included in the link budget.
w_r = link_budget_dim1.beam_spread(r0=r0)
# The power and intensity at the receiver is computed from the link budget.
# All gains, losses and efficiencies are given in the link budget class.
P_r_0 = link_budget_dim1.P_r_0_func()
I_r_0 = link_budget_dim1.I_r_0_func()

#------------------------------------------------------------------------
#----------MONTE-CARLO-SIMULATION-FOR-ALL-STATISTICAL-FLUCTUATIONS-------
#-----------------------COMBINING-ALL-FLUCTUATIONS-----------------------
#------------------------------------------------------------------------

# First, the Monte Carlo simulations for pointing jitter are simulated with the PDF distributions chosen in input.py.
r_pj, h_pj = LCT.pointing(zenith_angles_dim1,
                          ranges_dim1,
                          w_r,
                          dist_type="gaussian",
                          steps=steps,
                          r0 = r0)

# Now, the Monte Carlo simulations for scintillation and beam wander are simulated with the PDF distributions chosen in input.py.
h_scint = turbulence_dim1.PDF(ranges_dim1,
                              w_r,
                              steps=steps,
                              zenith_angles=zenith_angles_dim1)

r_bw, h_bw = turbulence_dim1.PDF(ranges_dim1,
                                 w_r,
                                 steps=steps,
                                 zenith_angles=zenith_angles_dim1,
                                 effect="beam wander")

# The displacement fluctuations due to pointing error and beam wander are summed here.
r_tot = r_pj #+ r_bw
# h_tot = gaussian_beam(h_scint, np.abs(r_tot), D_ac, D_ac)

# Here the normalized power fluctuations are simply multiplied with each other.
# REF: REFERENCE POWER VECTORS FOR OPTICAL LEO DOWNLINK CHANNEL, D. GIGGENBACH ET AL. Fig.1.
h_tot = h_att * h_scint * h_pj

# Making sure that all negative values (if present) are converted to ZERO values
for i in range(len(h_tot)):
    for j in range(len(h_tot[0,:])):
        if h_tot[i, j] < 0.0:
            h_tot[i,j] = 0.0

#------------------------------------------------------------------------
#---------------------COMPUTING-Pr-&-SNR-&-BER-VALUES--------------------
#------------------------------------------------------------------------

# I_r = h_tot.transpose() * I_r_0
# w_0, w_r = beam_spread(D_sc, ranges_dim1)
# w_LT = beam_spread_turbulence(r0, w_0, w_r)
# r = np.linspace(0, D_sc/2, 1000)
# P_r = I_to_P(I_r, r, w_0, w_LT).transpose()


# The fluctuating signal power at the receiver is computed by multiplying the static power with the power fluctuations (h).
P_r = (h_tot.transpose() * P_r_0).transpose()
N_p = LCT.Np_func(P_r)


# All relevant noise types are computed with analytical equations.
# These equations are approximations, based on the assumption of a gaussian distribution for each noise type.
noise_sh = LCT.noise(noise_type="shot", P_r=P_r)
noise_th = LCT.noise(noise_type="thermal")
noise_bg = LCT.noise(noise_type="background", P_r=P_r, I_sun=I_sun)
noise_tot = LCT.noise(noise_type="total")

# The threshold SNR is computed with the analytical relationship between SNR and BER, based on the modulation scheme.
# The threshold P_r is then computed from the threshold SNR, the noise and the detection technique.
# Finally, the photon count Np (expressed in # photons / bit) is then computed with Np = Pr / (Ep * Dr * eff_quantum)
SNR_thres, P_r_thres, Np_thres = LCT.threshold()

# The actually received SNR is computed for each sample, as it directly depends on the noise and the received power P_r.
# The BER is then computed with the analytical relationship between SNR and BER, based on the modulation scheme.
SNR, SNR_avg = LCT.SNR_func(P_r)
BER, BER_avg = LCT.BER_func()
#------------------------------------------------------------------------
#-----------------------------FADE-STATISTICS----------------------------
#------------------------------------------------------------------------

# Here, the fading statistics are computed numerically.
# For each sample, the received power is evaluated and compared with the threshold power.
# When it is lower, it is counted as a fade. One fade is equal to the time step of 0.1 ms.
fades = np.zeros((len(zenith_angles_dim1), 3))
for i in range(0, len(zenith_angles_dim1)):
    # Number of fades
    fades[i, 0] = np.count_nonzero(P_r[i] < P_r_thres[i])
    # Fade time
    fades[i, 1] = fades[i, 0] * dt
    # Fractional fade time
    fades[i, 2] = fades[i, 1] / duration

# ------------------------------------------------------------------------
# ---------------------------SAVE-TO-DATABASE-----------------------------
# ------------------------------------------------------------------------

# This is a list of the performance variables (metrics) that are computed in dimension 1 and saved to the database
data_metrics = ['elevation', 'P_r', 'P_r_0', 'h_tot', 'h_att', 'h_scint', 'h_pj', 'h_bw', 'SNR', 'BER',
                'number of fades', 'fade time', 'fractional fade time',
                'P_r threshold', 'SNR threshold', 'Np threshold', 'Data rate', 'Np', 'noise']
number_of_metrics = len(data_metrics)

# A data array is created here and filled with all the computed performance variables
data = np.zeros((len(zenith_angles_dim1), number_of_metrics))
data[:, 0] = np.rad2deg(elevation_angles_dim1)

for i in range(0, len(zenith_angles_dim1)):
    data[i, 1] = np.mean(P_r[i])
    data[i, 2] = P_r_0[i]
    data[i, 3] = np.mean(h_tot[i])
    data[i, 4] = np.mean(h_att[i])
    data[i, 5] = np.mean(h_scint[i])
    data[i, 6] = np.mean(h_pj[i])
    data[i, 7] = np.mean(h_bw[i])
    data[i, 8] = SNR_avg[i]
    data[i, 9] = BER_avg[i]
    data[i, 10] = fades[i, 0]
    data[i, 11] = fades[i, 1]
    data[i, 12] = fades[i, 2]
    data[i, 13] = np.mean(P_r_thres[i])
    data[i, 14] = SNR_thres
    data[i, 15] = np.mean(Np_thres[i])
    data[i, 16] = data_rate
    data[i, 17] = np.mean(N_p[i])
    data[i, 18] = np.mean(noise_tot[i])

# Add row to data array with zero values
data = np.vstack((np.zeros(len(data_metrics)), data))

# Save data to sqlite3 database, 18 metrics are saved for each elevation angle (18+1 columns and N rows)
save_data(data)

#------------------------------------------------------------------------
#-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
#------------------------------------------------------------------------

# print(data[:, 0])
# print(data[:, 1])
# print(data[:, 2])
# print(data[:, 3])
# print(data[:, 5])
# print(data[:, 6])

LCT.plot(t = t, data=data, plot = "BER & SNR")
LCT.plot(t = t, data=data, plot = "data vs elevation")
turbulence_dim1.plot(t = t, plot='Cn')
LCT.plot(t = t, data=data, plot="pointing")
turbulence_dim1.plot(t = t, P_r=P_r, elevation=elevation_angles_dim1, plot="scint pdf")
turbulence_dim1.plot(t = t, plot='bw pdf')

turbulence_dim1.print()

# plt.show()