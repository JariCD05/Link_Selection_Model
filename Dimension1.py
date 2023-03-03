import numpy as np
import csv
import sqlite3


from scipy.stats import norm
from input import *
from helper_functions import *
import Atmosphere as atm
import LCT
import Link_budget as LB



#---------------------------------------------------------------------------------------------
#---------------------------------DIMENSION-1-SIMULATION-SETUP--------------------------------
#---------------------------------------------------------------------------------------------

# First, the setup parameters of dimension 1 are defined. These include the simulation time, number of samples, step size and cutoff frequencies for the filters.
# Here, the time interval is set for the simulation of dimension 1.
duration = 100
dt = step_size_dim1
t = np.arange(0.0, duration, dt)
sampling_frequency = 1/dt # 0.1 ms intervals
nyquist = sampling_frequency / 2
samples = len(t)

turbulence_frequency = 1000.0 #1/0.001 # 1 ms intervals
jitter_freq1 = [100.0, 300.0]
jitter_freq2 = [900.0, 1100.0]


#-------------------------------------------------------------------------
#-----------------------------------ITERABLES-----------------------------
#-------------------------------------------------------------------------
# A sequence of elevation and zenith angles are generated here.
# These angles will be looped, where for each angle a Monte Carlo simulation is done with 1.000.000 samples.
zenith_angles = np.flip(np.arange(0.0,zenith_max, np.deg2rad(1.0)))

elevation_angles_dim1 = np.pi/2 - zenith_angles
# These angles are corrected with a refractive index.
zenith_angles = np.arcsin( 1/ n_index * np.sin(zenith_angles) )
# The range is computed, based on the aircraft height, spacecraft height and the zenith angles.
ranges = np.sqrt((h_SC-h_AC)**2 + 2*h_SC*R_earth + R_earth**2 * np.cos(zenith_angles)**2) - (R_earth+h_AC)*np.cos(zenith_angles)
# The slew rate is computed for computation of the Cn^2 model, inside the turbulence class.
# For this, Kepler's third law is used (simplifying the constellation height to be constant)
slew_rate = 1 / np.sqrt((R_earth+h_SC)**3 / mu_earth)

# One iterable can be defined for which dimension 1 is simulated. The iterable by default is the vector of zenith angles.
# This iterable is used for the mapping of dimension 1 output with dimension 2 geometrical output.
iterable = zenith_angles

#------------------------------------------------------------------------
#-----------------------------------LCT----------------------------------
#------------------------------------------------------------------------
# The LCT class is initiated here. Inside the LCT class, there are multiple methods that are run later in the pipeline.
LCT = LCT.terminal_properties()

#------------------------------------------------------------------------
#-------------------------------TURBULENCE-------------------------------
#------------------------------------------------------------------------

# The turbulence class is initiated here. Inside the turbulence class, there are multiple methods that are run directly.
# Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
# With Cn^2 and r0, the variances for scintillation and beam wander are computed
turbulence_dim1 = atm.turbulence(ranges=ranges, link=link)


turbulence_dim1.windspeed_func(slew=slew_rate, Vg=speed_AC, wind_model_type=wind_model_type)
turbulence_dim1.Cn_func(turbulence_model=turbulence_model)
r0 = turbulence_dim1.r0_func(zenith_angles)
turbulence_dim1.var_rytov_func(zenith_angles)
turbulence_dim1.var_scint_func(zenith_angles=zenith_angles)
# The loss due to the turbulence beam spread effect is computed and taken as a static loss T_beamspread.
turbulence_dim1.beam_spread()
turbulence_dim1.var_bw(zenith_angles=zenith_angles)
turbulence_dim1.var_AoA(zenith_angles=zenith_angles)
# The loss due to wave front (phase) errors is computed and taken as a static loss defined by the Strehl ratio S.
turbulence_dim1.Strehl_ratio_func(tip_tilt="YES")
# ------------------------------------------------------------------------
# -----------------------------ATTENUATION--------------------------------
# ------------------------------------------------------------------------

# The atmospheric attenutuation/extinction class is set up here.
attenuation = atm.attenuation(range_link=ranges)
h_att = attenuation.T_ext(zenith_angles=zenith_angles, method=method_att)
# ------------------------------------------------------------------------
# --------------------------STATIC-LINK-BUDGET----------------------------
# ------------------------------------------------------------------------

# The link budget class is initiated here.
link_budget_dim1 = LB.link_budget(ranges=ranges,
                                  Strehl_ratio=turbulence_dim1.Strehl_ratio,
                                  w_ST=turbulence_dim1.w_ST,
                                  T_beamspread=turbulence_dim1.T_beamspread,
                                  T_att=h_att)
# The power and intensity at the receiver is computed from the link budget.
# All gains, losses and efficiencies are given in the link budget class.
P_r_0 = link_budget_dim1.P_r_0_func()
I_r_0 = link_budget_dim1.I_r_0_func()
# ------------------------------------------------------------------------
# ------------------------INITIALIZING-ALL-VECTORS------------------------
# ------------------------MONTE-CARLO-SIMULATIONS-------------------------
# ------------------------------------------------------------------------

# For each fluctuating variable, a vector is initialized with a standard normal distribution (std=1, mean=0)
# For all jitter related vectors (beam wander, angle-of-arrival, mechanical TX jitter, mechanical RX jitter), two variables are initialized for both X- and Y-components

angle_pj_t_X = norm.rvs(scale=1, loc=0, size=samples)
angle_pj_t_Y = norm.rvs(scale=1, loc=0, size=samples)
angle_pj_r_X = norm.rvs(scale=1, loc=0, size=samples)
angle_pj_r_Y = norm.rvs(scale=1, loc=0, size=samples)

# The turbulence vectors (beam wander and angle-of-arrival) differ for each iterable.
# Hence they are initialized seperately for each iterable and stored in one 2D array

h_scint     = np.empty((len(iterable), samples))                            # Should have 2D with size ( len of iterable list, # of samples )
angle_bw_X  = np.empty((len(iterable), samples))                            # Should have 2D with size ( len of iterable list, # of samples )
angle_bw_Y  = np.empty((len(iterable), samples))                            # Should have 2D with size ( len of iterable list, # of samples )
angle_aoa_X = np.empty((len(iterable), samples))                            # Should have 2D with size ( len of iterable list, # of samples )
angle_aoa_Y = np.empty((len(iterable), samples))                            # Should have 2D with size ( len of iterable list, # of samples )
for i in range(len(iterable)):
    h_scint[i]     = norm.rvs(scale=1, loc=0, size=samples)
    angle_bw_X[i]  = norm.rvs(scale=1, loc=0, size=samples)
    angle_bw_Y[i]  = norm.rvs(scale=1, loc=0, size=samples)
    angle_aoa_X[i] = norm.rvs(scale=1, loc=0, size=samples)
    angle_aoa_Y[i] = norm.rvs(scale=1, loc=0, size=samples)


# ------------------------------------------------------------------------
# -----------------------FILTERING-&-NORMALIZATION------------------------
# ------------------------------------------------------------------------

# All vectors are filtered and normalized, such that we end up with a standard normal distribution again, but now with a defined spectrum.
# The turbulence vectors are filtered with a low-pass filter with a default cut-off frequency of 1 kHz.
# The turbulence vectors are filtered with a band-pass filter with a default cut-off frequency ranges of [0.1- 0.2] Hz, [1.0- 1.1] Hz.

h_scint     = filtering(effect='scintillation', order=5, data=h_scint, f_cutoff=turbulence_frequency,
                    filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
angle_bw_X  = filtering(effect='beam wander', order=5, data=angle_bw_X, f_cutoff=turbulence_frequency,
                    filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
angle_bw_Y  = filtering(effect='beam wander', order=5, data=angle_bw_Y, f_cutoff=turbulence_frequency,
                    filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
angle_aoa_X = filtering(effect='angle of arrival', order=5, data=angle_aoa_X, f_cutoff=turbulence_frequency,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')
angle_aoa_Y = filtering(effect='angle of arrival', order=5, data=angle_aoa_Y, f_cutoff=turbulence_frequency,
                        filter_type='lowpass', f_sampling=sampling_frequency, plot='no')

angle_pj_t_X = filtering(effect='TX jitter', order=5, data=angle_pj_t_X, f_cutoff=jitter_freq1, f_cutoff_bandpass1=jitter_freq2,
                    filter_type='bandpass', f_sampling=sampling_frequency, plot='no')
angle_pj_t_Y = filtering(effect='TX jitter', order=5, data=angle_pj_t_Y, f_cutoff=jitter_freq1, f_cutoff_bandpass1=jitter_freq2,
                    filter_type='bandpass', f_sampling=sampling_frequency, plot='no')
angle_pj_r_X = filtering(effect='RX jitter', order=5, data=angle_pj_r_X, f_cutoff=jitter_freq1, f_cutoff_bandpass1=jitter_freq2,
                    filter_type='bandpass', f_sampling=sampling_frequency, plot='no')
angle_pj_r_Y = filtering(effect='RX jitter', order=5, data=angle_pj_r_Y, f_cutoff=jitter_freq1, f_cutoff_bandpass1=jitter_freq2,
                    filter_type='bandpass', f_sampling=sampling_frequency, plot='no')
# ------------------------------------------------------------------------
# --------------------------REDISTRIBUTE-SAMPLES--------------------------
# ------------------------------------------------------------------------

# Each vector is redistribution to a defined distribution. All distributions are defined in input.py
# For scintillation vector, the default distribution is LOGNORMAL.
# For the beam wander vectors (X- and Y-comp.) and the angle-of-arrival vectors (X- and Y-comp.), the default distribution is RAYLEIGH.
# For the TX jitter vectors (X- and Y-comp.) and the TX jitter vectors (X- and Y-comp.), the default distribution is RICE.

# The standard normal distribution of scintillation samples are converted to
h_scint = turbulence_dim1.PDF(data=h_scint,
                              steps=samples,
                              iterable=zenith_angles,
                              effect="scintillation")

angle_bw = turbulence_dim1.PDF(data=[angle_bw_X, angle_bw_Y],
                               steps=samples,
                               iterable=zenith_angles,
                               effect="beam wander")

angle_aoa = turbulence_dim1.PDF(data=[angle_aoa_X, angle_aoa_Y],
                                steps=samples,
                                iterable=zenith_angles,
                                effect="angle of arrival")

# First, the Monte Carlo simulations for pointing jitter are simulated with the PDF distributions chosen in input.py.
angle_pj_t = LCT.pointing(data=[angle_pj_t_X, angle_pj_t_Y],
                          steps=samples,
                          effect='TX jitter')

# First, the Monte Carlo simulations for pointing jitter are simulated with the PDF distributions chosen in input.py.
angle_pj_r = LCT.pointing(data=[angle_pj_r_X, angle_pj_r_Y],
                          steps=samples,
                          effect='RX jitter')


# The radial angle fluctuations for TX (mech. TX jitter and beam wander) are super-positioned
# # The radial angle fluctuations for RX (mech. RX jitter and angle-of-arrival) are super-positioned

# The super-positioned vectors for TX are projected over a Gaussian beam profile to obtain the loss fraction h_TX(t)
# The super-positioned vectors for RX are projected over a Airy disk profile to obtain the loss fraction h_RX(t)
angle_TX = angle_pj_t + angle_bw
angle_RX = angle_pj_r + angle_aoa
h_TX = gaussian_profile(angle_TX, angle_div)                                # Should have 2D with size ( len of iterable list, # of samples )
h_RX = airy_profile(angle_RX, D_r, focal_length)                            # Should have 2D with size ( len of iterable list, # of samples )

# The combined power vector is obtained by multiplying all three power vectors with each other (under the assumption of statistical independence between the three vectors)
# REF: REFERENCE POWER VECTORS FOR OPTICAL LEO DOWNLINK CHANNEL, D. GIGGENBACH ET AL. Fig.1.
h_tot = h_scint * h_TX * h_RX

# For each jitter related vector, the power vector is also computed separately for analysis of the separate contributions
h_bw = gaussian_profile(angle_bw, angle_div)                                # Should have 2D with size ( len of iterable list, # of samples )
h_pj_t = gaussian_profile(angle_pj_t, angle_div)                            # Should have 2D with size ( len of iterable list, # of samples )
h_pj_r = airy_profile(angle_pj_r, D_r, focal_length)                        # Should have 1D with size ( # of samples )
h_aoa = airy_profile( angle_aoa, D_r, focal_length)                         # Should have 2D with size ( len of iterable list, # of samples )

#------------------------------------------------------------------------
#------------------------COMPUTING-Pr-&-SNR-&-BER------------------------
#------------------------------------------------------------------------

# The fluctuating signal power at the receiver is computed by multiplying the static power (P_r_0) with the combined power vector (h_tot)
# The PPB is computed with P_r and data_rate, according to (A.MAJUMDAR, 2008, EQ.3.29, H.HEMMATI, 2004, EQ.4.1-1)
P_r = (h_tot.transpose() * P_r_0).transpose()
PPB = LCT.PPB_func(P_r, data_rate)

# All relevant noise types are computed with analytical equations.
# These equations are approximations, based on the assumption of a gaussian distribution for each noise type.
# The noise without any incoming signal is equal to
noise_sh = LCT.noise(noise_type="shot", P_r=P_r)                            # Should have 2D with size ( len of iterable list, # of samples )
noise_th = LCT.noise(noise_type="thermal")                                  # Should be a scalar
noise_bg = LCT.noise(noise_type="background", P_r=P_r, I_sun=I_sun)         # Should have 2D with size ( len of iterable list, # of samples )
noise_beat = LCT.noise(noise_type="beat")                                   # Should have 2D with size ( len of iterable list, # of samples )
noise_tot = LCT.noise(noise_type="total")                                   # Should have 2D with size ( len of iterable list, # of samples )

# The actually received SNR is computed for each sample, as it directly depends on the noise and the received power P_r.
# The BER is then computed with the analytical relationship between SNR and BER, based on the modulation scheme.
SNR = LCT.SNR_func(P_r=P_r, detection=detection)
BER = LCT.BER_func(modulation=modulation)

# The amount of erroneous bits is computed and stored as a vector.
# The total number of errors is computed by summing this vector
errors = np.cumsum(data_rate * dt * BER, axis=1)                            # Should have 2D with size ( len of iterable list, # of samples )
total_errors = errors[:, -1]                                                # Should have 1D with size ( len of iterable list )
total_bits   = data_rate * duration                                         # Should be a scalar

#------------------------------------------------------------------------
#---------------------------COMPUTING-THRESHOLD--------------------------
#------------------------------------------------------------------------

# The threshold SNR is computed with the analytical relationship between SNR and BER, based on the modulation scheme.
# The threshold P_r is then computed from the threshold SNR, the noise and the detection technique.
# The PPB is computed with the threshold P_r and data_rate, according to (A.MAJUMDAR, 2008, EQ.3.29, H.HEMMATI, 2004, EQ.4.1-1)
LCT.threshold(BER_thres=BER_thres)
#------------------------------------------------------------------------
#---------------------------------CODING---------------------------------
#------------------------------------------------------------------------

# Coding implementation is based on an analytical Reed-Solomon approximation, taken from (CCSDS Historical Document, 2006, CH.5.5, EQ.3-4)
# Input parameters for the coding scheme are total symbols per codeword (N), number of information symbols per code word (K) and symbol length

# The coding method in the LCT class simulates the coding scheme and computes the coded SER and BER from the channel BER and the coding input parameters
BER_coded = LCT.coding(duration=duration,
                                     K=K,
                                     N=N,
                                     samples=samples)
# The amount of erroneous bits is computed and stored as a vector.
# The total number of errors is computed by summing this vector
errors_coded = np.cumsum(data_rate * dt * BER_coded, axis=1)
total_errors_coded = errors_coded[:, -1]

# TO BE ADDED LATER
# number_of_bits = data_rate * duration
# step_size_codewords = K / data_rate
# mapping = list(np.arange(0, samples, step_size_codewords).astype(int))

# bits_per_codeword = number_of_bits / number_of_codewords
# BER_per_codeword = BER[0, mapping]
# error_bits_per_codeword = errors[0, mapping]

# codewords = t[mapping]

# SER_uncoded = np.zeros((len(zenith_angles_dim1), number_of_codewords))
# BER_uncoded = np.zeros((len(zenith_angles_dim1), number_of_codewords))
# SER_coded = np.zeros((len(zenith_angles_dim1), number_of_codewords))
# BER_coded = np.zeros((len(zenith_angles_dim1), number_of_codewords))

# for i in range(len(zenith_angles_dim1)):
#     print('coding test', E)
#     SER_uncoded[i, :] = SER[i, mapping]
#     BER_uncoded[i, :] = BER[i, mapping]
    # for j in range(number_of_codewords):
    #     SER_coded[i,j] = SER[i, j] * sum(binom(N - 1, k) * SER[i,j] ** k * (1 - SER[i,j]) ** (N - k - 1) for k in range(E, N - 1))


# Interleaving

#------------------------------------------------------------------------
#-----------------------------FADE-STATISTICS----------------------------
#------------------------------------------------------------------------

# Here, the fading statistics are computed numerically.
# For each sample, the received power is evaluated and compared with the threshold power for all 3 thresholds (BER = 1.0E-9, 1.0E-6, 1.0E-3)
# When P_r is lower than P_r_thres, it is counted as a fade. One fade time is equal to the time step of 0.1 ms.

fades = np.zeros((len(BER_thres), len(iterable)))
# Fade times for the BER threshold of 1.0E-9
fades[0] = np.count_nonzero((P_r < LCT.P_r_thres[0]), axis=1) * dt             # Should have 1D with size ( len of iterable list )
# Fade times for the BER threshold of 1.0E-6
fades[1] = np.count_nonzero((P_r < LCT.P_r_thres[0]), axis=1) * dt             # Should have 1D with size ( len of iterable list )
# Fade times for the BER threshold of 1.0E-3
fades[2] = np.count_nonzero((P_r < LCT.P_r_thres[0]), axis=1) * dt             # Should have 1D with size ( len of iterable list )

# ------------------------------------------------------------------------
# ---------------------------SAVE-TO-DATABASE-----------------------------
# ------------------------------------------------------------------------
# Here, a selection is made of relevant output data, which is first written as a list of strings, then stored in a array with name 'data'
# Then, one row is added to the top of this array to account for a zero value (needed when mapping to dim 2) and finally data is saved to sqlite3 database
# This is a list of the performance variables (metrics) that are saved in a database
data_metrics = ['elevation',
                'P_r',
                'PPB',
                'h_tot',
                'h_scint',
                'h_TX',
                'h_RX',
                'BER max',
                'fade time (BER=1.0E-9)',
                'fade time (BER=1.0E-6)',
                'fade time (BER=1.0E-3)',
                'Number of error bits (uncoded)',
                'Number of error bits (coded)']
number_of_metrics = len(data_metrics)

# A data array is created here and filled with all the relevant performance output variables
data = np.zeros((len(iterable), number_of_metrics))
data[:, 0] = np.rad2deg(elevation_angles_dim1)
data[:, 1] = P_r.mean(axis=1)
data[:, 2] = PPB.mean(axis=1)
data[:, 3] = h_tot.mean(axis=1)
data[:, 4] = h_scint.mean(axis=1)
data[:, 5] = h_TX.mean(axis=1)
data[:, 6] = h_RX.mean(axis=1)
data[:, 7] = BER.max(axis=1)
data[:, 8] = fades[0]
data[:, 9] = fades[1]
data[:,10] = fades[2]
data[:,11] = total_errors
data[:,12] = total_errors_coded

# Add row to data array with zero values
data = np.vstack((np.zeros(len(data_metrics)), data))

# Save data to sqlite3 database, 12 metrics are saved for each elevation angle (12+1 columns and N rows)
# UNCOMMENT WHEN YOU WANT TO SAVE DATA TO DATABASE
save_data(data)

# ------------------------------------------------------------------------
# ---------------UPDATE-LINK-BUDGET-WITH-FLUCTUATIONS-AND-Pr--------------
# ------------------------------------------------------------------------
link_budget_dim1.dynamic_contributions(performance_output=data)
# Initiate tracking. This method makes sure that a part of the incoming light is reserved for the tracking system.
# In the link budget this is translated to a fraction (standard is set to 0.9) of the light is subtracted from communication budget and used for tracking budget
link_budget_dim1.P_r_tracking_func()
link_budget_dim1.link_margin(P_r_thres=LCT.P_r_thres, PPB_thres=LCT.PPB_thres)



#------------------------------------------------------------------------
#-----------------------------PDF-VERIFICATION---------------------------
#------------------------------------------------------------------------
# Verification of the sampled data (comparison with theoretical PDFs)
test_index = [0, 2, 3]
# turbulence_dim1.test_PDF(effect = "scintillation", index = test_index)
# turbulence_dim1.test_PDF(effect = 'beam wander', index = test_index)
# turbulence_dim1.test_PDF(effect = 'angle of arrival', index = test_index)
# LCT.test_PDF(effect = 'TX jitter', index = test_index)
# LCT.test_PDF(effect = 'RX jitter', index = test_index)
# plt.show()

#------------------------------------------------------------------------
#-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
#------------------------------------------------------------------------
plot_index = 1
print('plot test1')
# LCT.plot(t = t, index=plot_index, plot = "BER & SNR")
# LCT.plot(t = t, data=data, plot = "data vs elevation")
# turbulence_dim1.plot(t = t, plot='Cn')
# turbulence_dim1.plot(t = t, plot='wind')
# LCT.plot(index = plot_index, t = t, data=data, plot="pointing")
# turbulence_dim1.plot(t = t, P_r=P_r, index = plot_index, elevation=elevation_angles_dim1, plot="beam wander")
# turbulence_dim1.plot(t = t, zenith=zenith_angles_dim1, plot="scint vs. elevation")
# turbulence_dim1.plot(t = t, P_r=P_r, index = plot_index, elevation=elevation_angles_dim1, plot="scint pdf")
# turbulence_dim1.plot(t = t, index = plot_index, plot='bw pdf')
# link_budget_dim1.plot(elevation=elevation_angles_dim1, index = [0, 2, 3, 5], type="gaussian beam profile")

link_budget_dim1.print(t=t, index = plot_index, type="link budget", elevation = elevation_angles_dim1)
turbulence_dim1.print(index = plot_index, elevation=np.rad2deg(elevation_angles_dim1))


# Plot various parameters over 100s for 1 elevation angle
def plot_dim1_output_parameters():
    # plot performance parameters
    fig_dim1, ax_dim1 = plt.subplots(5,1)
    ax_dim1[0].set_title('Dimension 1 parameters for $\epsilon$ = '+str(np.round(np.rad2deg(elevation_angles_dim1[plot_index]), 2)))
    ax_dim1[0].set_ylabel('Pr [dBm]')
    ax_dim1[0].plot(t, W2dBm(P_r[plot_index]), label='mean: '+str(W2dBm(np.mean(P_r[plot_index])))+' dBm \n'
                                                     'thres (BER=1.0E-9): '+str(np.round(W2dBm(LCT.P_r_thres[0]),2))+' dBm \n'
                                                     'thres (BER=1.0E-6): '+str(np.round(W2dBm(LCT.P_r_thres[1]),2))+' dBm \n'
                                                     'thres (BER=1.0E-3): '+str(np.round(W2dBm(LCT.P_r_thres[2]),2))+' dBm \n',
                                              linewidth='0.5')

    ax_dim1[1].set_ylabel('SNR [dB]')
    ax_dim1[1].plot(t, W2dB(SNR[plot_index]), label='mean: '+str(W2dB(np.mean(SNR[plot_index])))+' dB \n'
                                                    'thres (BER=1.0E-9): '+str(np.round(W2dB(LCT.SNR_thres[0]),2))+' dB \n'
                                                    'thres (BER=1.0E-6): '+str(np.round(W2dB(LCT.SNR_thres[1]),2))+' dB \n'
                                                    'thres (BER=1.0E-3): '+str(np.round(W2dB(LCT.SNR_thres[2]),2))+' dB \n',
                                              linewidth = '0.5')


    ax_dim1[2].plot(t, BER[plot_index], label='uncoded', linewidth='0.1')
    ax_dim1[2].plot(t, BER_coded[plot_index], label='RS (' +str(N)+', '+str(K)+') coded', linewidth='0.5')
    ax_dim1[2].set_ylabel('Error probability \n [# Error bits / total bits]')
    ax_dim1[2].set_yscale('log')
    ax_dim1[2].set_ylim(0.5, 1.0E-10)

    ax_dim1[3].plot(t, errors[plot_index]/1.0E6, label='total='+str(np.round(total_errors[plot_index]/1.0E6,4))+'Mb of '+str(np.sum(total_bits)/1.0E9)+'Gb',
                    linewidth='2')
    ax_dim1[3].plot(t, errors_coded[plot_index] / 1.0E6, label='total='+str(np.round(total_errors_coded[plot_index]/1.0E6,4))+'Mb of '+str(np.sum(total_bits)/1.0E9)+'Gb', linewidth='2')
    ax_dim1[3].set_ylabel('Number of cumulative \n Error bits [Mb]')
    ax_dim1[3].set_yscale('log')
    ax_dim1[3].set_ylim(0.01/1.0E6, total_errors[plot_index]/1.0E6)

    ax_dim1[4].plot(t, W2dBm(noise_tot[plot_index]), label='mean: '+str(W2dBm(np.mean(noise_tot[plot_index])))+' dBm',
                    linewidth='0.5')
    ax_dim1[4].set_ylabel('Noise [dBm]')
    ax_dim1[4].set_xlabel('Time [s]')

    ax_dim1[0].legend()
    ax_dim1[1].legend()
    ax_dim1[2].legend()
    ax_dim1[3].legend()
    ax_dim1[4].legend()

    # plot radials
    fig_losses, ax_losses = plt.subplots(5,1)
    ax_losses[0].set_title('Dimension 1 Fluctuating Vectors for $\epsilon$ = '+str(np.rad2deg(np.round(elevation_angles_dim1[plot_index], 2))))
    ax_losses[0].plot(t, W2dB(h_scint[plot_index]), label='std (1 rms): '+str(np.round(turbulence_dim1.std_scint[plot_index],2))+', '+'mean: '+str(np.round(np.mean(h_scint[plot_index]),2)),
                      linewidth='0.5')
    ax_losses[0].set_ylabel('h scint [dB]')
    ax_losses[1].plot(t, angle_bw[plot_index]*1.0E6, label='std (1 rms): '+str(np.round(turbulence_dim1.std_bw[plot_index]*1.0E6,2))+' urad, '+'mean: '+str(np.round(np.mean(angle_bw[plot_index]*1.0E6),2))+' urad',
                      linewidth='0.5')
    ax_losses[1].set_ylabel('angle bw [urad]')
    ax_losses[2].plot(t, angle_pj_t*1.0E6, label='std (1 rms): '+str(np.round(LCT.std_pj_t*1.0E6,2))+' urad, '+'mean: '+str(np.round(np.mean(angle_pj_t*1.0E6),2))+' urad',
                      linewidth='0.5')
    ax_losses[2].set_ylabel('angle pj t [urad]')
    ax_losses[3].plot(t, angle_pj_r*1.0E6, label='std (1 rms): '+str(np.round(LCT.std_pj_r*1.0E6,2))+' urad, '+'mean: '+str(np.round(np.mean(angle_pj_r*1.0E6),2))+' urad',
                      linewidth='0.5')
    ax_losses[3].set_ylabel('angle pj r [urad]')
    ax_losses[4].plot(t, angle_aoa[plot_index]*1.0E6, label='std (1 rms): '+str(np.round(turbulence_dim1.std_aoa[plot_index]*1.0E6,2))+' urad, '+'mean: '+str(np.round(np.mean(angle_aoa[plot_index]*1.0E6),2))+' urad',
                      linewidth='0.5')
    ax_losses[4].set_ylabel('angle AoA [urad]')
    ax_losses[4].set_xlabel('Time [s]')

    ax_losses[0].legend()
    ax_losses[1].legend()
    ax_losses[2].legend()
    ax_losses[3].legend()
    ax_losses[4].legend()

    fig_losses_tot, ax_losses_tot = plt.subplots(4,1)
    ax_losses_tot[0].set_title('Dimension 1 Power Vectors (RX & TX) for $\epsilon$ = '+str(np.rad2deg(np.round(elevation_angles_dim1[plot_index], 2))))
    ax_losses_tot[0].plot(t, W2dB(h_tot[plot_index]), label='mean: '+str(np.round(np.mean(W2dB(h_tot[plot_index])),2)),
                      linewidth='0.5')
    ax_losses_tot[0].set_ylabel('h tot [dB]')
    ax_losses_tot[1].plot(t, W2dB(h_scint[plot_index]), label='mean: '+str(np.round(np.mean(W2dB(h_scint[plot_index])),2)),
                      linewidth='0.5')
    ax_losses_tot[1].set_ylabel('h scint [dB]')
    ax_losses_tot[2].plot(t, W2dB(h_TX[plot_index]), label='mean: '+str(np.round(np.mean(W2dB(h_TX[plot_index])),2)),
                      linewidth='0.5')
    ax_losses_tot[2].set_ylabel('h TX [dB]')
    ax_losses_tot[3].plot(t, W2dB(h_RX[plot_index]), label='mean: '+str(np.round(np.mean(W2dB(h_RX[plot_index])),2)),
                      linewidth='0.5')
    ax_losses_tot[3].set_ylabel('h RX [dB]')
    ax_losses_tot[3].set_xlabel('Time [s]')

    ax_losses_tot[0].legend()
    ax_losses_tot[1].legend()
    ax_losses_tot[2].legend()
    ax_losses_tot[3].legend()

    fig_coding, ax_coding = plt.subplots(2,1)
    ax_coding[0].set_title('Dimension 1 error performance for $\epsilon$ = '+str(np.rad2deg(np.round(elevation_angles_dim1[plot_index], 2))))
    ax_coding[0].plot(t, errors[plot_index]/1.0E6, label='Uncoded, total='+str(np.round(total_errors[plot_index]/1.0E6,4))+'Mb')
    ax_coding[0].plot(t,errors_coded[plot_index]/1.0E6, label='RS ('+str(N)+', '+str(K)+') coded, total='+str(np.round(total_errors_coded[plot_index]/1.0E6,4))+'Mb')
    ax_coding[0].set_yscale('log')
    ax_coding[0].set_ylim(0.01/1.0E6, total_errors[plot_index]/1.0E6)
    ax_coding[0].set_ylabel('Number of cumulative \n Error bits [Mb]')


    ax_coding[1].plot(t,BER[plot_index], label='Uncoded',
                      linewidth='0.5')
    ax_coding[1].plot(t,BER_coded[plot_index], label='RS ('+str(N)+', '+str(K)+') coded',
                      linewidth='0.5')
    ax_coding[1].set_yscale('log')
    ax_coding[1].set_ylim(0.5, 1.0E-10)
    ax_coding[1].set_ylabel('Error probability \n [# Error bits / total bits]')

    ax_coding[0].legend()
    ax_coding[1].legend()

    plt.show()

# plot_dim1_output_parameters()