# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

# Import input parameters and helper functions
#from input_old import *
from JM_INPUT_CONFIG_FILE import *
from helper_functions import *


# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
#from JM_Atmosphere import JM_attenuation, JM_turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level


#import link applicabality
from JM_Sat_Applicable_links import applicable_satellites



#import performance paramaters 
from JM_Perf_Param_Availability import Availability_performance
from JM_Perf_Param_BER import ber_performance
from JM_Perf_Param_Latency import Latency_performance
from JM_Perf_Param_Throughput import Throughput_performance
from JM_Perf_Param_Cost import Cost_performance
from JM_Perf_Param_Latency_data_transfer import Latency_data_transfer_performance





class mission_level:
    def __init__(self, elevation_cross_section, elevation, time_links, ranges, zenith, heights_AC,heights_SC, slew_rates, speeds_AC, index_elevation, t_micro, samples_channel_level, lengths):
        self.elevation_cross_section = elevation_cross_section
        self.elevation = elevation
        self.time_links = time_links
        self.ranges = ranges
        self.zenith = zenith
        self.heights_AC = heights_AC
        self.heights_SC = heights_SC
        self.slew_rates = slew_rates
        self.speeds_AC = speeds_AC
        self.index_elevation = index_elevation
        self.t_micro = t_micro
        self.samples_channel_level = samples_channel_level
        self.lengths = lengths

       

    def calculate_mission_level(self):

        indices, time_cross_section = cross_section(self.elevation_cross_section, self.elevation, self.time_links)

        #print('')
        #print('-------------------------------------LINK-LEVEL------------------------------------------')
        #print('')
        #------------------------------------------------------------------------
        #-------------------------------ATTENUATION------------------------------
        att = attenuation(att_coeff=att_coeff, H_scale=scale_height)
        att.h_ext_func(range_link=self.ranges, zenith_angles=self.zenith, method=method_att)
        att.h_clouds_func(method=method_clouds)
        h_ext = att.h_ext * att.h_clouds
        # Print attenuation parameters
        #att.print()
        #------------------------------------------------------------------------
        #------------------------------------LCT---------------------------------
        #------------------------------------------------------------------------
        # Compute the sensitivity and compute the threshold
        LCT = terminal_properties()
        LCT.BER_to_P_r(BER = BER_thres,
                       modulation = modulation,
                       detection = detection,
                       threshold = True)
        PPB_thres = PPB_func(LCT.P_r_thres, data_rate)
        #-------------------------------TURBULENCE-------------------------------
        # The turbulence class is initiated here. Inside the turbulence class, there are multiple methods that are run directly.
        # Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
        # With Cn^2 and r0, the variances for scintillation and beam wander are computed
        turb = turbulence(ranges=self.ranges,
                          h_AC=self.heights_AC,
                          h_SC=self.heights_SC,
                          zenith_angles=self.zenith,
                          angle_div=angle_div)
        turb.windspeed_func(slew=self.slew_rates,
                            Vg=self.speeds_AC,
                            wind_model_type=wind_model_type)
        turb.Cn_func()
        turb.frequencies()
        r0 = turb.r0_func()
        turb.var_rytov_func()
        turb.var_scint_func()
        turb.WFE(tip_tilt="YES")
        turb.beam_spread()
        turb.var_bw_func()
        turb.var_aoa_func()
        #---------------CHANNEL LEVEL ----------------
        #for i in indices:
            #turb.print(index=i, elevation=np.rad2deg(elevation), ranges=ranges, Vg=link_geometry.speed_AC.mean(),slew=slew_rates)              # PRINT STATEMETN FOR CHANNEL LEVEL OUTPUT
        # ------------------------------------------------------------------------
        # -----------------------------LINK-BUDGET--------------------------------
        # The link budget class computes the static link budget (without any micro-scale effects)
        # Then it generates a link margin, based on the sensitivity
        link = link_budget(angle_div=angle_div, w0=w0, ranges=self.ranges, h_WFE=turb.h_WFE, w_ST=turb.w_ST, h_beamspread=turb.h_beamspread, h_ext=h_ext)
        link.sensitivity(LCT.P_r_thres, PPB_thres)
        # Pr0 (for COMMUNICATION and ACQUISITION phase) is computed with the link budget
        P_r_0, P_r_0_acq = link.P_r_0_func()
        #link.print(index=indices[index_elevation], elevation=elevation, static=True)
        # ------------------------------------------------------------------------
        # -------------------------MACRO-SCALE-SOLVER-----------------------------
        noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun, index=indices[self.index_elevation])
        SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=detection,
                                          noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
        BER_0 = LCT.BER_func(Q=Q_0, modulation=modulation)
        # ------------------------------------------------------------------------
        # ----------------------------MICRO-SCALE-MODEL---------------------------
        # Here, the channel level is simulated, losses and Pr as output
        P_r, P_r_perfect_pointing, PPB, elevation_angles, losses, angles = \
            channel_level(t=self.t_micro,
                          link_budget=link,
                          plot_indices=indices,
                          LCT=LCT, turb=turb,
                          P_r_0=P_r_0,
                          ranges=self.ranges,
                          angle_div=link.angle_div,
                          elevation_angles=self.elevation,
                          samples=self.samples_channel_level,
                          turb_cutoff_frequency=turbulence_freq_lowpass)
        h_tot = losses[0]
        h_scint = losses[1]
        h_RX    = losses[2]
        h_TX    = losses[3]
        h_bw    = losses[4]
        h_aoa   = losses[5]
        h_pj_t  = losses[6]
        h_pj_r  = losses[7]
        h_tot_no_pointing_errors = losses[-1]
        r_TX = angles[0] * self.ranges[:, None]
        r_RX = angles[1] * self.ranges[:, None]
        # Here, the bit level is simulated, SNR, BER and throughput as output
        if coding == 'yes':
            SNR, BER, throughput, BER_coded, throughput_coded, P_r_coded, G_coding = \
                bit_level(LCT=LCT,
                          t=self.t_micro,
                          plot_indices=indices,
                          samples=self.samples_channel_level,
                          P_r_0=P_r_0,
                          P_r=P_r,
                          elevation_angles=self.elevation,
                          h_tot=h_tot)
        else:
            SNR, BER, throughput = \
                bit_level(LCT=LCT,
                          t=self.t_micro,
                          plot_indices=indices,
                          samples=self.samples_channel_level,
                          P_r_0=P_r_0,
                          P_r=P_r,
                          elevation_angles=self.elevation,
                          h_tot=h_tot)
        
        self.throughput = throughput 
        
        # ----------------------------FADE-STATISTICS-----------------------------
        number_of_fades = np.sum((P_r[:, 1:] < LCT.P_r_thres[1]) & (P_r[:, :-1] > LCT.P_r_thres[1]), axis=1)
        fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / self.samples_channel_level
        mean_fade_time = fractional_fade_time / number_of_fades * interval_channel_level
        
        # Power penalty in order to include a required fade fraction.
        # REF: Giggenbach (2008), Fading-loss assessment
        h_penalty   = penalty(P_r=P_r, desired_frac_fade_time=desired_frac_fade_time)                                           
        h_penalty_perfect_pointing   = penalty(P_r=P_r_perfect_pointing, desired_frac_fade_time=desired_frac_fade_time)
        P_r_penalty_perfect_pointing = P_r_perfect_pointing.mean(axis=1) * h_penalty_perfect_pointing
        
        # ---------------------------------LINK-MARGIN--------------------------------
        margin     = P_r / LCT.P_r_thres[1]
        
        # -------------------------------DISTRIBUTIONS----------------------------
        # Local distributions for each macro-scale time step (over micro-scale interval)
        pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r),len(P_r_0),min=-60.0,max=-20.0,steps=1000)
        pdf_BER, cdf_BER, x_BER, std_BER, mean_BER = distribution_function(np.log10(BER),len(P_r_0),min=-30.0,max=0.0,steps=10000)
        if coding == 'yes':
            pdf_BER_coded, cdf_BER_coded, x_BER_coded, std_BER_coded, mean_BER_coded = \
                distribution_function(np.log10(BER_coded),len(P_r_0),min=-30.0,max=0.0,steps=10000)
        
        # Global distributions over macro-scale interval
        P_r_total = P_r.flatten()
        BER_total = BER.flatten()
        P_r_pdf_total, P_r_cdf_total, x_P_r_total, std_P_r_total, mean_P_r_total = distribution_function(data=W2dBm(P_r_total), length=1, min=-60.0, max=0.0, steps=1000)
        BER_pdf_total, BER_cdf_total, x_BER_total, std_BER_total, mean_BER_total = distribution_function(data=np.log10(BER_total), length=1, min=np.log10(BER_total.min()), max=np.log10(BER_total.max()), steps=1000)
        if coding == 'yes':
            BER_coded_total = BER_coded.flatten()
            BER_coded_pdf_total, BER_coded_cdf_total, x_BER_coded_total, std_BER_coded_total, mean_BER_coded_total = \
                distribution_function(data=np.log10(BER_coded_total), length=1, min=-30.0, max=0.0, steps=100)
        # ------------------------------------------------------------------------
        # -------------------------------AVERAGING--------------------------------
        # ------------------------------------------------------------------------
        # ---------------------------UPDATE-LINK-BUDGET---------------------------
        # All micro-scale losses are averaged and added to the link budget
        # Also adds a penalty term to the link budget as a requirement for the desired fade time, defined in input.py
        link.dynamic_contributions(PPB=PPB.mean(axis=1),
                                   T_dyn_tot=h_tot.mean(axis=1),
                                   T_scint=h_scint.mean(axis=1),
                                   T_TX=h_TX.mean(axis=1),
                                   T_RX=h_RX.mean(axis=1),
                                   h_penalty=h_penalty,
                                   P_r=P_r.mean(axis=1),
                                   BER=BER.mean(axis=1))
        if coding == 'yes':
            link.coding(G_coding=G_coding.mean(axis=1),
                        BER_coded=BER_coded.mean(axis=1))
            P_r = P_r_coded
        # A fraction (0.9) of the light is subtracted from communication budget and used for tracking budget
        link.tracking()
        link.link_margin()

        self.BER_outcome = BER.mean(axis=1)
        # CREATE AVAIALABILITY VECTOR
        availability_vector = (link.LM_comm_BER6 >= 1.0).astype(int)
        self.availability_vector = availability_vector

        return self.availability_vector, self.throughput, self.BER_outcome

        
