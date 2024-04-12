# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_applicable_links import applicable_links

#from JM_Perf_Param_Availability import Availability_performance
#
#
#print('')
#print('------------------END-TO-END-LASER-SATCOM-MODEL-------------------------')
##------------------------------------------------------------------------
##------------------------------TIME-VECTORS------------------------------
##------------------------------------------------------------------------
## Macro-scale time vector is generated with time step 'step_size_link'
## Micro-scale time vector is generated with time step 'step_size_channel_level'
#
#t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
#samples_mission_level = len(t_macro)
#t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
#samples_channel_level = len(t_micro)
#print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
#print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)
#
#print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
#print('')
#print('-----------------------------------MISSION-LEVEL-----------------------------------------')
##------------------------------------------------------------------------
##------------------------------------LCT---------------------------------
##------------------------------------------------------------------------
## Compute the sensitivity and compute the threshold
#LCT = terminal_properties()
#LCT.BER_to_P_r(BER = BER_thres,
#               modulation = modulation,
#               detection = detection,
#               threshold = True)
#PPB_thres = PPB_func(LCT.P_r_thres, data_rate)
#
##------------------------------------------------------------------------
##-----------------------------LINK-GEOMETRY------------------------------
##------------------------------------------------------------------------
## Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
## First both AIRCRAFT and SATELLITES are propagated with 'link_geometry.propagate'
## Then, the relative geometrical state is computed with 'link_geometry.geometrical_outputs'
## Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation
#link_geometry = link_geometry()
#link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
#                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
#link_geometry.geometrical_outputs()
## Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
#time = link_geometry.time
#mission_duration = time[-1] - time[0]
## Update the samples/steps at mission level
#samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])
#
#Links_applicable = applicable_links(time=time)
#applicable_output, sats_applicable = Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
#
#
## Define cross-section of macro-scale simulation based on the elevation angles.
## These cross-sections are used for micro-scale plots.
#elevation_cross_section = [2.0, 20.0, 40.0]
#index_elevation = 1
#
## Retrieve data for all links directly
#time_links = flatten(applicable_output['time'])
#time_links_hrs = [t / 3600.0 for t in time_links]
#ranges = flatten(applicable_output['ranges'])
#elevation = flatten(applicable_output['elevation'])
#zenith = flatten(applicable_output['zenith'])
#slew_rates = flatten(applicable_output['slew rates'])
#heights_SC = flatten(applicable_output['heights SC'])
#heights_AC = flatten(applicable_output['heights AC'])
#speeds_AC = flatten(applicable_output['speeds AC'])
#
#
#time_per_link       = applicable_output['time']
#time_per_link_hrs   = time_links / 3600.0
#ranges_per_link     = applicable_output['ranges'    ]
#elevation_per_link  = applicable_output['elevation' ]
#zenith_per_link     = applicable_output['zenith'    ]
#slew_rates_per_link = applicable_output['slew rates']
#heights_SC_per_link = applicable_output['heights SC']
#heights_AC_per_link = applicable_output['heights AC']
#speeds_AC_per_link  = applicable_output['speeds AC' ]
#
#
#
#
#
#print(elevation)
#print(len(elevation))
#print(len(elevation_per_link))
#
#
#
#
#
#indices, time_cross_section = cross_section(elevation_cross_section, elevation, time_links)
#
#print('')
#print('-------------------------------------LINK-LEVEL------------------------------------------')
#print('')
##------------------------------------------------------------------------
##-------------------------------ATTENUATION------------------------------
#
#att = attenuation(att_coeff=att_coeff, H_scale=scale_height)
#att.h_ext_func(range_link=ranges, zenith_angles=zenith, method=method_att)
#att.h_clouds_func(method=method_clouds)
#h_ext = att.h_ext * att.h_clouds
## Print attenuation parameters
#att.print()
##------------------------------------------------------------------------
##-------------------------------TURBULENCE-------------------------------
## The turbulence class is initiated here. Inside the turbulence class, there are multiple methods that are run directly.
## Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
## With Cn^2 and r0, the variances for scintillation and beam wander are computed
#
#
#
#
#turb = turbulence(ranges=ranges,
#                  h_AC=heights_AC,
#                  h_SC=heights_SC,
#                  zenith_angles=zenith,
#                  angle_div=angle_div)
#turb.windspeed_func(slew=slew_rates,
#                    Vg=speeds_AC,
#                    wind_model_type=wind_model_type)
#turb.Cn_func()
#turb.frequencies()
#r0 = turb.r0_func()
#turb.var_rytov_func()
#turb.var_scint_func()
#turb.WFE(tip_tilt="YES")
#turb.beam_spread()
#turb.var_bw_func()
#turb.var_aoa_func()
#
#print('')
#print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
#print('')
#print('-----------------------------------CHANNEL-LEVEL-----------------------------------------')
#print('')
#for i in indices:
#    turb.print(index=i, elevation=np.rad2deg(elevation), ranges=ranges, Vg=link_geometry.speed_AC.mean(),slew=slew_rates)
## ------------------------------------------------------------------------
## -----------------------------LINK-BUDGET--------------------------------
## The link budget class computes the static link budget (without any micro-scale effects)
## Then it generates a link margin, based on the sensitivity
#link = link_budget(angle_div=angle_div, w0=w0, ranges=ranges, h_WFE=turb.h_WFE, w_ST=turb.w_ST, h_beamspread=turb.h_beamspread, h_ext=h_ext)
#link.sensitivity(LCT.P_r_thres, PPB_thres)
#
## Pr0 (for COMMUNICATION and ACQUISITION phase) is computed with the link budget
#P_r_0, P_r_0_acq = link.P_r_0_func()
#link.print(index=indices[index_elevation], elevation=elevation, static=True)
#
## ------------------------------------------------------------------------
## -------------------------MACRO-SCALE-SOLVER-----------------------------
#noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun, index=indices[index_elevation])
#SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=detection,
#                                  noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
#BER_0 = LCT.BER_func(Q=Q_0, modulation=modulation)
#
#
#
## ------------------------------------------------------------------------
## ----------------------------MICRO-SCALE-MODEL---------------------------
## Here, the channel level is simulated, losses and Pr as output
#P_r, P_r_perfect_pointing, PPB, elevation_angles, losses, angles = \
#    channel_level(t=t_micro,
#                  link_budget=link,
#                  plot_indices=indices,
#                  LCT=LCT, turb=turb,
#                  P_r_0=P_r_0,
#                  ranges=ranges,
#                  angle_div=link.angle_div,
#                  elevation_angles=elevation,
#                  samples=samples_channel_level,
#                  turb_cutoff_frequency=turbulence_freq_lowpass)
#h_tot = losses[0]
#h_scint = losses[1]
#h_RX    = losses[2]
#h_TX    = losses[3]
#h_bw    = losses[4]
#h_aoa   = losses[5]
#h_pj_t  = losses[6]
#h_pj_r  = losses[7]
#h_tot_no_pointing_errors = losses[-1]
#r_TX = angles[0] * ranges[:, None]
#r_RX = angles[1] * ranges[:, None]
#
#
#
#
#
## Here, the bit level is simulated, SNR, BER and throughput as output
#if coding == 'yes':
#    SNR, BER, throughput, BER_coded, throughput_coded, P_r_coded, G_coding = \
#        bit_level(LCT=LCT,
#                  t=t_micro,
#                  plot_indices=indices,
#                  samples=samples_channel_level,
#                  P_r_0=P_r_0,
#                  P_r=P_r,
#                  elevation_angles=elevation,
#                  h_tot=h_tot)
#
#else:
#    SNR, BER, throughput = \
#        bit_level(LCT=LCT,
#                  t=t_micro,
#                  plot_indices=indices,
#                  samples=samples_channel_level,
#                  P_r_0=P_r_0,
#                  P_r=P_r,
#                  elevation_angles=elevation,
#                  h_tot=h_tot)
#
#
#
##throughput = np.array_split(throughput, num_satellites)
#throughput = np.array_split(throughput, num_satellites)
#
#
## ----------------------------FADE-STATISTICS-----------------------------
#
#number_of_fades = np.sum((P_r[:, 1:] < LCT.P_r_thres[1]) & (P_r[:, :-1] > LCT.P_r_thres[1]), axis=1)
#fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / samples_channel_level
#mean_fade_time = fractional_fade_time / number_of_fades * interval_channel_level
#
## Power penalty in order to include a required fade fraction.
## REF: Giggenbach (2008), Fading-loss assessment
#h_penalty   = penalty(P_r=P_r, desired_frac_fade_time=desired_frac_fade_time)                                           
#h_penalty_perfect_pointing   = penalty(P_r=P_r_perfect_pointing, desired_frac_fade_time=desired_frac_fade_time)
#P_r_penalty_perfect_pointing = P_r_perfect_pointing.mean(axis=1) * h_penalty_perfect_pointing
#
## ---------------------------------LINK-MARGIN--------------------------------
#margin     = P_r / LCT.P_r_thres[1]
#
## -------------------------------DISTRIBUTIONS----------------------------
## Local distributions for each macro-scale time step (over micro-scale interval)
#pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r),len(P_r_0),min=-60.0,max=-20.0,steps=1000)
#pdf_BER, cdf_BER, x_BER, std_BER, mean_BER = distribution_function(np.log10(BER),len(P_r_0),min=-30.0,max=0.0,steps=10000)
#if coding == 'yes':
#    pdf_BER_coded, cdf_BER_coded, x_BER_coded, std_BER_coded, mean_BER_coded = \
#        distribution_function(np.log10(BER_coded),len(P_r_0),min=-30.0,max=0.0,steps=10000)
#
## Global distributions over macro-scale interval
#P_r_total = P_r.flatten()
#BER_total = BER.flatten()
#P_r_pdf_total, P_r_cdf_total, x_P_r_total, std_P_r_total, mean_P_r_total = distribution_function(data=W2dBm(P_r_total), length=1, min=-60.0, max=0.0, steps=1000)
#BER_pdf_total, BER_cdf_total, x_BER_total, std_BER_total, mean_BER_total = distribution_function(data=np.log10(BER_total), length=1, min=np.log10(BER_total.min()), max=np.log10(BER_total.max()), steps=1000)
#
#if coding == 'yes':
#    BER_coded_total = BER_coded.flatten()
#    BER_coded_pdf_total, BER_coded_cdf_total, x_BER_coded_total, std_BER_coded_total, mean_BER_coded_total = \
#        distribution_function(data=np.log10(BER_coded_total), length=1, min=-30.0, max=0.0, steps=100)
#
#
## ------------------------------------------------------------------------
## -------------------------------AVERAGING--------------------------------
## ------------------------------------------------------------------------
#
## ---------------------------UPDATE-LINK-BUDGET---------------------------
## All micro-scale losses are averaged and added to the link budget
## Also adds a penalty term to the link budget as a requirement for the desired fade time, defined in input.py
#link.dynamic_contributions(PPB=PPB.mean(axis=1),
#                           T_dyn_tot=h_tot.mean(axis=1),
#                           T_scint=h_scint.mean(axis=1),
#                           T_TX=h_TX.mean(axis=1),
#                           T_RX=h_RX.mean(axis=1),
#                           h_penalty=h_penalty,
#                           P_r=P_r.mean(axis=1),
#                           BER=BER.mean(axis=1))
#
#
#if coding == 'yes':
#    link.coding(G_coding=G_coding.mean(axis=1),
#                BER_coded=BER_coded.mean(axis=1))
#    P_r = P_r_coded
## A fraction (0.9) of the light is subtracted from communication budget and used for tracking budget
#link.tracking()
#link.link_margin()
#
#
## ------------------------------------------------------------------------
## --------------------------PERFORMANCE-METRICS---------------------------
## ------------------------------------------------------------------------
#
#
#availability_vector = (link.LM_comm_BER6 >= 1.0).astype(int)
#availability_vector = np.array_split(availability_vector, num_satellites)
##print(availability_vector)
#
#Availability_performance_instance = Availability_performance(time, link_geometry, availability_vector)
#availability_performance = Availability_performance_instance.calculate_availability_performance()

class ber_performance():
    def __init__(self, time, link_geometry, throughput):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.Links_applicable = applicable_links(time=time)
        self.applicable_output, self.sats_applicable = self.Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
        self.time = time
        self.speed_of_light = speed_of_light
        self.throughput = throughput

    def calculate_BER_performance(self):
        self.BER_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

        for s in range(num_satellites):
            # Convert the throughput list for the current satellite to a NumPy array if not already
            throughput_array = np.array(self.throughput[s])

            # Find indices where throughput equals 250,000,000
            qualifying_indices = np.where(throughput_array == 250000000)[0]

            if len(qualifying_indices) == 0:
                # If no instance of 250,000,000 is found, skip this satellite
                continue

            # Get the index of the first occurrence of 250,000,000
            first_qualifying_index = qualifying_indices[0]

            # Initialize the running sum to 0 initially
            running_sum = 0

            # Loop through time in reverse order starting from the first qualifying index
            for index in range(len(self.time)-1, first_qualifying_index-1, -1):
                # Increment running_sum if the condition is met
                if throughput_array[index] == 250000000:
                    running_sum += 1
                # Update throughput performance with the current running sum
                self.BER_performance[s][index] = running_sum

            # Ensure all values before the first qualifying index are set to 0
            for index in range(first_qualifying_index):
                self.BER_performance[s][index] = 0

        return self.BER_performance
    
    def calculate_normalized_BER_performance(self, data, availability_performance):
        max_time = np.max(availability_performance, axis=1)
        self.normalized_BER_performance = data / max_time[:, np.newaxis]

        return self.normalized_BER_performance
    


    def calculate_BER_performance_including_penalty(self, T_acq, step_size_link):
        # Convert T_acq to the number of time steps
        delay_steps = int(np.ceil(T_acq / step_size_link))

        self.BER_performance_including_penalty = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

        for s in range(num_satellites):
            # Convert the throughput list for the current satellite to a NumPy array if not already
            throughput_array = np.array(self.throughput[s])

            # Find indices where throughput equals 250,000,000
            qualifying_indices = np.where(throughput_array == 250000000)[0]

            if len(qualifying_indices) == 0:
                # If no instance of 250,000,000 is found, skip this satellite
                continue

            # Get the index of the first occurrence of 250,000,000
            first_qualifying_index = qualifying_indices[0]

            # Adjust the starting index for calculating BER performance by adding the delay
            # Ensure this adjusted index does not exceed the length of the time array
            adjusted_start_index = min(first_qualifying_index + delay_steps, len(self.time))

            # Initialize the running sum to 0 initially
            running_sum = 0

            # Update the BER performance list starting from the adjusted index after considering the delay
            for index in range(len(self.time)-1, adjusted_start_index-1, -1):
                # Increment running_sum if the condition is met
                if throughput_array[index] == 250000000:
                    running_sum += 1
                self.BER_performance_including_penalty[s][index] = running_sum

            # For indices before the adjusted start index, the performance is considered not applicable due to the delay
            # So, you might want to set a default value that indicates non-applicability or simply leave it as 0
            for index in range(adjusted_start_index):
                self.BER_performance_including_penalty[s][index] = 0  # or another value indicating non-applicability

        return self.BER_performance_including_penalty


    def calculate_normalized_BER_performance_including_penalty(self, data, sats_applicable):
        # Calculate the maximum time a satellite is applicable to normalize against it
        max_time = np.nansum(sats_applicable, axis=1)  # Using sum to accommodate for delay in applicability
        # Prevent division by zero by replacing zeros with NaN or another suitable value
        max_time[max_time == 0] = np.nan
        self.normalized_BER_performance_including_penalty = data / max_time[:, np.newaxis]

        return self.normalized_BER_performance_including_penalty


#BER_performance_instance = ber_performance(time, link_geometry, throughput)
#BER_performance = BER_performance_instance.calculate_BER_performance()
#BER_performance_including_penalty = BER_performance_instance.calculate_BER_performance_including_penalty(T_acq = 20, step_size_link = 5)
#normalized_BER_performance = BER_performance_instance.calculate_normalized_BER_performance(data = BER_performance, availability_performance= availability_performance)
#normalized_BER_performance_including_penalty = BER_performance_instance.calculate_normalized_BER_performance_including_penalty(data = BER_performance_including_penalty, sats_applicable=sats_applicable)
#print(BER_performance)
#print(normalized_BER_performance)
#print(BER_performance_including_penalty )
#print(normalized_BER_performance_including_penalty )

    def BER_visualization(self, BER_performance, normalized_BER_performance, BER_performance_including_penalty, normalized_BER_performance_including_penalty):
        # To remove all zero values for the BER visualizations, we'll modify the plotting logic
        # to filter out zero values before plotting

        # Plotting with zero values filtered out
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Filter and Plot 1: BER Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(BER_performance[s]) if v > 0]
            non_zero_values = [v for v in BER_performance[s] if v > 0]
            axs[0, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 0].set_title('BER  Performance $Q_{BER} (t_{j})$ ')
        axs[0, 0].set_xlabel('Time Steps')
        axs[0, 0].set_ylabel('BER Performance [# timesteps]')
        axs[0, 0].legend()

        # Filter and Plot 2: Normalized BER Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_BER_performance[s]) if v > 0]
            non_zero_values = [v for v in normalized_BER_performance[s] if v > 0]
            axs[0, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 1].set_title('Normalized BER Performance $\hat{Q}_{BER} (t_{j})$')
        axs[0, 1].set_xlabel('Time Steps')
        axs[0, 1].set_ylabel('Normalized BER Performance [-]')
        axs[0, 1].legend()

        # Filter and Plot 3: BER Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(BER_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in BER_performance_including_penalty[s] if v > 0]
            axs[1, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 0].set_title('BER Performance Including Penalty $Q_{BER} (t_{j})$')
        axs[1, 0].set_xlabel('Time Steps')
        axs[1, 0].set_ylabel('BER Performance w/ Penalty [# timesteps]')
        axs[1, 0].legend()

        # Filter and Plot 4: Normalized BER Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_BER_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in normalized_BER_performance_including_penalty[s] if v > 0]
            axs[1, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 1].set_title('Normalized BER Performance Including Penalty $\hat{Q}_{BER} (t_{j})$')
        axs[1, 1].set_xlabel('Time Steps')
        axs[1, 1].set_ylabel('Normalized BER Performance w/ Penalty [-]')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def BER_performance_vs_availability_visualization(self, throughput, availability_vector):
        plt.figure(figsize=(14, 7))

        # Assuming throughput is a 2D array where each row corresponds to a satellite's throughput data
        # and availability_vector is similarly structured.
        # Convert throughput to Gb/s for visualization
        throughput_gbps = np.array(throughput) / 1e9  # Convert bits per second to Gbps

        # Time points (assuming equal spacing)
        time_points = np.arange(throughput_gbps.shape[1])  # Assuming all satellites have the same number of time points

        # Plot throughput for each satellite
        for s in range(throughput_gbps.shape[0]):  # Loop over satellites
            plt.plot(time_points, throughput_gbps[s, :], label=f'Sat {s+1} Throughput (Gb/s)', marker='o', linestyle='-', alpha=0.7)

        # Since we're plotting availability on the same figure, let's handle that next
        ax2 = plt.gca().twinx()  # Create a twin Axes sharing the xaxis
        # Convert availability to integers (1 or 0), with NaN values handled appropriately
        availability_int = np.where(np.isnan(availability_vector), 0, availability_vector)

        # Plotting availability might be tricky if you want to overlay it directly, 
        # So, an alternative approach would be plotting it separately or using it to shade areas of unavailability.
        # Here, we'll demonstrate a simple approach to visualize availability by highlighting periods of unavailability
        for s in range(availability_int.shape[0]):
            for i in range(availability_int.shape[1]):
                if availability_int[s, i] == 0:
                    plt.axvspan(i-0.5, i+0.5, color='grey', alpha=0.2)  # Shade regions of unavailability

        plt.title('Throughput and Availability Visualization')
        plt.xlabel('Time Point')
        plt.ylabel('Throughput (Gb/s)')
        ax2.set_ylabel('Availability')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()





