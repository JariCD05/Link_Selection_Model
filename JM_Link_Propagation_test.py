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
#from JM_applicable_links import applicable_links



class link_propagation_test():
    #Links_applicable = applicable_links(time=time)
    #applicable_output, applicable_total_output, sats_visibility = Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
    #index = 0
    #for i in sats_visibility(index):   
        #this needs to be done without the link selection performed
        def initialize_macro_timescale(self):
            t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
            samples_mission_level = len(t_macro)
            print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)

        #this needs to be done without the link selection performed
        def initialize_micro_timescale(self):
            t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
            samples_channel_level = len(t_micro)
            print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

        #this needs to be done without the link selection performed
        def compute_sensity_and_threshold(self):
            LCT = terminal_properties()
            LCT.BER_to_P_r(BER = BER_thres,
                   modulation = modulation,
                   detection = detection,
                   threshold = True)
            PPB_thres = PPB_func(LCT.P_r_thres, data_rate)

        #this needs to be done without the link selection performed, because here only the link geometry is defined
        #the naming used to be link_geometry = link_geometry() without the uppercase but this didn't seem to work
        def initiate_link_geometry(self,t_macro, time):
            Link_geometry = link_geometry()
            Link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                            aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
            Link_geometry.geometrical_outputs()
            # Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
            time = Link_geometry.time
            mission_duration = time[-1] - time[0]
            # Update the samples/steps at mission level
            samples_mission_level = number_sats_per_plane * number_of_planes * len(Link_geometry.geometrical_output['elevation'])


            Link_applicability = applicable_links()
            availability_output, availability_total_output = Link_applicability .applicability(link_geometry.geometrical_output, time, step_size_link)

            # Options are to analyse 1 link or analyse 'all' links
            #   (1) link_number == 'all'   : Creates 1 vector for each geometric variable for each selected link & creates a flat vector
            #   (2) link_number == 1 number: Creates 1 vector for each geometric variable

            if link_number == 'all':
                time_links  = flatten(availability_output['time'     ])
                time_links_hrs = time_links / 3600.0
                ranges     = flatten(availability_output['ranges'    ])
                elevation  = flatten(availability_output['elevation' ])
                zenith     = flatten(availability_output['zenith'    ])
                slew_rates = flatten(availability_output['slew rates'])
                heights_SC = flatten(availability_output['heights SC'])
                heights_AC = flatten(availability_output['heights AC'])
                speeds_AC  = flatten(availability_output['speeds AC'])

                time_per_link       = availability_output['time'      ]
                time_per_link_hrs   = time_links / 3600.0
                ranges_per_link     = availability_output['ranges'    ]
                elevation_per_link  = availability_output['elevation' ]
                zenith_per_link     = availability_output['zenith'    ]
                slew_rates_per_link = availability_output['slew rates']
                heights_SC_per_link = availability_output['heights SC']
                heights_AC_per_link = availability_output['heights AC']
                speeds_AC_per_link  = availability_output['speeds AC' ]

            else:
                time_links     = availability_output['time'      ][link_number]
                time_links_hrs = time_links / 3600.0
                ranges         = availability_output['ranges'    ][link_number]
                elevation      = availability_output['elevation' ][link_number]
                zenith         = availability_output['zenith'    ][link_number]
                slew_rates     = availability_output['slew rates'][link_number]
                heights_SC     = availability_output['heights SC'][link_number]
                heights_AC     = availability_output['heights AC'][link_number]
                speeds_AC      = availability_output['speeds AC' ][link_number]


            # Define cross-section of macro-scale simulation based on the elevation angles.
            # These cross-sections are used for micro-scale plots.
            elevation_cross_section = [2.0, 20.0, 40.0]
            index_elevation = 1
            indices, time_cross_section = cross_section(elevation_cross_section, elevation, time_links)



        # In the old scenerario the previous def would have importated the link selection model class and 
        # I will check if all following def are dependent on if a link is selected or that is more general environmental related

        # The ranges used for h_ext_func are the distance between AC and SC, which in the end is an output from link_geometry in which all geomterical outputs are given for a potential link 
        # No dependencies on if a link is selected or nog
        def attenuation(self, ranges, zenith):
            att = attenuation(att_coeff=att_coeff, H_scale=scale_height)
            att.h_ext_func(range_link=ranges, zenith_angles=zenith, method=method_att)
            att.h_clouds_func(method=method_clouds)
            h_ext = att.h_ext * att.h_clouds
            # Print attenuation parameters
            #att.print()


        # Tijmen: All these inputs for the turbulence model are in the end output from some link_geometry(), can't those be defined in one def and that we just have to call 1 input here?
        # No depencies on link selected, as all are outputs from link_geometry()
        def turbulence(self, ranges, heights_AC, heights_SC, zenith, slew_rates, speeds_AC, indices, elevation):
            turb = turbulence(ranges=ranges,
                      h_AC=heights_AC,
                      h_SC=heights_SC,
                      zenith_angles=zenith,
                      angle_div=angle_div)
            turb.windspeed_func(slew=slew_rates,
                        Vg=speeds_AC,
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

            for i in indices:
                turb.print(index=i, elevation=np.rad2deg(elevation), ranges=ranges, Vg=link_geometry.speed_AC.mean(),slew=slew_rates)


    # Tijmen, what is the thing here with also having to providng index_elevation as an input


        def link_budget(self, ranges, h_ext, PPB_thres, elevation, indices, index_elevation):
            LCT = terminal_properties()
            turb = turbulence()
            link = link_budget(angle_div=angle_div, w0=w0, ranges=ranges, h_WFE=turb.h_WFE, w_ST=turb.w_ST, h_beamspread=turb.h_beamspread, h_ext=h_ext)
            link.sensitivity(LCT.P_r_thres, PPB_thres)
            # Pr0 (for COMMUNICATION and ACQUISITION phase) is computed with the link budget
            P_r_0, P_r_0_acq = link.P_r_0_func()
            link.print(index=indices[index_elevation], elevation=elevation, static=True)        

        def macro_scale_noise(self, P_r_0, indices, index_elevation ):
            LCT = terminal_properties()
            noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=P_r_0, I_sun=I_sun, index=indices[index_elevation])
            SNR_0, Q_0 = LCT.SNR_func(P_r=P_r_0, detection=detection,
                                      noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
            BER_0 = LCT.BER_func(Q=Q_0, modulation=modulation)  

        def micro_scale_channel_level(self, t_micro, indices, LCT, turb, P_r_0, ranges, elevation, samples_channel_level):
            turb=turbulence()
            link = link_budget()
            P_r, P_r_perfect_pointing, PPB, elevation_angles, losses, angles = \
                channel_level(t=t_micro,
                                link_budget=link,
                                plot_indices=indices,
                                LCT=LCT, turb=turb,
                                P_r_0=P_r_0,
                                ranges=ranges,
                                angle_div=link.angle_div,
                                elevation_angles=elevation,
                                samples=samples_channel_level,
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
            r_TX = angles[0] * ranges[:, None]
            r_RX = angles[1] * ranges[:, None]

        def simulate_bit_level(self, t_micro, indices, samples_channel_level, P_r_0, P_r, elevation, h_tot):
            LCT = terminal_properties()
            if coding == 'yes':
                SNR, BER, throughput, BER_coded, throughput_coded, P_r_coded, G_coding = \
                    bit_level(LCT=LCT,
                              t=t_micro,
                              plot_indices=indices,
                              samples=samples_channel_level,
                              P_r_0=P_r_0,
                              P_r=P_r,
                              elevation_angles=elevation,
                              h_tot=h_tot)

            else:
                SNR, BER, throughput = \
                    bit_level(LCT=LCT,
                              t=t_micro,
                              plot_indices=indices,
                              samples=samples_channel_level,
                              P_r_0=P_r_0,
                              P_r=P_r,
                              elevation_angles=elevation,
                              h_tot=h_tot)

        def fading_statistics(self, P_r, samples_channel_level):
            LCT = terminal_properties()
            number_of_fades = np.sum((P_r[:, 1:] < LCT.P_r_thres[1]) & (P_r[:, :-1] > LCT.P_r_thres[1]), axis=1)
            fractional_fade_time = np.count_nonzero((P_r < LCT.P_r_thres[1]), axis=1) / samples_channel_level
            mean_fade_time = fractional_fade_time / number_of_fades * interval_channel_level

        def power_penalty(self, P_r, P_r_perfect_pointing):
            h_penalty   = penalty(P_r=P_r, desired_frac_fade_time=desired_frac_fade_time)                                           
            h_penalty_perfect_pointing   = penalty(P_r=P_r_perfect_pointing, desired_frac_fade_time=desired_frac_fade_time)
            P_r_penalty_perfect_pointing = P_r_perfect_pointing.mean(axis=1) * h_penalty_perfect_pointing       

        def link_margin(self, P_r):
            LCT = terminal_properties()
            margin     = P_r / LCT.P_r_thres[1]

        def distributions_macro_scale(self, P_r, P_r_0, BER, BER_coded):

            pdf_P_r, cdf_P_r, x_P_r, std_P_r, mean_P_r = distribution_function(W2dBm(P_r),len(P_r_0),min=-60.0,max=-20.0,steps=1000)
            pdf_BER, cdf_BER, x_BER, std_BER, mean_BER = distribution_function(np.log10(BER),len(P_r_0),min=-30.0,max=0.0,steps=10000)
            if coding == 'yes':
                pdf_BER_coded, cdf_BER_coded, x_BER_coded, std_BER_coded, mean_BER_coded = \
                    distribution_function(np.log10(BER_coded),len(P_r_0),min=-30.0,max=0.0,steps=10000)

        def distributions_global_macro_scale(self, P_r, BER, BER_coded):

            P_r_total = P_r.flatten()
            BER_total = BER.flatten()
            P_r_pdf_total, P_r_cdf_total, x_P_r_total, std_P_r_total, mean_P_r_total = distribution_function(data=W2dBm(P_r_total), length=1, min=-60.0, max=0.0, steps=1000)
            BER_pdf_total, BER_cdf_total, x_BER_total, std_BER_total, mean_BER_total = distribution_function(data=np.log10(BER_total), length=1, min=np.log10(BER_total.min()), max=np.log10(BER_total.max()), steps=1000)

            if coding == 'yes':
                BER_coded_total = BER_coded.flatten()
                BER_coded_pdf_total, BER_coded_cdf_total, x_BER_coded_total, std_BER_coded_total, mean_BER_coded_total = \
                    distribution_function(data=np.log10(BER_coded_total), length=1, min=-30.0, max=0.0, steps=100)

        def averaging_link_budget(self, PPB, h_tot, h_scint, h_TX, h_RX, h_penalty, BER, G_coding, BER_coded, P_r_coded):
            link = link_budget()
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


        # ------------------------------------------------------------------------
        # -----------------------------PERFORMANCE PARAMETER FUNCTIONS----------------------------------
        # ------------------------------------------------------------------------

        def calculate_latency(self, geometrical_output, num_satellites):
            link_geometry.geometrical_outputs()
            range_index = 0
            ranges = geometrical_output['ranges'] 
            self.propagation_latency = np.zeros((num_satellites, len(time)))
            for sat_index in range(len(ranges)):
                for range_index in range(len(time)):
                    latency_propagation = ranges[sat_index][range_index] / speed_of_light
                    self.propagation_latency[sat_index,range_index] = latency_propagation
                else:
                    latency_propagation = 0 
                    self.propagation_latency[sat_index,range_index] = latency_propagation
            #print(ranges)
        
        def visualize_propagation_latency(self, time):
            # Assuming 'time' is a 1D numpy array with the same length as the columns in self.propagation_latency
            plt.figure(figsize=(14, 8))

            for sat_index in range(self.propagation_latency.shape[0]):
                plt.plot(time, self.propagation_latency[sat_index, :], label=f'Sat {sat_index + 1}')

            plt.xlabel('Time (s)')
            plt.ylabel('Propagation Latency (s)')
            plt.title('Propagation Latency Over Time for Each Satellite')
            plt.legend()
            plt.grid(True)
            plt.show()


    # Jari, have a look at the mask.astype steup and where it comes from
        def availability(self, time_links):
            # Availability
            # No availability is assumed below link margin threshold
            link = link_budget()
            availability_vector = mask.astype(int)
            find_link_margin = np.where(link.LM_comm_BER6 < 1.0)[0]
            time_link_fail = time_links[find_link_margin]        
            find_time = np.where(np.in1d(time, time_link_fail))[0]
            availability_vector[find_time] = 0.0

        def throughput(self, throughput, find_link_margin, indices, index_elevation, SNR_penalty):
            LCT = terminal_properties()
            link = link_budget()
            throughput[find_link_margin] = 0.0
            # Potential throughput with the Shannon-Hartley theorem
            noise_sh, noise_th, noise_bg, noise_beat = LCT.noise(P_r=link.P_r, I_sun=I_sun, index=indices[index_elevation])
            SNR_penalty, Q_penalty = LCT.SNR_func(link.P_r, detection=detection,
                                      noise_sh=noise_sh, noise_th=noise_th, noise_bg=noise_bg, noise_beat=noise_beat)
            potential_throughput = BW * np.log2(1 + SNR_penalty)


        # Thijmen, this was the start you made, I created the run_simulation
        # Let's continue with one of the two and remove the other one
        def results(self, geometrical_output, time, step_size=1.0):
            index = 0
            elevation_angles = geometrical_output['elevation']

            # Initialize the LOS matrix
            num_satellites = len(geometrical_output['pos SC'])
            self.los_matrix = np.zeros((num_satellites, len(time)))
            self.Potential_link_time = np.zeros((num_satellites, len(time)))

            for index in range(time):
                self.calculate_visibility_matrix(index, geometrical_output, elevation_angles)
                self.calculate_link_time_performance(num_satellites=num_satellites)
                self.calculate_latency(geometrical_output, num_satellites)

        def run_simulation(self, index, ranges, zenith, elevation, index_elevation, start_time, end_time, SNR_penalty, step_size_link, find_link_margin, h_ext, PPB, elevation_angles, P_r, P_r_0, P_r_perfect_pointing,t_micro, heights_AC, heights_SC, slew_rates, speeds_AC, indices, PPB_thres, geometrical_output, num_satellites, time_links, throughput, samples_channel_level, h_tot, h_scint, h_TX, h_RX, h_penalty, BER, G_coding, BER_coded, P_r_coded):
            # Initialize macro and micro timescales
            self.initialize_macro_timescale()
            self.initialize_micro_timescale()

            # Compute sensitivity and threshold based on terminal properties
            self.compute_sensity_and_threshold()

            # Initiate and compute geometrical aspects of link geometry over time
            for time_step in np.linspace(start_time, end_time, num=int((end_time - start_time) / step_size_link)):
                self.initiate_link_geometry(time_step, np.linspace(start_time, end_time, num=self.samples_mission_level))

            # Initiate atmospheric attenuation and turbulence parameters
            # Assuming 'ranges' and 'zenith' are pre-computed based on the geometry
            self.attenuation(ranges, zenith)
            self.turbulence(ranges, heights_AC, heights_SC, zenith, slew_rates, speeds_AC, indices, elevation)

            # Compute link budget, considering the atmospheric effects and terminal properties
            self.link_budget(ranges, h_ext, PPB_thres, elevation, indices, index_elevation)

            # Initiate the macro scale noise model
            self.macro_scale_noise(P_r_0, indices, index_elevation)

            # Initiate micro scale channel model
            self.micro_scale_channel_level(t_micro, indices, P_r_0, ranges, elevation, samples_channel_level)

            # Simulate bit level performance
            self.simulate_bit_level(t_micro, indices, samples_channel_level, P_r_0, P_r, elevation, h_tot)

            # Compute fading statistics
            self.fading_statistics(P_r, samples_channel_level)

            # Compute power penalty due to imperfect pointing
            self.power_penalty(P_r, P_r_perfect_pointing)

            # Compute the link margin
            self.link_margin(P_r)

            # Compute distributions at macro scale and global macro scale
            self.distributions_macro_scale(P_r, P_r_0, BER, BER_coded)
            self.distributions_global_macro_scale(P_r, BER, BER_coded)

            # Average out the link budget contributions and dynamics
            self.averaging_link_budget(PPB, h_tot, h_scint, h_TX, h_RX, h_penalty, BER, G_coding, BER_coded, P_r_coded)

            # Calculate the visibility matrix and link time performance
            self.calculate_visibility_matrix(index, geometrical_output, elevation_angles)
            self.calculate_link_time_performance(num_satellites)

            # Calculate latency and availability metrics
            self.calculate_latency(geometrical_output, num_satellites)
            self.availability(time_links)

            # Calculate the final throughput considering SNR penalty
            self.throughput(throughput, find_link_margin, indices, index_elevation, SNR_penalty)
    #index=+1

#initialization code to tart link propagation test


t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

link_propagation_test = link_propagation_test()
link_geometry = link_geometry()
link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
geometrical_output = link_geometry.geometrical_outputs()
# Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
time = link_geometry.time
mission_duration = time[-1] - time[0]
# Update the samples/steps at mission level
samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])



#applicable_output, applicable_total_output, sats_visibility =applicable_links.applicability(time = time, geometrical_output = geometrical_output)
Latency_propagation = link_propagation_test.calculate_latency(geometrical_output=geometrical_output, num_satellites= num_satellites)
Visualize_latency_propagation = link_propagation_test.visualize_propagation_latency(time)
