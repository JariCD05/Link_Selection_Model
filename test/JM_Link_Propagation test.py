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




class LinkPropagation:
    def __init__(self, time):
        self.time = time
        self.links = np.zeros(len(time))  # Assuming this might represent the active link or similar
        self.number_of_links = 0
        self.total_handover_time = 0  # seconds
        self.total_acquisition_time = 0  # seconds
        self.acquisition = np.zeros(len(time))  # Assuming acquisition status over time

        # New attributes for tracking link applicability
        self.elevation_min = np.deg2rad(0)  # Minimum elevation for link applicability
        self.applicable_links = {}  # Store applicable links over time
        self.los_matrix = np.zeros((num_satellites, len(time)))  # Line of Sight matrix
        self.Potential_link_time = np.zeros((num_satellites, len(time)))  # Potential link time
        self.propagation_latency = np.zeros((num_satellites, len(time)))  # Latency per link

    def check_link_applicable(self, geometrical_output):
    
        for sat_id, data in geometrical_output.items():
            elevations = data['elevation']  # Assuming this is how elevation data is structured
            self.applicable_links[sat_id] = elevations > self.elevation_min
    
    def propagate_applicable_links(self):
        """
        Loop through applicable links and perform propagation calculations for each.
        """
        # Iterate over all time steps and satellites to update the Line of Sight (LOS) matrix
        for sat_id, is_applicable_array in self.applicable_links.items():
            for time_index, is_applicable in enumerate(is_applicable_array):
                self.los_matrix[int(sat_id), time_index] = is_applicable

        self.calculate_link_time_performance(num_satellites)
        self.calculate_latency(geometrical_output, num_satellites)

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

    # In the old scenerario the next step would be to initialize the routing model here
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
    def calculate_visibility_matrix(self, index, geometrical_output, elevation_angles):
        start_elev = []
        sats_in_LOS  = []
        for i in range(len(geometrical_output['pos SC'])):
            elev_last = elevation_angles[i][index - 1]
            elev      = elevation_angles[i][index]

                # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD (DIRECT LOS), 
                # Find satellites for an active link, using the condition:
                # (1) The current satellite elevation is higher that the defined minimum (elev > elevation_min)
            if elev > elevation_min :
                start_elev.append(elev)
                sats_in_LOS.append(i)
                self.los_matrix[i, index] = 1

    def calculate_link_time_performance(self, num_satellites):
        # Calculate the potential link time of a link
        visibility_index = 0 
        # Calculate future visibility for each satellite
        for sat_index in range(num_satellites):                                                         # check for all satellites within the available satellites
            for visibility_index in range(len(time)): 
                link_time = 0                                                                           #starting potential time is zere
                if self.los_matrix[sat_index, visibility_index] == 0:                                   # if sattelite is not within line of sight, potential time is zero
                    self.Potential_link_time[sat_index,visibility_index] = 0
                else:
                    # Find future instances of visibility
                    future_index = visibility_index                                                     # if not, look at all future time instances and if these are still a 1, add this time to the link time
                    while future_index <len(time):
                        if self.los_matrix[sat_index, future_index] == 1:
                            link_time +=1
                        else:
                            break
                        future_index+=1
                    self.Potential_link_time[sat_index, visibility_index] = link_time
    
    def calculate_latency(self, geometrical_output, num_satellites):
        range_index = 0
        ranges = geometrical_output['ranges']
        self.propagation_latency = np.zeros((num_satellites, len(time)))
        for sat_index in range(len(ranges)):
            for range_index in range(len(time)):
                if self.los_matrix[sat_index, range_index] ==1:
                    latency_propagation = ranges[sat_index][range_index] / speed_of_light
                    self.propagation_latency[sat_index,range_index] = latency_propagation
                else:
                    latency_propagation = 0 
                    self.propagation_latency[sat_index,range_index] = latency_propagation

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





   

