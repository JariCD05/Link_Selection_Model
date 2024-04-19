# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

# Import input parameters and helper functions
from input import *
from helper_functions import *
from JM_Visualisations_mission_level import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
#from JM_Atmosphere import JM_attenuation, JM_turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level


#import link applicabality
from JM_applicable_links import applicable_links



#import performance paramaters 
from JM_Perf_Param_Availability import Availability_performance
from JM_Perf_Param_BER import ber_performance
from JM_Perf_Param_Latency import Latency_performance
from JM_Perf_Param_Throughput import Throughput_performance
from JM_Perf_Param_Cost import Cost_performance
from JM_Perf_Param_latency_data_transfer import Latency_data_transfer_performance
from throughput_test import SatelliteThroughputCorrected


#import link selection
from JM_Link_Selection import link_selection


#import dynamic visualisation
from JM_Dynamic_Link_Selection_visualization import Dynamic_link_selection_visualization
class Mission_Level:
    def __init__(self, end_time, start_time, step_size_link, step_size_AC, step_size_SC, aircraft_filename_load, number_sats_per_plane, number_of_planes):
        self.end_time = end_time
        self.start_time = start_time
        self.step_size_link = step_size_link
        self.step_size_AC = step_size_AC
        self.step_size_SC = step_size_SC
        self.aircraft_filename_load = aircraft_filename_load
        self.number_sats_per_plane = number_sats_per_plane
        self.number_of_planes = number_of_planes
        self.setup_time_vectors()
        self.initialize_link_geometry()

    def setup_time_vectors(self):
        self.t_macro = np.arange(0.0, (self.end_time - self.start_time), self.step_size_link)
        self.samples_mission_level = len(self.t_macro)
        print(f'Macro-scale: Interval={self.end_time - self.start_time} seconds, step size={self.step_size_link} seconds, macro-scale steps={self.samples_mission_level}')

    def initialize_link_geometry(self):
        # Initialize and compute link geometry and applicable links
        self.link_geometry = link_geometry()
        self.link_geometry.propagate(time=self.t_macro, step_size_AC=self.step_size_AC, step_size_SC=self.step_size_SC, aircraft_filename=self.aircraft_filename_load, step_size_analysis=False, verification_cons=False)
        self.link_geometry.geometrical_outputs()
        self.time = self.link_geometry.time
        self.mission_duration = self.time[-1] - self.time[0]
        self.samples_mission_level = self.number_sats_per_plane * self.number_of_planes * len(self.link_geometry.geometrical_output['elevation'])
        print(f'Initialized link geometry with mission duration: {self.mission_duration} seconds')
        self.compute_applicable_links()

    def compute_applicable_links(self):
        self.Links_applicable = applicable_links(time=self.time)
        self.applicable_output, self.sats_applicable = self.Links_applicable.applicability(self.link_geometry.geometrical_output, self.time, self.step_size_link)
        self.extract_link_data()
        print('Computed applicable links and extracted link data.')

    def extract_link_data(self):
        # Extract and flatten data for all links directly
        self.time_links = np.concatenate(self.applicable_output['time'])
        self.ranges = np.concatenate(self.applicable_output['ranges'])
        self.elevation = np.concatenate(self.applicable_output['elevation'])
        self.zenith = np.concatenate(self.applicable_output['zenith'])
        self.slew_rates = np.concatenate(self.applicable_output['slew rates'])
        self.heights_SC = np.concatenate(self.applicable_output['heights SC'])
        self.heights_AC = np.concatenate(self.applicable_output['heights AC'])
        self.speeds_AC = np.concatenate(self.applicable_output['speeds AC'])

    def extract_LCT(self):
        LCT = terminal_properties()
        LCT.BER_to_P_r(BER = BER_thres,
                   modulation = modulation,
                   detection = detection,
                   threshold = True)
        

        self.PPB_thres = PPB_func(LCT.P_r_thres, data_rate)


    def calculate_attenuation_and_turbulence(self):
        self.att = attenuation(att_coeff=att_coeff, H_scale=scale_height)
        self.att.h_ext_func(range_link=self.ranges, zenith_angles=self.zenith, method=method_att)
        self.att.h_clouds_func(method=method_clouds)
        self.h_ext = self.att.h_ext * self.att.h_clouds
        self.att.print()

        self.turb = turbulence(ranges=self.ranges, h_AC=self.heights_AC, h_SC=self.heights_SC, zenith_angles=self.zenith, angle_div=angle_div)
        self.turb.windspeed_func(slew=self.slew_rates, Vg=self.speeds_AC, wind_model_type=wind_model_type)
        self.turb.Cn_func()
        self.turb.frequencies()
        self.r0 = self.turb.r0_func()
        self.turb.var_rytov_func()
        self.turb.var_scint_func()
        self.turb.WFE(tip_tilt="YES")
        self.turb.beam_spread()
        self.turb.var_bw_func()
        self.turb.var_aoa_func()
        print('Attenuation and turbulence calculated.')

    def run_link_budget_and_channel_model(self):
        LCT = 
        self.link = link_budget(angle_div=angle_div, w0=w0, ranges=self.ranges, h_WFE=self.turb.h_WFE, w_ST=self.turb.w_ST, h_beamspread=self.turb.h_beamspread, h_ext=self.h_ext)
        self.link.sensitivity(LCT.P_r_thres, self.PPB_thres)
        self.P_r_0, self.P_r_0_acq = self.link.P_r_0_func()
        self.link.print()

        self.channel_model = channel_level(t=self.t_micro, link_budget=self.link, LCT=LCT, turb=self.turb, samples=self.samples_channel_level)
        print('Link budget and channel model run.')



