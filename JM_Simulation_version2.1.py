# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import math
import csv
from copy import deepcopy

# Import input parameters and helper functions
#from input_old import *
from JM_INPUT_CONFIG_FILE import *
from helper_functions import *


# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence

# Import supporting micro-scale calculation functions
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level

#Import aircraft data
from JM_AC_propagation import geometrical_data_AC


# Import link applicabality
from JM_Sat_Applicable_links import applicable_satellites

# Import link applicabality
from JM_Sat_Visible_links import visible_satellites
#from JM_Sat_Visible_links_copy import visible_satellites_copy

# Import propagated applicability
from JM_Sat_Applicable_Total_Output import applicable_sat_propagation

# Import performance paramaters 
from JM_Perf_Param_Availability import Availability_performance
from JM_Perf_Param_BER import ber_performance
from JM_Perf_Param_Latency import Latency_performance
from JM_Perf_Param_Throughput import Throughput_performance
from JM_Perf_Param_Cost import Cost_performance
from JM_Perf_Param_Latency_data_transfer import Latency_data_transfer_performance

# Import mission level
from JM_mission_level import mission_level

# Import link selection
from JM_Link_selection_No_link import link_selection_no_link


# Import dynamic visualisation
from JM_Dynamic_Link_Selection_visualization import Dynamic_link_selection_visualization

# Import mission analysis
from JM_mission_performance import SatelliteLinkMetrics

# Import visualization
from JM_visualizations_mission_level import SatelliteDataVisualizer


print('')
print('------------------END-TO-END-LASER-SATCOM-MODEL-------------------------')
#------------------------------------------------------------------------
#------------------------------TIME-VECTORS------------------------------
#------------------------------------------------------------------------
# Macro-scale time vector is generated with time step 'step_size_link'
# Micro-scale time vector is generated with time step 'step_size_channel_level'

t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------
# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
# First both AIRCRAFT and SATELLITES are propagated with 'link_geometry.propagate'
# Then, the relative geometrical state is computed with 'link_geometry.geometrical_outputs'
# Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation
link_geometry = link_geometry()
link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
link_geometry.geometrical_outputs()

# Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
time = link_geometry.time
propagated_time = link_geometry.time
mission_duration = time[-1] - time[0]

# Update the samples/steps at mission level
samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])

# Define cross-section of macro-scale simulation based on the elevation angles.
# These cross-sections are used for micro-scale plots.
elevation_cross_section = [2.0, 20.0, 40.0]
index_elevation = 1

# ----- PROPAGATED APPLICABILITY ----------
Propagated_Applicability_instance = applicable_sat_propagation(time=time)
applicable_propagated_output, propagated_sats_applicable = Propagated_Applicability_instance.sat_propagation(link_geometry.geometrical_output, time=time)

length_applicability = max_consecutive_ones(propagated_sats_applicable)
max_length_applicability = max(length_applicability)


# ----- PROPAGATED GEOMETRICAL OUTPUT -------
propagated_sats_applicable = flatten(propagated_sats_applicable)
initial_time_links = flatten(applicable_propagated_output['time'])
time_links_hrs = [t / 3600.0 for t in initial_time_links]
initial_ranges = flatten(applicable_propagated_output['ranges'])
initial_elevation = flatten(applicable_propagated_output['elevation'])
initial_zenith = flatten(applicable_propagated_output['zenith'])
initial_slew_rates = flatten(applicable_propagated_output['slew rates'])
initial_heights_SC = flatten(applicable_propagated_output['heights SC'])
initial_heights_AC = flatten(applicable_propagated_output['heights AC'])
initial_speeds_AC = flatten(applicable_propagated_output['speeds AC'])


time_per_link       = applicable_propagated_output['time']
time_per_link_hrs   = initial_time_links / 3600.0
ranges_per_link     = applicable_propagated_output['ranges'    ]
elevation_per_link  = applicable_propagated_output['elevation' ]
zenith_per_link     = applicable_propagated_output['zenith'    ]
slew_rates_per_link = applicable_propagated_output['slew rates']
heights_SC_per_link = applicable_propagated_output['heights SC']
heights_AC_per_link = applicable_propagated_output['heights AC']
speeds_AC_per_link  = applicable_propagated_output['speeds AC' ]

# ------ CALCULATE SMALLEST LATENCY; USED FOR NORMALIZATION -------
smallest_range = min(x for x in initial_ranges if x != 0 and not math.isnan(x))
smallest_latency = smallest_range/speed_of_light


# initiate index and active satellite
index = 0 
score_of_active_satellite = 0
active_satellite = 'No Link'

# CREATE LIST TO APPENDS VALUES
list_of_activated_satellite_index = []                                              # The index of the active satellite
list_of_activated_sattellite_numbers = []                                           # The number of the active satellite (i.e. omitting sat 0)
list_of_scores_active_satellites = []                                               # The combined weight of the performance values and the accompanied weight
list_of_sattellite_availability_while_being_active = []                             # List to check if there is a positive link margin given that the satellite is made active
list_of_total_scores_all_satellite = np.full((max_satellites, len(time)), np.nan)   # List with lenght time and collumns number of max satellites where al performance values are stored 

# APPEND PHYSICAL LAYER PERFORMANCE
list_of_throughput_active_satellite = np.full((max_satellites, len(time)), np.nan) 
list_of_BER_active_satellite = np.full((max_satellites, len(time)), np.nan) 

stored_throughput_active_satellite = []
stored_BER_active_satellite = []

previous_satellite_position_indices = None
satellite_memory_indeces = None

# APPEND APPLICABILITY
list_of_applicable_satellites = []

# APPEND PERFORMANCE SCORES

# Availability
list_of_availability_performance = []
list_of_normalized_availability_performance = []
list_of_penalized_availability_performance = []
list_of_normalized_penalized_availability_performance = []

# BER
list_of_BER_performance = []
list_of_normalized_BER_performance = []
list_of_penalized_BER_performance = []
list_of_normalized_penalized_BER_performance = []

# Cost
list_of_cost_performance = []
list_of_normalized_cost_performance = []
list_of_penalized_cost_performance = []
list_of_normalized_penalized_cost_performance = []

# Latency
list_of_latency_performance = []
list_of_normalized_latency_performance = []
list_of_penalized_latency_performance = []
list_of_normalized_penalized_latency_performance = []

# Latency Data Transfer
list_of_latency_data_transfer_performance = []
list_of_normalized_latency_data_transfer_performance = []
list_of_penalized_latency_data_transfer_performance = []
list_of_normalized_penalized_latency_data_transfer_performance = []

# Throughput
list_of_throughput_performance = []
list_of_normalized_throughput_performance = []
list_of_penalized_throughput_performance = []
list_of_normalized_penalized_throughput_performance = []



#-----START OF CALCULATION TO SELECT SATELLITE FOR EACH TIME INDEX-------------------
for index in range(len(time)):

    #append performance values of previous PREVIOUS time instance
    if index >=2:

        # Cost
        list_of_cost_performance.append(cost_performance)
        list_of_normalized_cost_performance.append(normalized_cost_performance)


        # Latency
        list_of_latency_performance.append(latency_performance)
        list_of_normalized_latency_performance.append(normalized_latency_performance)


        # Latency Data Transfer
        list_of_latency_data_transfer_performance.append(latency_data_transfer_performance)
        list_of_normalized_latency_data_transfer_performance.append(normalized_latency_data_transfer_performance)
        list_of_penalized_latency_data_transfer_performance.append(penalized_latency_data_transfer_performance)
        list_of_normalized_penalized_latency_data_transfer_performance.append(normalized_penalized_latency_data_transfer_performance)

        #Availability
        list_of_availability_performance.append(availability_performance)
        list_of_normalized_availability_performance.append(normalized_availability_performance)
        list_of_penalized_availability_performance.append(penalized_availability_performance)
        list_of_normalized_penalized_availability_performance.append(normalized_penalized_availability_performance)

        # BER
        list_of_BER_performance.append(BER_performance)
        list_of_normalized_BER_performance.append(normalized_BER_performance)
        list_of_penalized_BER_performance.append(penalized_BER_performance)
        list_of_normalized_penalized_BER_performance.append(normalized_penalized_BER_performance)


        # Throughput
        list_of_throughput_performance.append(throughput_performance)
        list_of_normalized_throughput_performance.append(normalized_throughput_performance)
        list_of_penalized_throughput_performance.append(penalized_throughput_performance)
        list_of_normalized_penalized_throughput_performance.append(normalized_penalized_throughput_performance)
        


    #------------------------------------------------------------
    
    list_of_activated_satellite_index.append(active_satellite)
    list_of_scores_active_satellites.append(score_of_active_satellite)

    print("------------------------------------------------")

    #--------AIRCRAFT PROPAGATION
    aircraft_data_instance = geometrical_data_AC(index, time=time[index])
    AC_geometrical_output = aircraft_data_instance.AC_propagation(link_geometry.geometrical_output)

    #print(AC_geometrical_output['heights AC'])

    #---------VISIBILITY---------
    Visiblity_instance = visible_satellites(index, time=time[index])
    visibility_output, sats_visible = Visiblity_instance.visibility(link_geometry.geometrical_output,time[index])

    #---------APPLICABILITY ----------
    Applicability_instance= applicable_satellites(index, time=time[index])
    sats_applicable = Applicability_instance.applicability(visibility_output)

    # build in process such that if sats_applicable has 1 satellite it is made active
    print(f"Timestamp={index}, sats applicable {sats_applicable}")
    reference_positional_data = sats_applicable

    #--------CLEAN SATS APPLICABLE------
    # REMOVE NAN VALUES AND THEREFORE ONLY PROPAGATING THE APPLICABLE SATELLITES
    # ONLY SELECT VALUES FROM APPLICABLE SATELLITES 
    
    satellite_position_indices = [position for position, value in enumerate(sats_applicable) if value == 1]
    satellite_position_copy = deepcopy(satellite_position_indices)
    start_time_indices = [idx*len(time)+index for idx in satellite_position_indices]
    
    # Example using 'ranges' and 'propagated_sats_applicable'
    collected_time_links = collect_data(start_time_indices, initial_time_links, propagated_sats_applicable)
    collected_ranges, lengths = collect_data_and_lengths(start_time_indices, initial_ranges, propagated_sats_applicable)
    collected_elevation = collect_data(start_time_indices, initial_elevation, propagated_sats_applicable)
    collected_zenith = collect_data(start_time_indices, initial_zenith, propagated_sats_applicable)
    collected_slew_rates = collect_data(start_time_indices, initial_slew_rates, propagated_sats_applicable)
    collected_heights_SC = collect_data(start_time_indices, initial_heights_SC, propagated_sats_applicable)
    collected_heights_AC = collect_data(start_time_indices, initial_heights_AC, propagated_sats_applicable)
    collected_speeds_AC = collect_data(start_time_indices, initial_speeds_AC, propagated_sats_applicable)
    
    # FLATTEN OUT COLLECTED LIST PER TIME INDEX
    time_links = flatten(collected_time_links)
    ranges = flatten(collected_ranges)
    elevation = flatten(collected_elevation)
    zenith = flatten(collected_zenith)
    slew_rates = flatten(collected_slew_rates)
    heights_SC = flatten(collected_heights_SC)
    heights_AC = flatten(collected_heights_AC)
    speeds_AC = flatten(collected_speeds_AC)
    


    #----- split up ranges according to size of the satellite applicability array -----
    ranges_split = split_data_by_lengths(ranges, lengths)
    num_satellites = len(satellite_position_indices)






    #------ CALCULATION OF INSTANTENEOUS PERFORMANCE PARAMETERS, WHICH ARE INDEPENDENT ON MACRO-MICRO MODEL
    Cost_performance_instance = Cost_performance(time, lengths, num_satellites)
    Latency_performance_instance = Latency_performance(time, ranges_split, lengths, num_satellites, smallest_latency)
    Latency_data_transfer_performance_instance = Latency_data_transfer_performance(time, ranges_split, lengths, num_satellites, smallest_latency, acquisition_time_steps)

    # Cost performance parameter
    cost_performance = Cost_performance_instance.calculate_cost_performance()
    normalized_cost_performance = Cost_performance_instance.calculate_normalized_cost_performance()
    penalized_cost_performance = Cost_performance_instance.calculate_penalized_cost_performance()
    normalized_penalized_cost_performance = Cost_performance_instance.calculate_normalized_penalized_cost_performance()

    # Latency performance parameter
    latency_performance = Latency_performance_instance.calculate_latency_performance()                                         
    normalized_latency_performance = Latency_performance_instance.calculate_normalized_latency_performance()

    # Latency data transfer performance parameter
    latency_data_transfer_performance = Latency_data_transfer_performance_instance.calculate_latency_data_transfer_performance()                                          
    normalized_latency_data_transfer_performance = Latency_data_transfer_performance_instance.calculate_normalized_latency_data_transfer_performance()
    penalized_latency_data_transfer_performance = Latency_data_transfer_performance_instance.calculate_penalized_latency_data_transfer_performance()  
    normalized_penalized_latency_data_transfer_performance = Latency_data_transfer_performance_instance.calculate_normalized_penalized_latency_data_transfer_performance()  

    ## Cost
    #list_of_cost_performance.append(cost_performance)
    #list_of_normalized_cost_performance.append(normalized_cost_performance)
#
    #
    ## Latency
    #list_of_latency_performance.append(latency_performance)
    #list_of_normalized_latency_performance.append(normalized_latency_performance)
#
    #
    ## Latency Data Transfer
    #list_of_latency_data_transfer_performance.append(latency_data_transfer_performance)
    #list_of_normalized_latency_data_transfer_performance.append(normalized_latency_data_transfer_performance)
    #list_of_penalized_latency_data_transfer_performance.append(penalized_latency_data_transfer_performance)
    #list_of_normalized_penalized_latency_data_transfer_performance.append(normalized_penalized_latency_data_transfer_performance)
        


    
    if active_satellite == 'No Link':
        if not satellite_position_indices:
            active_satellite = 'No Link'
            print(f'No satellite applicable at timestamp {index}')

            # Save values for further analysis - active satellite - current availability 
            list_of_activated_sattellite_numbers.append(active_satellite)
            list_of_sattellite_availability_while_being_active.append(0)
            stored_throughput_active_satellite.append(0)

            list_of_applicable_satellites.append(sats_applicable)
            continue

        mission_level_instance = mission_level(elevation_cross_section, elevation, time_links, ranges, zenith, heights_AC,heights_SC, slew_rates, speeds_AC, index_elevation, t_micro, samples_channel_level, lengths)
        availability_vector, throughput, BER_outcome = mission_level_instance.calculate_mission_level()

        ## CREATE AVAIALABILITY AND THROUGHPUT VECTOR
        
        availability_vector = split_data_by_lengths(availability_vector, lengths)
        throughput = split_data_by_lengths(throughput, lengths)
        #print(throughput)
        #print(availability_vector)
        #print(BER_outcome)
        
        # Initiate performance parameters
        # Initiate performance parameters
        Availability_performance_instance = Availability_performance(time, availability_vector, lengths, num_satellites, max_length_applicability, acquisition_time_steps) 
        BER_performance_instance = ber_performance( time, throughput, lengths, num_satellites, max_length_applicability, acquisition_time_steps) 
        Throughput_performance_instance = Throughput_performance(time, throughput, lengths, num_satellites, acquisition_time_steps) 

        # Now call the method on the instance and initiliaze the six matrices
        #Availability
        availability_performance = Availability_performance_instance.calculate_availability_performance()
        normalized_availability_performance = Availability_performance_instance.calculate_normalized_availability_performance()

        penalized_availability_performance = Availability_performance_instance.calculate_penalized_availability_performance()
        normalized_penalized_availability_performance = Availability_performance_instance.calculate_normalized_penalized_availability_performance()

        #BER
        BER_performance = BER_performance_instance.calculate_BER_performance()
        normalized_BER_performance = BER_performance_instance.calculate_normalized_BER_performance()
        penalized_BER_performance = BER_performance_instance.calculate_penalized_BER_performance()
        normalized_penalized_BER_performance = BER_performance_instance.calculate_normalized_penalized_BER_performance()

        #Throughput
        throughput_performance = Throughput_performance_instance.calculate_throughput_performance()         
        normalized_throughput_performance = Throughput_performance_instance.calculate_normalized_throughput_performance()
        penalized_throughput_performance = Throughput_performance_instance.calculate_penalized_throughput_performance()
        normalized_penalized_throughput_performance = Throughput_performance_instance.calculate_normalized_penalized_throughput_performance()

        #print(throughput_performance)
        #print(normalized_throughput_performance)

        normalized_values = [normalized_availability_performance, normalized_BER_performance, normalized_cost_performance, normalized_latency_performance, normalized_latency_data_transfer_performance, normalized_throughput_performance]
        normalized_penalized_values = [normalized_penalized_availability_performance, normalized_penalized_BER_performance, normalized_penalized_cost_performance, normalized_latency_performance, normalized_penalized_latency_data_transfer_performance, normalized_penalized_throughput_performance]

        # LINK SELECTION INSTANCE 
        if num_opt_head == 1:
            link_selection_no_link_instance = link_selection_no_link(num_satellites, time, normalized_values, normalized_penalized_values, weights, satellite_position_indices, max_satellites, active_satellite)
            weighted_scores = link_selection_no_link_instance.calculate_weighted_performance(index)
            best_satellite, max_score, activated_sattellite_index, activated_sattellite_number = link_selection_no_link_instance.select_best_satellite(index)
        else: # IN THIS CASE FOR THE PENALIZED VALUES ALSO THE NORMALIZED_VALUES ARE USED
            link_selection_no_link_instance = link_selection_no_link(num_satellites, time, normalized_values, normalized_values, weights, satellite_position_indices, max_satellites, active_satellite)
            weighted_scores = link_selection_no_link_instance.calculate_weighted_performance(index)
            best_satellite, max_score, activated_sattellite_index, activated_sattellite_number = link_selection_no_link_instance.select_best_satellite(index)

    
        print("Satellite made active", activated_sattellite_number)
        
        
        # -------- UPDATE AND SAVE ACTIVE SATELLITE --------------
        active_satellite = activated_sattellite_index
        score_of_active_satellite = max_score
        

       
        # Update total scores for each satellite at their respective positions
        historical_scores = link_selection_no_link_instance.get_historical_scores()
        for idx, satellite_idx in enumerate(satellite_position_indices):
            if satellite_idx < max_satellites:
                list_of_total_scores_all_satellite[satellite_idx, index] = weighted_scores[idx]  
        
        print("------------------------------------------------")

        #------- THIS STATES IF THE SATELLITE MADE ACTIVE IS CURRENT AVAILABLE; USE FOR COMPLETE MISSION ANALYSIS---
        current_availability = availability_vector[best_satellite][0]
        current_throughput = throughput[best_satellite][0]
        #current_BER = BER_outcome[best_satellite][0]

        

        # Save values for further analysis - active satellite - current availability 
        list_of_activated_sattellite_numbers.append(activated_sattellite_number)
        list_of_sattellite_availability_while_being_active.append(current_availability)
        stored_throughput_active_satellite.append(current_throughput)
        #stored_BER_active_satellite.append(current_BER)

        list_of_applicable_satellites.append(sats_applicable)

        previous_satellite_position_indices = satellite_position_indices

        ##append performance parameters
        ##Availability
        #list_of_availability_performance.append(availability_performance)
        #list_of_normalized_availability_performance.append(normalized_availability_performance)
        #list_of_penalized_availability_performance.append(penalized_availability_performance)
        #list_of_normalized_penalized_availability_performance.append(normalized_penalized_availability_performance)
#
        ## BER
        #list_of_BER_performance.append(BER_performance)
        #list_of_normalized_BER_performance.append(normalized_BER_performance)
        #list_of_penalized_BER_performance.append(penalized_BER_performance)
        #list_of_normalized_penalized_BER_performance.append(normalized_penalized_BER_performance)
#
#
        ## Throughput
        #list_of_throughput_performance.append(throughput_performance)
        #list_of_normalized_throughput_performance.append(normalized_throughput_performance)
        #list_of_penalized_throughput_performance.append(penalized_throughput_performance)
        #list_of_normalized_penalized_throughput_performance.append(normalized_penalized_throughput_performance)


        continue

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------Version 2.1 integration---------------------------------------------------------------------------------
#-----------------------------------------------------Check if current position indices is equal to previous indices-----------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# If equal simply take values of previous throughput and BER

    if active_satellite != 'No Link' and satellite_position_indices == previous_satellite_position_indices :
        
        # CHECK IF IT IS RISING OR FALLING SATELLITE, REMOVING FALLING SATELLITE 
        # IF SATELLITE IS FALLING WE KNOW FOR SURE IT WILL NOT OUTPERFORM CURRENT ACTIVE SATELLITE
        marked_for_deletion = []
        for s in range(num_satellites):
            satellite_index = satellite_position_indices[s]
            # Ensure the current satellite is not the active one
            if satellite_index != active_satellite:
                # Ensure we have valid scores to compare (not NaN and sufficient historical data)
                current_score = list_of_total_scores_all_satellite[satellite_index][index-1]
                previous_score = list_of_total_scores_all_satellite[satellite_index][index-2]
                previous_previous_score = list_of_total_scores_all_satellite[satellite_index][index-3]
                previous_previous_previous_score = list_of_total_scores_all_satellite[satellite_index][index-4]

                # Check if both scores are numbers and that the current score is less than the previous score
                if not np.isnan(current_score) and not np.isnan(previous_score) and current_score <= previous_score <= previous_previous_score <=previous_previous_previous_score:
                    # Mark the satellite as inapplicable
                    marked_for_deletion.append(satellite_position_indices[s])
                    print(f"Satellite {satellite_index + 1} removed from analysis at time index {index}. Reason: Falling satellite performance.")
                    
        # DELETE INDECES WITH MARKED SATELLITES BASED ON FALLING CONDITION
        for marked_satellite in marked_for_deletion:
            satellite_deletion_index = satellite_position_indices.index(marked_satellite)
            list_of_total_scores_all_satellite[marked_satellite][index] = list_of_total_scores_all_satellite[marked_satellite][index-1]
            del satellite_position_indices[satellite_deletion_index]

        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # CHECK IF SUM OF INSTANTENEOUS PERFORMANCE PARAMETERS + MAX SCORE FOR PROPAGETED ONCE IS SMALLER THAN PREVIOUS SCORE. IF THIS IS THE CASE THERE IS A VERY SMALL CHANGE THAT THIS SATELLITE WILL OUTPERFORM THE CURRENT SATELLITE AND THUS IT WILL BE REMOVED 
        removed_satellites = [] 
        #active_satellite_index = satellite_position_indices.index(active_satellite)  # Get the index of the active satellite in the list

        for s in range(num_satellites):
            # Check condition and ensure the current satellite isn't the active one
            if (satellite_index != active_satellite and normalized_cost_performance[s] + normalized_latency_performance[s] + normalized_latency_data_transfer_performance[s] + weights[0] + weights[1] + weights[5] < list_of_scores_active_satellites[index-1]):
                removed_satellites.append(satellite_position_indices[s])  # Store only actually removed satellites

        # Print which satellites have been removed
        if removed_satellites:
            print("Satellites removed from the analysis:", removed_satellites)
        else:
            print("No satellites were removed based on potential outperforming current satellite conditions.")

        num_satellites = len(satellite_position_indices)
        
        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # CHECK IF ONLY ONE SATELLITE IS APPLICABLE AT THIS TIMESTAMP AND IF IT IS EQUAL TO CURRENT ACTIVE SATELLITE, IF SO: NO CALCULATION IS PERFORMED AND CURRENTLY ACTIVE SATELLITE IS KEPT ACTIVE
        previous_index = list_of_activated_satellite_index[index - 1]
        if isinstance(previous_index, int):  # Check if it's an integer
            # Perform operations that require a valid integer index
            if np.nansum(reference_positional_data) == 1 and reference_positional_data[previous_index] == 1:
                active_satellite = activated_sattellite_index
                print(f'Only 1 applicable satellite which is equal to previous active satellite, thus satellite {active_satellite+1} remains active')

            
                # Save values for further analysis - active satellite - current availability 
                list_of_activated_sattellite_numbers.append(activated_sattellite_number)
                list_of_sattellite_availability_while_being_active.append(1.0)
                stored_throughput_active_satellite.append(current_throughput)
             
                list_of_applicable_satellites.append(sats_applicable)

                #------- THIS STATES IF THE SATELLITE MADE ACTIVE IS CURRENT AVAILABLE; USE FOR COMPLETE MISSION ANALYSIS---
                current_availability = availability_vector[best_satellite][1]
                current_throughput = throughput[best_satellite][1]
                
                continue

        #------------------ INSTEAD OF PROPAGATION THE MISSION _LEVEL CALCULATIONS HERE, WE SIMPLY REMOVE THE FIRST INSTANCE HERE ------------------------
        for s in range(len(satellite_position_indices)):
            throughput[s] = throughput[s][1:]
            availability_vector[s] = availability_vector[s][1:]
            
        
        # Initiate performance parameters
        Availability_performance_instance = Availability_performance(time, availability_vector, lengths, num_satellites, max_length_applicability, acquisition_time_steps) 
        BER_performance_instance = ber_performance( time, throughput, lengths, num_satellites, max_length_applicability, acquisition_time_steps) 
        Throughput_performance_instance = Throughput_performance(time, throughput, lengths, num_satellites, acquisition_time_steps) 

        # Now call the method on the instance and initiliaze the six matrices
        #Availability
        availability_performance = Availability_performance_instance.calculate_availability_performance()
        normalized_availability_performance = Availability_performance_instance.calculate_normalized_availability_performance()

        penalized_availability_performance = Availability_performance_instance.calculate_penalized_availability_performance()
        normalized_penalized_availability_performance = Availability_performance_instance.calculate_normalized_penalized_availability_performance()


        #BER
        BER_performance = BER_performance_instance.calculate_BER_performance()
        normalized_BER_performance = BER_performance_instance.calculate_normalized_BER_performance()
        penalized_BER_performance = BER_performance_instance.calculate_penalized_BER_performance()
        normalized_penalized_BER_performance = BER_performance_instance.calculate_normalized_penalized_BER_performance()


        #Throughput
        throughput_performance = Throughput_performance_instance.calculate_throughput_performance()             # NORMAL EQUATION IS calculate_throughput_performance OPTIONAL: calculate_throughput_performance_including_decay
        normalized_throughput_performance = Throughput_performance_instance.calculate_normalized_throughput_performance()
        penalized_throughput_performance = Throughput_performance_instance.calculate_penalized_throughput_performance()
        normalized_penalized_throughput_performance = Throughput_performance_instance.calculate_normalized_penalized_throughput_performance()

        #CREATE COMBINED WEIGHTS AND NORMALIZED VALUES ARRAYS
        normalized_values = [normalized_availability_performance, normalized_BER_performance, normalized_cost_performance, normalized_latency_performance, normalized_latency_data_transfer_performance, normalized_throughput_performance]
        normalized_penalized_values = [normalized_penalized_availability_performance, normalized_penalized_BER_performance, normalized_penalized_cost_performance, normalized_latency_performance, normalized_penalized_latency_data_transfer_performance, normalized_penalized_throughput_performance]
        
        # LINK SELECTION INSTANCE 
        if num_opt_head == 1:
            link_selection_no_link_instance = link_selection_no_link(num_satellites, time, normalized_values, normalized_penalized_values, weights, satellite_position_indices, max_satellites, active_satellite)
            weighted_scores = link_selection_no_link_instance.calculate_weighted_performance(index)
            best_satellite, max_score, activated_sattellite_index, activated_sattellite_number = link_selection_no_link_instance.select_best_satellite(index)
        else: # IN THIS CASE FOR THE PENALIZED VALUES ALSO THE NORMALIZED_VALUES ARE USED
            link_selection_no_link_instance = link_selection_no_link(num_satellites, time, normalized_values, normalized_values, weights, satellite_position_indices, max_satellites, active_satellite)
            weighted_scores = link_selection_no_link_instance.calculate_weighted_performance(index)
            best_satellite, max_score, activated_sattellite_index, activated_sattellite_number = link_selection_no_link_instance.select_best_satellite(index)
    
        print("Satellite made active", activated_sattellite_number)
        
     
        
        active_satellite = activated_sattellite_index
        score_of_active_satellite = max_score


        historical_scores = link_selection_no_link_instance.get_historical_scores()
        # Update total scores for each satellite at their respective positions
        for idx, satellite_idx in enumerate(satellite_position_indices):
            if satellite_idx < max_satellites:
                list_of_total_scores_all_satellite[satellite_idx, index] = weighted_scores[idx]    

        print("------------------------------------------------")

        #------- THIS STATES IF THE SATELLITE MADE ACTIVE IS CURRENT AVAILABLE; USE FOR COMPLETE MISSION ANALYSIS---
        current_availability = availability_vector[best_satellite][0]
        current_throughput = throughput[best_satellite][0]
        #current_BER = BER_outcome[best_satellite][0]

        # Save active satellite number for further analysis 
        list_of_activated_sattellite_numbers.append(activated_sattellite_number)
        list_of_sattellite_availability_while_being_active.append(current_availability)
        stored_throughput_active_satellite.append(current_throughput)
        #stored_BER_active_satellite.append(current_BER)

        # APPEND APPLICABLE SATELLITE
        list_of_applicable_satellites.append(sats_applicable)

        # Update satllite position_indices
        # Needed to compare against next time index position index
        previous_satellite_position_indices = satellite_position_copy


        
        ##append performance parameters
        #list_of_availability_performance.append(availability_performance)
        #list_of_normalized_availability_performance.append(normalized_availability_performance)
        #list_of_penalized_availability_performance.append(penalized_availability_performance)
        #list_of_normalized_penalized_availability_performance.append(normalized_penalized_availability_performance)
#
        #        
        ##Availability
        #list_of_availability_performance.append(availability_performance)
        #list_of_normalized_availability_performance.append(normalized_availability_performance)
        #list_of_penalized_availability_performance.append(penalized_availability_performance)
        #list_of_normalized_penalized_availability_performance.append(normalized_penalized_availability_performance)
#
        ## BER
        #list_of_BER_performance.append(BER_performance)
        #list_of_normalized_BER_performance.append(normalized_BER_performance)
        #list_of_penalized_BER_performance.append(penalized_BER_performance)
        #list_of_normalized_penalized_BER_performance.append(normalized_penalized_BER_performance)
#
#
        ## Throughput
        #list_of_throughput_performance.append(throughput_performance)
        #list_of_normalized_throughput_performance.append(normalized_throughput_performance)
        #list_of_penalized_throughput_performance.append(penalized_throughput_performance)
        #list_of_normalized_penalized_throughput_performance.append(normalized_penalized_throughput_performance)

       
        
        continue

#----------------------------------------------------------------------------END OF -------------------------------------------------------------------------------------------
#----------------------------------------------------------------------Version 2.1 integration---------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if active_satellite != 'No Link':
        #sat_check_indices = satellite_position_indices
        # IF NO SATELLITES ARE APPLICABLE WE HAVE NO LINK
        if not satellite_position_indices:
            active_satellite = 'No Link'
            print(f'No satellite applicable at timestamp {index}')

            # Save values for further analysis - active satellite - current availability 
            list_of_activated_sattellite_numbers.append(active_satellite)
            list_of_sattellite_availability_while_being_active.append(0)
            stored_throughput_active_satellite.append(0)

            list_of_applicable_satellites.append(sats_applicable)
            #stored_BER_active_satellite.append(current_BER)
            copy_of_memory_satellite_indeces = satellite_memory_indeces
            continue

        #---------------------------------------------- BUILD FEATURE THAT FOR ACTIVE SATELLITE NUMBER WE TAKE 


        # CHECK IF IT IS RISING OR FALLING SATELLITE, REMOVING FALLING SATELLITE 
        # IF SATELLITE IS FALLING WE KNOW FOR SURE IT WILL NOT OUTPERFORM CURRENT ACTIVE SATELLITE
        marked_for_deletion = []
        for s in range(num_satellites):
            satellite_index = satellite_position_indices[s]
            # Ensure the current satellite is not the active one
            if satellite_index != active_satellite:
                # Ensure we have valid scores to compare (not NaN and sufficient historical data)
                current_score = list_of_total_scores_all_satellite[satellite_index][index-1]
                previous_score = list_of_total_scores_all_satellite[satellite_index][index-2]
                previous_previous_score = list_of_total_scores_all_satellite[satellite_index][index-3]
                previous_previous_previous_score = list_of_total_scores_all_satellite[satellite_index][index-4]

                # Check if both scores are numbers and that the current score is less than the previous score
                if not np.isnan(current_score) and not np.isnan(previous_score) and current_score <= previous_score <= previous_previous_score <=previous_previous_previous_score:
                    # Mark the satellite as inapplicable
                    marked_for_deletion.append(satellite_position_indices[s])
                    print(f"Satellite {satellite_index + 1} removed from analysis at time index {index}. Reason: Falling satellite performance.")
                    
        # DELETE INDECES WITH MARKED SATELLITES BASED ON FALLING CONDITION
        for marked_satellite in marked_for_deletion:
            satellite_deletion_index = satellite_position_indices.index(marked_satellite)
            list_of_total_scores_all_satellite[marked_satellite][index] = list_of_total_scores_all_satellite[marked_satellite][index-1]
            del satellite_position_indices[satellite_deletion_index]

        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # CHECK IF SUM OF INSTANTENEOUS PERFORMANCE PARAMETERS + MAX SCORE FOR PROPAGETED ONCE IS SMALLER THAN PREVIOUS SCORE. IF THIS IS THE CASE THERE IS A VERY SMALL CHANGE THAT THIS SATELLITE WILL OUTPERFORM THE CURRENT SATELLITE AND THUS IT WILL BE REMOVED 
        removed_satellites = [] 
        #active_satellite_index = satellite_position_indices.index(active_satellite)  # Get the index of the active satellite in the list

        for s in range(num_satellites):
            # Check condition and ensure the current satellite isn't the active one
            if (satellite_index != active_satellite and normalized_cost_performance[s] + normalized_latency_performance[s] + normalized_latency_data_transfer_performance[s] + weights[0] + weights[1] + weights[5] < list_of_scores_active_satellites[index-1]):
                removed_satellites.append(satellite_position_indices[s])  # Store only actually removed satellites

        # Print which satellites have been removed
        if removed_satellites:
            print("Satellites removed from the analysis:", removed_satellites)
        else:
            print("No satellites were removed based on potential outperforming current satellite conditions.")

        num_satellites = len(satellite_position_indices)
        
        
        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # CHECK IF ONLY ONE SATELLITE IS APPLICABLE AT THIS TIMESTAMP AND IF IT IS EQUAL TO CURRENT ACTIVE SATELLITE, IF SO: NO CALCULATION IS PERFORMED AND CURRENTLY ACTIVE SATELLITE IS KEPT ACTIVE
        previous_index = list_of_activated_satellite_index[index - 1]
        if isinstance(previous_index, int):  # Check if it's an integer
            # Perform operations that require a valid integer index
            if np.nansum(reference_positional_data) == 1 and reference_positional_data[previous_index] == 1:
                active_satellite = activated_sattellite_index
                print(f'Only 1 applicable satellite which is equal to previous active satellite, thus satellite {active_satellite+1} remains active')

            
                # Save values for further analysis - active satellite - current availability 
                list_of_activated_sattellite_numbers.append(activated_sattellite_number)
                list_of_sattellite_availability_while_being_active.append(1.0)
                stored_throughput_active_satellite.append(current_throughput)
             
                list_of_applicable_satellites.append(sats_applicable)
                copy_of_memory_satellite_indeces = satellite_memory_indeces
                continue

        #----- split up ranges according to size of the satellite applicability array -----
        #ranges_split = split_data_by_lengths(ranges, lengths)
        num_satellites = len(satellite_position_indices)



        #BUILD HERE SMART INSTANCE WHERE FOR THE PREVIOUS ALREADY ACTIVE SATELLITES IT SIMPLY SHIFT AN INDEX, INSTEAD OF RECALCULCULATING EVERYTHING.
        mission_level_instance = mission_level(elevation_cross_section, elevation, time_links, ranges, zenith, heights_AC,heights_SC, slew_rates, speeds_AC, index_elevation, t_micro, samples_channel_level, lengths)
        availability_vector, throughput, added_BER_outcome = mission_level_instance.calculate_mission_level()


        availability_vector = split_data_by_lengths(availability_vector, lengths)
        throughput = split_data_by_lengths(throughput, lengths)
        
        
       

        # Initiate performance parameters
        Availability_performance_instance = Availability_performance(time, availability_vector, lengths, num_satellites, max_length_applicability, acquisition_time_steps) 
        BER_performance_instance = ber_performance( time, throughput, lengths, num_satellites, max_length_applicability, acquisition_time_steps) 
        Throughput_performance_instance = Throughput_performance(time, throughput, lengths, num_satellites, acquisition_time_steps) 

        # Now call the method on the instance and initiliaze the six matrices
        #Availability
        availability_performance = Availability_performance_instance.calculate_availability_performance()
        normalized_availability_performance = Availability_performance_instance.calculate_normalized_availability_performance()

        penalized_availability_performance = Availability_performance_instance.calculate_penalized_availability_performance()
        normalized_penalized_availability_performance = Availability_performance_instance.calculate_normalized_penalized_availability_performance()


        #BER
        BER_performance = BER_performance_instance.calculate_BER_performance()
        normalized_BER_performance = BER_performance_instance.calculate_normalized_BER_performance()
        penalized_BER_performance = BER_performance_instance.calculate_penalized_BER_performance()
        normalized_penalized_BER_performance = BER_performance_instance.calculate_normalized_penalized_BER_performance()


        #Throughput
        throughput_performance = Throughput_performance_instance.calculate_throughput_performance()             # NORMAL EQUATION IS calculate_throughput_performance OPTIONAL: calculate_throughput_performance_including_decay
        normalized_throughput_performance = Throughput_performance_instance.calculate_normalized_throughput_performance()
        penalized_throughput_performance = Throughput_performance_instance.calculate_penalized_throughput_performance()
        normalized_penalized_throughput_performance = Throughput_performance_instance.calculate_normalized_penalized_throughput_performance()

        #CREATE COMBINED WEIGHTS AND NORMALIZED VALUES ARRAYS
        normalized_values = [normalized_availability_performance, normalized_BER_performance, normalized_cost_performance, normalized_latency_performance, normalized_latency_data_transfer_performance, normalized_throughput_performance]
        normalized_penalized_values = [normalized_penalized_availability_performance, normalized_penalized_BER_performance, normalized_penalized_cost_performance, normalized_latency_performance, normalized_penalized_latency_data_transfer_performance, normalized_penalized_throughput_performance]
        
        # LINK SELECTION INSTANCE 
        if num_opt_head == 1:
            link_selection_no_link_instance = link_selection_no_link(num_satellites, time, normalized_values, normalized_penalized_values, weights, satellite_position_indices, max_satellites, active_satellite)
            weighted_scores = link_selection_no_link_instance.calculate_weighted_performance(index)
            best_satellite, max_score, activated_sattellite_index, activated_sattellite_number = link_selection_no_link_instance.select_best_satellite(index)
        else: # IN THIS CASE FOR THE PENALIZED VALUES ALSO THE NORMALIZED_VALUES ARE USED
            link_selection_no_link_instance = link_selection_no_link(num_satellites, time, normalized_values, normalized_values, weights, satellite_position_indices, max_satellites, active_satellite)
            weighted_scores = link_selection_no_link_instance.calculate_weighted_performance(index)
            best_satellite, max_score, activated_sattellite_index, activated_sattellite_number = link_selection_no_link_instance.select_best_satellite(index)
        print("Satellite made active", activated_sattellite_number)
        
        #------- THIS STATES IF THE SATELLITE MADE ACTIVE IS CURRENT AVAILABLE; USE FOR COMPLETE MISSION ANALYSIS----------
        current_availability = availability_vector[best_satellite][0]
        
        active_satellite = activated_sattellite_index
        score_of_active_satellite = max_score


        historical_scores = link_selection_no_link_instance.get_historical_scores()
        # Update total scores for each satellite at their respective positions
        for idx, satellite_idx in enumerate(satellite_position_indices):
            if satellite_idx < max_satellites:
                list_of_total_scores_all_satellite[satellite_idx, index] = weighted_scores[idx]    

        print("------------------------------------------------")

        #------- THIS STATES IF THE SATELLITE MADE ACTIVE IS CURRENT AVAILABLE; USE FOR COMPLETE MISSION ANALYSIS---------- NOT USED YET
        current_availability = availability_vector[best_satellite][0]
        current_throughput = throughput[best_satellite][0]
        #current_BER = BER_outcome[best_satellite][0]

        # Save active satellite number for further analysis 
        list_of_activated_sattellite_numbers.append(activated_sattellite_number)
        list_of_sattellite_availability_while_being_active.append(current_availability)
        stored_throughput_active_satellite.append(current_throughput)
        #stored_BER_active_satellite.append(current_BER)

        # APPEND APPLICABLE SATELLITE
        list_of_applicable_satellites.append(sats_applicable)
        # Update satllite position_indices
        previous_satellite_position_indices = satellite_position_copy


        
        ##append performance parameters
        #list_of_availability_performance.append(availability_performance)
        #list_of_normalized_availability_performance.append(normalized_availability_performance)
        #list_of_penalized_availability_performance.append(penalized_availability_performance)
        #list_of_normalized_penalized_availability_performance.append(normalized_penalized_availability_performance)
#
        #        
        ##Availability
        #list_of_availability_performance.append(availability_performance)
        #list_of_normalized_availability_performance.append(normalized_availability_performance)
        #list_of_penalized_availability_performance.append(penalized_availability_performance)
        #list_of_normalized_penalized_availability_performance.append(normalized_penalized_availability_performance)
#
        ## BER
        #list_of_BER_performance.append(BER_performance)
        #list_of_normalized_BER_performance.append(normalized_BER_performance)
        #list_of_penalized_BER_performance.append(penalized_BER_performance)
        #list_of_normalized_penalized_BER_performance.append(normalized_penalized_BER_performance)
#
#
        ## Throughput
        #list_of_throughput_performance.append(throughput_performance)
        #list_of_normalized_throughput_performance.append(normalized_throughput_performance)
        #list_of_penalized_throughput_performance.append(penalized_throughput_performance)
        #list_of_normalized_penalized_throughput_performance.append(normalized_penalized_throughput_performance)


        # check for copy
        copy_of_memory_satellite_indeces = satellite_memory_indeces

    index =+1


#print(list_of_total_scores_all_satellite)
#print(list_of_activated_sattellite_numbers)
#print(list_of_sattellite_availability_while_being_active)
#print(stored_throughput_active_satellite)
#print(stored_BER_active_satellite)


dynamic_link_selection_visualization_instance = Dynamic_link_selection_visualization(
    active_satellites=list_of_activated_sattellite_numbers,
    max_satellites=max_satellites
)

dynamic_link_selection_visualization_instance.run()

#-----------------------------------------------------------------------------------------------------------------------------
print("-----------------------------------------------------------------------------------------------------------------------------")
print("---------------------------------------------------Total mission analysis----------------------------------------------------")


for i in range(1, len(list_of_activated_sattellite_numbers)):
    if list_of_activated_sattellite_numbers[i] != list_of_activated_sattellite_numbers[i - 1]:
        # Change detected, set the next acquisition_time_steps entries to zero
        for j in range(i + 1, min(i + 1 + acquisition_time_steps, len(list_of_sattellite_availability_while_being_active))):
            list_of_sattellite_availability_while_being_active[j] = 0

#stored_throughput_active_satellite *= list_of_sattellite_availability_while_being_active 

Mission_performance_instance = SatelliteLinkMetrics(list_of_activated_sattellite_numbers, list_of_sattellite_availability_while_being_active, step_size_link, acquisition_time_steps, total_mission_time = mission_duration, stored_throughput_active_satellite=stored_throughput_active_satellite)
mission_performance = Mission_performance_instance.calculate_metrics()
print(mission_performance)
#print(list_of_applicable_satellites)

#print("----------------------------------------------------list_of_availability_performance-----------------------------------------------")
#print(list_of_availability_performance)
#("-------------------------------------------------list_of_normalized_availability_performance--------------------------------------------------")
#print(list_of_normalized_availability_performance)
#("-----------------------------------------------list_of_penalized_availability_performance----------------------------------------------------")
#print(list_of_penalized_availability_performance)
#("----------------------------------------list_of_normalized_penalized_availability_performance-----------------------------------------------------------")
#print(list_of_normalized_penalized_availability_performance)
#print("------------------------------------------list_of_BER_performance---------------------------------------------------------")
#print(list_of_BER_performance)
#("--------------------------------------list_of_normalized_BER_performance-------------------------------------------------------------")
#print(list_of_normalized_BER_performance)
#("---------------------------------------list_of_penalized_BER_performance------------------------------------------------------------")
#print(list_of_penalized_BER_performance)
#("--------------------------------list_of_normalized_penalized_BER_performance-------------------------------------------------------------------")
#print(list_of_normalized_penalized_BER_performance)
#print("---------------------------------------list_of_throughput_performance------------------------------------------------------------")
#print(list_of_throughput_performance)
#print("-------------------------------------list_of_normalized_throughput_performance--------------------------------------------------------------")
#print(list_of_normalized_throughput_performance)
#print("-----------------------------------------list_of_penalized_throughput_performance----------------------------------------------------------")
#print(list_of_penalized_throughput_performance)
#print("--------------------------------------------list_of_normalized_penalized_throughput_performance-------------------------------------------------------")
#print(list_of_normalized_penalized_throughput_performance)


#-------------

# Ensure all lists have been defined and populated similarly
# This script assumes you have the lists already populated as shown in previous examples

# Prepare data for CSV
headers = [
    
    "Availability", "Normalized Availability", "Penalized Availability", "Normalized Penalized Availability",
    "BER", "Normalized BER", "Penalized BER", "Normalized Penalized BER",
    "Cost", "Normalized Cost", "Penalized Cost", "Normalized Penalized Cost",
    "Latency", "Normalized Latency", "Penalized Latency", "Normalized Penalized Latency",
    "Latency Data Transfer", "Normalized Latency Data Transfer", "Penalized Latency Data Transfer", "Normalized Penalized Latency Data Transfer",
    "Throughput", "Normalized Throughput", "Penalized Throughput", "Normalized Penalized Throughput"
]

# Zip together all the lists into rows
rows = zip(
    list_of_availability_performance, list_of_normalized_availability_performance, list_of_penalized_availability_performance, list_of_normalized_penalized_availability_performance,
    list_of_BER_performance, list_of_normalized_BER_performance, list_of_penalized_BER_performance, list_of_normalized_penalized_BER_performance,
    list_of_cost_performance, list_of_normalized_cost_performance, list_of_penalized_cost_performance, list_of_normalized_penalized_cost_performance,
    list_of_latency_performance, list_of_normalized_latency_performance, list_of_penalized_latency_performance, list_of_normalized_penalized_latency_performance,
    list_of_latency_data_transfer_performance, list_of_normalized_latency_data_transfer_performance, list_of_penalized_latency_data_transfer_performance, list_of_normalized_penalized_latency_data_transfer_performance,
    list_of_throughput_performance, list_of_normalized_throughput_performance, list_of_penalized_throughput_performance, list_of_normalized_penalized_throughput_performance
)

# list_of_applicable_satellites = list_of_applicable_satellites[1:]

# p = []

# for i in range(len(time)):

#    ap = np.array(deepcopy(list_of_applicable_satellites[i]))
#    print(np.nonzero(ap == 1))
#    print(list_of_availability_performance[i])
#    ap[np.nonzero(ap == 1)] = list_of_availability_performance[i]
#    p.append(ap)


# print(p)

df = pd.DataFrame({
    "timestamp": np.arange(len(time)),
    "Available link": list_of_sattellite_availability_while_being_active,
    "Throughput": stored_throughput_active_satellite
})

for sat_index in range(max_satellites):
    df["applicable sat. " + str(sat_index + 1)] = np.array(list_of_applicable_satellites)[:, sat_index]




print(list_of_activated_sattellite_numbers)
print(list_of_sattellite_availability_while_being_active)
print(stored_throughput_active_satellite)



    
df.to_csv('performance_data.csv', index=False)


print(f"Availability weight: {client_input_availability}, BER weight: {client_input_BER}, Cost weight: {client_input_cost}, Latency weight: {client_input_latency}, Latency Data Transfer weight: {client_input_latency_data_transfer}, Throughput weight: {client_input_throughput}")

# Create an instance of the visualizer
visualizer_instance = SatelliteDataVisualizer(time=time, activated_satellites=list_of_activated_sattellite_numbers, satellite_availability=list_of_sattellite_availability_while_being_active, throughput=stored_throughput_active_satellite, step_size_link=step_size_link, sats_applicable=list_of_applicable_satellites, max_num_satellites= max_satellites, list_of_total_scores_all_satellite=list_of_total_scores_all_satellite)

# Run all visualizations
visualizer_instance.plot_satellite_visibility_scatter_update()
visualizer_instance.plot_satellite_visibility_scatter_only_visibility()
#visualizer_instance.plot_cumulative_throughput()

updated_stored_throughput = [a * b for a, b in zip(list_of_sattellite_availability_while_being_active, stored_throughput_active_satellite)]
#print(updated_stored_throughput)

#----------------------------------------------------------------------------------------------------

# Create a figure with two subplots sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot the complete throughput data in the first subplot
ax1.plot(time, updated_stored_throughput, label='Throughput/s', color='orange')
ax1.set_title('Physical throughput performance')
ax1.set_ylabel('Throughput (bps)')
ax1.grid(True)
ax1.set_ylim(0, 2.55e9)  # Set y-axis to full scale to show all data

# Plot the zoomed in range between 2.45 and 2.5 billion in the second subplot
ax2.plot(time, updated_stored_throughput, label='Throughput/s', color='blue')
ax2.set_title('Zoomed physical throughput performance')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Throughput (bps)')
ax2.grid(True)
ax2.set_ylim(2.45e9, 2.55e9)  # Zoom in on the 2.45 to 2.55 billion range
accumulated_throughput = sum(updated_stored_throughput)*step_size_link/1E12 
# Adjust layout for better spacing
plt.tight_layout()

# Show accumulated throughput on the plot
fig.text(0.85, 0.55, f'Accumulated Throughput: {accumulated_throughput:.2f} T bits', fontsize=10, bbox={"facecolor":"white", "alpha":0.5, "pad":5})

plt.show()
