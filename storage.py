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
                continue

        #----- split up ranges according to size of the satellite applicability array -----
        #ranges_split = split_data_by_lengths(ranges, lengths)
        num_satellites = len(satellite_position_indices)



        #BUILD HERE SMART INSTANCE WHERE FOR THE PREVIOUS ALREADY ACTIVE SATELLITES IT SIMPLY SHIFT AN INDEX, INSTEAD OF RECALCULCULATING EVERYTHING.
        mission_level_instance = mission_level(elevation_cross_section, elevation, time_links, ranges, zenith, heights_AC,heights_SC, slew_rates, speeds_AC, index_elevation, t_micro, samples_channel_level, lengths)
        availability_vector, throughput, added_BER_outcome = mission_level_instance.calculate_mission_level()


        availability_vector = split_data_by_lengths(availability_vector, lengths)
        throughput = split_data_by_lengths(throughput, lengths)
        
        
       
