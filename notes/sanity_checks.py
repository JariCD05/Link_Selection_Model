#print(f"Time index {index}: visible Satellites - {sats_visible}")
#print(f"Time index {index}: applicable Satellites - {sats_applicable}")
#sanity check if applicability works
##if not np.array_equal(sats_visible, sats_applicable, equal_nan=True):
#print(f"Time index {index}: Difference detected - Visible: {sats_visible}, Applicable: {sats_applicable}")
import numpy as np
def remove_nan_values(input_array):
    """Remove all NaN values from a numpy array and return the filtered array."""
    filtered_array = input_array[~np.isnan(input_array)]
    return filtered_array

# Using the function
sample_array = np.array([3, np.nan, 7, np.nan, 10])
clean_array = remove_nan_values(sample_array)
print(clean_array)


#----- PROPAGTED GEOMETRICAL DATA FOR APPLICABLE SATELLITES

#time_links = flatten(applicable_propagated_output['time'])
#time_links_hrs = [t / 3600.0 for t in time_links]
#ranges = flatten(applicable_propagated_output['ranges'])
#elevation = flatten(applicable_propagated_output['elevation'])
#zenith = flatten(applicable_propagated_output['zenith'])
#slew_rates = flatten(applicable_propagated_output['slew rates'])
#heights_SC = flatten(applicable_propagated_output['heights SC'])
#heights_AC = flatten(applicable_propagated_output['heights AC'])
#speeds_AC = flatten(applicable_propagated_output['speeds AC'])
#
#
#time_per_link       = applicable_propagated_output['time']
#time_per_link_hrs   = time_links / 3600.0
#ranges_per_link     = applicable_propagated_output['ranges'    ]
#elevation_per_link  = applicable_propagated_output['elevation' ]
#zenith_per_link     = applicable_propagated_output['zenith'    ]
#slew_rates_per_link = applicable_propagated_output['slew rates']
#heights_SC_per_link = applicable_propagated_output['heights SC']
#heights_AC_per_link = applicable_propagated_output['heights AC']
#speeds_AC_per_link  = applicable_propagated_output['speeds AC' ]
#print(ranges)