import gzip
import csv
import pandas as pd
import sys
import numpy as np
from opensky_api import OpenSkyApi, StateVector

# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion, frame_conversion, time_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

api = OpenSkyApi()


#--------------------------------------------------
#--------------DOWNLOAD-CSV-FILE-------------------
#--------------------------------------------------

# Define the icao 24 code of the airframe in the OpenSky database
icao24_code = "ab58b2"

OpenSky_filename = r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\states\states_2022-06-27-23.csv"

with open(OpenSky_filename, newline='') as csvfile:
    # Read the csv file
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    # Convert the csv file to a list of rows, then to an array
    rows = np.array(list(csv_reader))

    # Filter the array with only rows that contain the selected icao24 code
    aircraft_data = rows[rows[:,1] == icao24_code]





#--------------------------------------------------
#--------------REQUEST-FROM-OPEN-SKY-API-----------
#--------------------------------------------------

# THESE ARE SEVERAL ADS-B CODES THAT CORRESPOND TO AIRFRAMES
# icao24: 4951d9, type: A320, registration: CS-TNY, airline: AirPortugal
# icao24: 3807fa, type: A380


# bbox = (min latitude, max latitude, min longitude, max longitude)
# time = time_conversion.calendar_date_to_julian_day_since_epoch()

# state_vector = StateVector()
duration = 1 # hour
time_secs_0 = 1513165649
time_secs_1 = time_secs_0 + duration * 3600
time_secs_list = np.arange(time_secs_0, time_secs_1, 1)

# for t in range(time_secs_0, time_secs_0+10):
#     print(t)
#     print(api.get_states(time_secs=time_secs_0, icao24='3807fa'))

# states = api.get_states(time_secs=time_secs_0, icao24='3807fa')
#
#
# for s in states.states:
#     print(s)
#     print("(%r, %r, %r, %r)" % (s.time, s.latitude, s.baro_altitude, s.velocity))




