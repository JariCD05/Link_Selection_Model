import gzip
import csv
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import traffic
from traffic.data import opensky
from traffic.core import Traffic

from cartes.crs import valid_crs
from cartes.crs import EuroPP, PlateCarree
from cartes.utils.features import countries


print('Packages loaded')
#--------------------------------------------------
#--------------DOWNLOAD-CSV-FILE-------------------
#--------------------------------------------------

# # Define the icao 24 code of the airframe in the OpenSky database
# icao24_code = "ab58b2"
#
# OpenSky_filename = r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\states\states_2022-06-27-23.csv"
#
# with open(OpenSky_filename, newline='') as csvfile:
#     # Read the csv file
#     csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     # Convert the csv file to a list of rows, then to an array
#     rows = np.array(list(csv_reader))
#
#     # Filter the array with only rows that contain the selected icao24 code
#     aircraft_data = rows[rows[:,1] == icao24_code]





#--------------------------------------------------
#--------------REQUEST-FROM-OPEN-SKY-API-----------
#--------------------------------------------------

t0 = "2019-02-05 15:45"
t1 = "2019-02-06 05:45"
departure_airport = 'EGLL'
arrival_airport = 'KJFK'

flightlist = opensky.flightlist(
    start=t0,
    stop=t1,
    departure_airport=departure_airport,
    arrival_airport=arrival_airport)
print(flightlist.head())

chosen_flight = flightlist.iloc[0,:]

flight = opensky.history(
    start=str(chosen_flight.loc['firstseen']),
    stop=str(chosen_flight.loc['lastseen']),
    callsign=str(chosen_flight.loc['callsign']),
    count=True,
    return_flight=True,
)

# print(valid_crs('North America'))
from cartes.crs import EPSG_8857, EPSG_6931

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=PlateCarree()))
    ax.add_feature(countries())
    ax.set_extent([0, -80, 45, 55])  # DUB - LON
    # ax.set_extent([-160, -120, 20, 35]) # HONO - LAX
    # ax.set_extent([-10, 100, 50, 10]) # DUB - LON
    # ax.set_extent([-140, -60, 30, 55]) # JFK - LAX
    ax.spines["geo"].set_visible(False)

    # no specific method for that in traffic
    # but switch back to pandas DataFrame for manual plot
    flight.data.plot.scatter(
        ax=ax,
        x="longitude",
        y="latitude",
        c='count',
        transform=PlateCarree(),
        colormap='cividis',
        s=5,
    )
plt.show()

filename = r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\JFK_LAX.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\LON_JFK.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\SYD_MEL.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\DUB_LON.csv"

print('SAVE CSV FILE')
flight.to_csv(filename)

print('IMPORT CSV FILE')
flight_imported = pd.read_csv(filename)
print((flight_imported['groundspeed']).to_numpy())
