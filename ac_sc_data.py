import gzip
import csv
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

print('Packages loaded')

#----------------------AIRCRAFT--------------------
#--------------------------------------------------
#--------------REQUEST-FROM-OPEN-SKY-API-----------
#--------------------------------------------------
import traffic
from traffic.data import opensky
from traffic.core import Traffic

from cartes.crs import valid_crs
from cartes.crs import EuroPP, PlateCarree
from cartes.utils.features import countries



t0 = "2019-02-01 12:45"
t1 = "2019-03-06 05:45"
departure_airport = 'ESSA'
arrival_airport = 'ENGM'

flightlist = opensky.flightlist(
    start=t0,
    stop=t1,
    departure_airport=departure_airport,
    arrival_airport=arrival_airport)
print(flightlist.head())

chosen_flight = flightlist.iloc[1,:]

flight = opensky.history(
    start=str(chosen_flight.loc['firstseen']),
    stop=str(chosen_flight.loc['lastseen']),
    callsign=str(chosen_flight.loc['callsign']),
    count=True,
    return_flight=True,
)

from cartes.crs import EPSG_8857, EPSG_6931

with plt.style.context("traffic"):
    fig, ax = plt.subplots(subplot_kw=dict(projection=PlateCarree()))
    ax.add_feature(countries())
    ax.set_extent([10, 20, 55, 63])  # OSLO - STOCKH (Verification)
    # ax.set_extent([-80, 0, 45, 55])  # LON - JFK (Verification)
    # ax.set_extent([-10, 10, 55, 40])  # BAR - LON (Verification)
    # ax.set_extent([140, 155, -42, -33])  # SYD - MEL (Verification)
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

filename = r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\ac_sc_data\traffic_trajectories\OSL_STK.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\SYD_MEL.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\BAR_LON.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\JFK_LAX.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\LON_JFK.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\SYD_MEL.csv"
    # r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\traffic_trajectories\DUB_LON.csv"

print('SAVE CSV FILE')
flight.to_csv(filename)

#---------------------SPACECRAFT-------------------
#--------------------------------------------------
#--------REQUEST-FROM-OPEN-CELESTRAK-API-----------
#--------------------------------------------------

import json
import requests
from sgp4.api import Satrec, jday, days2mdhms


# tle_urls = ['https://www.celestrak.com/NORAD/elements/oneweb.txt']
tle_urls = ['https://www.celestrak.com/NORAD/elements/starlink.txt']


def download_tle():
    tle_json = []
    for url in tle_urls:
        request = requests.get(url)
        tmp_dict = {}
        for i in request.text.split('\n'):
            try:
                if i[0] == '1':
                    tmp_dict['tle_1'] = i.strip()
                elif i[0] == '2':
                    tmp_dict['tle_2'] = i.strip()
                else:
                    tmp_dict['satellite_name'] = i.strip()

                if "tle_1" in tmp_dict and "tle_2" in tmp_dict and "satellite_name" in tmp_dict:
                    tle_json.append(tmp_dict)
                    tmp_dict = {}
                else:
                    pass
            except:
                pass
    filename = 'starlink_tle.json'
    with open(filename, 'w') as f:
        json.dump(tle_json, f, indent=3)
        print('[+] Downloaded TLE data in '+str(filename))

if __name__ == '__main__':
    print('[+] Downloading TLE data...')
    download_tle()

# f = open('oneweb_tle.json', "r")
# data = json.loads(f.read())
# for i in data:
#     satellite = Satrec.twoline2rv(i['tle_1'], i['tle_2'])
#
#     jd = satellite.jdsatepoch
#     fr = satellite.jdsatepochF
# f.close()
