import datetime
from datetime import date
from dateutil import parser

import numpy as np
import pandas
from matplotlib import pyplot as plt
from tudatpy.kernel import constants

from input import *

class aircraft:
    def __init__(self,
                 lat_init = 0.0,
                 lon_init = 0.0,
                 simulation_start_epoch=0.0,
                 simulation_end_epoch=constants.JULIAN_DAY,
                 ):

        self.lat_init = lat_init
        self.lon_init = lon_init

    def propagate(self,
                  simulation_start_epoch = 0.0,
                  simulation_end_epoch = 1000.0,
                  speed = False,  # Velocity profile array (North, East, Down)
                  height = False,
                  method = method_AC,
                  filename = False
                  ):
        print(method)
        if method == "opensky":
            print('Aircraft data from OPENSKY database')

            flight = pandas.read_csv(filename)
            t0 = parser.parse(str((flight['timestamp'].to_numpy())[0]))
            t1 = parser.parse(str((flight['timestamp'].to_numpy())[-1]))
            interval = (t1-t0).total_seconds()
            time = np.linspace(0, interval, len(flight['timestamp'].to_numpy()))

            lat = np.deg2rad(flight['latitude'].to_numpy())
            lon = np.deg2rad(flight['longitude'].to_numpy())

            heights = flight['altitude'].to_numpy() * 0.304 #Convert ft to m
            R = R_earth + height
            pos = np.transpose(np.array((np.cos(lat) * np.cos(lon),
                                          np.cos(lat) * np.sin(lon),
                                          np.sin(lat))) * R)
            ground_speed  = flight['groundspeed'].to_numpy() * 0.514444 #Convert kts to m/s
            vertical_speed = flight['vertical_rate'].to_numpy() * 0.00508 #Convert ft/min to m/s
            speed = np.sqrt(ground_speed**2 + vertical_speed**2)

            print(np.shape(time), 'ac time')
            return pos, heights, lat, lon, ground_speed, time


        # Propgation model for a straight flight with velocity profile v = [vx, vy, vz]
        # And with initial longitude, latitude and height
        elif method == "straight":
            print('Aircraft data from simplified straight flight')
            time = np.arange(start_time, end_time, step_size_link)
            # Initial position
            R   = np.zeros(len(time))
            lat = np.zeros(len(time))
            lon = np.zeros(len(time))
            pos = np.zeros((len(time), 3))
            R[0] = (R_earth + height)
            lat[0] = np.deg2rad(self.lat_init)
            lon[0] = np.deg2rad(self.lon_init)
            pos[0] = np.transpose(np.array((np.cos(lat[0]) * np.cos(lon[0]),
                                            np.cos(lat[0]) * np.sin(lon[0]),
                                            np.sin(lat[0]))) * R[0])


            interval = simulation_end_epoch - simulation_start_epoch
            for i in range(1, len(time)):

                latdot = v[0] /  R[i-1]
                londot = v[1] / (R[i-1] * np.cos(lat[i-1]))
                Rdot   = v[2]

                dlat = latdot * step_size_link
                dlon = londot * step_size_link
                dR   = Rdot * step_size_link
                lat[i] = lat[i-1] + dlat
                lon[i] = lon[i-1] + dlon
                R[i]   = R[i-1]   + dR
                

                pos[i] = np.transpose(np.array((np.cos(lat[i]) * np.cos(lon[i]),
                                                 np.cos(lat[i]) * np.sin(lon[i]),
                                                 np.sin(lat[i]))) * R[i])
            heights = R - R_earth


            # Correct for BC conditions (-pi < lat < pi)
            for i in range(len(lat)):
                if lat[i] > np.pi:
                    lat[i] -= 2*np.pi
                elif lat[i] < -np.pi:
                    lat[i] += 2*np.pi

            # Correct for BC conditions (-pi < lon < pi)
            for i in range(len(lon)):
                if lon[i] > np.pi:
                    lon[i] -= 2*np.pi
                elif lon[i] < -np.pi:
                    lon[i] += 2*np.pi

            return pos, heights, lat, lon, speed, time

        def interpolate(self):
            return

# aircraft = aircraft()
# aircraft.propagate(method="OpenSky_database")
