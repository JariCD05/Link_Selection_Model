import datetime
from datetime import date
from dateutil import parser

import numpy as np
import pandas
from matplotlib import pyplot as plt
from tudatpy.kernel import constants
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array

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
                  stepsize=1.0,
                  simulation_start_epoch = 0.0,
                  simulation_end_epoch = 1000.0,
                  height = False,
                  method = method_AC,
                  filename = False
                  ):
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
            R = R_earth + heights
            pos = np.transpose(np.array((np.cos(lat) * np.cos(lon),
                                          np.cos(lat) * np.sin(lon),
                                          np.sin(lat))) * R)
            ground_speed  = flight['groundspeed'].to_numpy() * 0.514444 #Convert kts to m/s
            vertical_speed = flight['vertical_rate'].to_numpy() * 0.00508 #Convert ft/min to m/s
            speed = np.sqrt(ground_speed**2 + vertical_speed**2)

            # properties_dict = dict()
            # for i in range(len(time)):
            #     properties_dict[time[i]] = np.array((lat[i], lon[i]))
            #
            # plt.plot(time, lat)
            # plt.show()
            # #
            # interpolator_settings = interpolators.lagrange_interpolation(8)
            # properties_interpolator = interpolators.create_one_dimensional_vector_interpolator(properties_dict, interpolator_settings)
            # lat_interpolator = interpolators.create_one_dimensional_vector_interpolator(lat_dict, interpolator_settings)
            # lon_interpolator = interpolators.create_one_dimensional_vector_interpolator(lon, interpolator_settings)
            # heights_interpolator = interpolators.create_one_dimensional_vector_interpolator(heights, interpolator_settings)
            # pos_interpolator = interpolators.create_one_dimensional_vector_interpolator(pos, interpolator_settings)
            # speed_interpolator = interpolators.create_one_dimensional_vector_interpolator(speed, interpolator_settings)
            #
            # lat_interpolated = dict()
            # lon_interpolated = dict()
            # heights_interpolated = dict()
            # pos_interpolated = dict()
            # speed_interpolated = dict()
            # properties_interpolated = dict()
            #
            # time_interpolated = np.arange(0, interval, 1.0)
            # for epoch in time_interpolated:
            #     # lat_interpolated[epoch] = lat_interpolator.interpolate(epoch)
            #     # lon_interpolated[epoch] = lon_interpolator.interpolate(epoch)
            #     # pos_interpolated[epoch] = pos_interpolator.interpolate(epoch)
            #     properties_interpolated[epoch] = properties_interpolator.interpolate(epoch)
            #
            # # lat = result2array(lat_interpolated)
            # # lon = result2array(lon_interpolated)
            # # pos = result2array(pos_interpolated)
            # properties = result2array(properties_interpolated)
            # lat = properties[:,0]
            # lon = properties[:,1]

            # plt.plot(time_interpolated, lat)
            # plt.show()


            return pos, heights, lat, lon, ground_speed, time


        # Propgation model for a straight flight with velocity profile v = [vx, vy, vz]
        # And with initial longitude, latitude and height
        elif method == "straight":
            print('Aircraft data from simplified straight flight')
            time = np.arange(simulation_start_epoch, simulation_end_epoch, stepsize)
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
                latdot = vel_AC[0] /  R[i-1]
                londot = vel_AC[1] / (R[i-1] * np.cos(lat[i-1]))
                Rdot   = vel_AC[2]

                dlat = latdot * stepsize
                dlon = londot * stepsize
                dR   = Rdot * stepsize
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

            speed = np.ones(len(time)) * speed_AC

            return pos, heights, lat, lon, speed, time

        def interpolate(self):
            return
