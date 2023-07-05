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
from helper_functions import *

class aircraft:
    def __init__(self,
                 lat_init = 0.0,
                 lon_init = 0.0,
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
            flight = pandas.read_csv(filename)
            t0 = parser.parse(str((flight['timestamp'].to_numpy())[0]))
            t1 = parser.parse(str((flight['timestamp'].to_numpy())[-1]))
            interval = (t1-t0).total_seconds()
            time_0 = np.zeros(len(flight['timestamp'].to_numpy()))

            for i in range(len(flight['timestamp'].to_numpy())):
                date_time_str = flight['timestamp'].to_numpy()[i][:-6]
                dt = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                t = dt.hour * 3600 + dt.minute * 60 + dt.second
                time_0[i] = t
            time_0 -= time_0[0]

            # lat = flight['latitude'].to_numpy()
            # lon = flight['longitude'].to_numpy()
            lat = np.deg2rad(flight['latitude'].to_numpy())
            lon = np.deg2rad(flight['longitude'].to_numpy())
            heights = flight['altitude'].to_numpy() * 0.304 #Convert ft to m
            ground_speed  = flight['groundspeed'].to_numpy() * 0.514444 #Convert kts to m/s
            vertical_speed = flight['vertical_rate'].to_numpy() * 0.00508 #Convert ft/min to m/s

            R = R_earth + heights
            pos = np.transpose(np.array((np.cos(lat) * np.cos(lon),
                                         np.cos(lat) * np.sin(lon),
                                         np.sin(lat))) * R).tolist()
            speed = np.sqrt(ground_speed**2 + vertical_speed**2)
            pos = np.asarray(pos)

            # # Option to plot one of the above state variables of the aircraft (lat, lon, heights, ground_speed, vertical_speed)
            # fig, ax = plt.subplots(1, 2)
            # # fig.suptitle('Aircraft trajectory \n'
            # #              +aircraft_filename_load[84:-4], fontsize=20)
            # ax[0].scatter(time_0/60.0, speed, s=8, label='raw')
            # ax[1].scatter(time_0/60.0, heights * 1.0E-3, s=8, label='raw')

            # ------------------------------------------------------------------------
            # -------------------------------INTERPOLATE------------------------------
            # ------------------------------------------------------------------------
            time = np.arange(0, interval, stepsize)
            lat     = interpolator(time_0, lat,       time, interpolation_type='linear')
            lon     = interpolator(time_0, lon,       time, interpolation_type='linear')
            heights = interpolator(time_0, heights,   time, interpolation_type='linear')
            pos_x   = interpolator(time_0, pos[:,0],  time, interpolation_type='linear')
            pos_y   = interpolator(time_0, pos[:, 1], time, interpolation_type='linear')
            pos_z   = interpolator(time_0, pos[:, 2], time, interpolation_type='linear')
            pos     = np.transpose(np.vstack((pos_x, pos_y, pos_z)))
            speed   = interpolator(time_0, speed, time, interpolation_type='linear')

            # ax[0].scatter(time/60.0, speed, s=1, label='interpolated')
            # ax[1].scatter(time/60.0, heights * 1.0E-3, s=1, label='interpolated')
            # ax[0].set_ylabel('Speed (m/s)', fontsize=10)
            # ax[1].set_ylabel('Altitude (km)', fontsize=10)
            # ax[1].set_xlabel('Time (min)', fontsize=10)
            # ax[0].set_xlabel('Time (min)', fontsize=10)
            # ax[0].legend(fontsize=10)
            # # ax[1].legend(fontsize=10)
            # ax[0].grid()
            # ax[1].grid()
            # plt.show()

            print('AIRCRAFT PROPAGATION MODEL')
            print('------------------------------------------------')
            print('Aircraft positional data retrieved from OPENSKY database')
            print('Initial latitude and longitude: (' + str(np.round(np.rad2deg(lat[0]), 1)) + 'deg, ' +
                                                        str(np.round(np.rad2deg(lon[0]), 1)) + 'deg)')
            print('Final latitude and longitude  : ('   + str(np.round(np.rad2deg(lat[-1]), 1)) + 'deg, ' +
                                                        str(np.round(np.rad2deg(lon[-1]), 1)) + 'deg)')
            print('Average altitude             : ' + str(np.round(heights[heights != 'nan'].mean() / 1e3, 2)) + ' km')
            print('Average flight speed          : '+str(np.round(speed.mean(),2))+' m/s')
            print('------------------------------------------------')
            return pos, heights, lat, lon, speed, time


        # Propgation model for a straight flight with velocity profile v = [vx, vy, vz]
        # And with initial longitude, latitude and height
        elif method == "straight":
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
            print('AIRCRAFT PROPAGATION MODEL')
            print('------------------------------------------------')
            print('Aircraft positional data retrieved from simplified straight flight algorithm')
            print('Initial latitude and longitude: (' + str(np.round(np.rad2deg(lat[0]), 1)) + 'deg, ' +
                  str(np.round(np.rad2deg(lon[0]), 1)) + 'deg)')
            print('Final latitude and longitude  : (' + str(np.round(np.rad2deg(lat[-1]), 1)) + 'deg, ' +
                  str(np.round(np.rad2deg(lon[-1]), 1)) + 'deg)')
            print('Constant altitude             : ' + str(np.round(heights.mean() / 1e3, 2)) + ' km')
            print('Constant flight speed         : ' + str(speed_AC)+' m/s')
            print('------------------------------------------------')
            return pos, heights, lat, lon, speed, time

        elif method == 'fixed':
            time = np.arange(simulation_start_epoch, simulation_end_epoch, stepsize)

            lat = np.ones(len(time)) * np.deg2rad(self.lat_init)
            lon = np.ones(len(time)) * np.deg2rad(self.lon_init)
            R   = np.ones(len(time)) * (R_earth + height)
            heights = R - R_earth
            pos = np.transpose(np.array((np.cos(lat[0])    * np.cos(lon[0]),
                                            np.cos(lat[0]) * np.sin(lon[0]),
                                            np.sin(lat[0])                 )) * R)
            speed = np.zeros(len(time))
            return pos, heights, lat, lon, speed, time