import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel import constants

from constants import *

class aircraft:
    def __init__(self,
                 lat_init = 0.0,
                 lon_init = 0.0,
                 simulation_start_epoch=0.0,
                 simulation_end_epoch=constants.JULIAN_DAY,
                 ):

        # Initial and terminal time
        self.lat_init = lat_init
        self.lon_init = lon_init


    def propagate(self,
                  simulation_start_epoch: float,
                  simulation_end_epoch: float,
                  v: np.array,  # Velocity profile array (North, East, Down)
                  states_array: np.array,
                  height: float,
                  propagator_type = "straight",  # Type of propagation
                  fixed_step_size = 10.0
                  ):

        # Propgation model for a straight flight with velocity profile v = [vx, vy, vz]
        # And with initial longitude, latitude and height
        if propagator_type == "straight":

            # Initial position
            R = states_array[:, 0] * 0
            lat = states_array[:, 0] * 0
            lon = states_array[:, 0] * 0
            pos = np.zeros((len(states_array[:, 0]), 3))
            R[0] = (R_earth + height)
            lat[0] = np.deg2rad(self.lat_init)
            lon[0] = np.deg2rad(self.lon_init)
            pos[0] = np.transpose(np.array((np.cos(lat[0]) * np.cos(lon[0]),
                                            np.cos(lat[0]) * np.sin(lon[0]),
                                            np.sin(lat[0]))) * R[0])


            interval = simulation_end_epoch - simulation_start_epoch
            for i in np.arange(1, int(interval/fixed_step_size)):

                latdot = v[0] /  R[i-1]
                londot = v[1] / (R[i-1] * np.cos(lat[i-1]))
                Rdot   = v[2]

                dlat = latdot * fixed_step_size
                dlon = londot * fixed_step_size
                dR   = Rdot * fixed_step_size
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

            return pos, heights, lat, lon

        # Other propagation methods can be added to this class
        elif propagator_type == "other":
            # Propagation model ...
            return




