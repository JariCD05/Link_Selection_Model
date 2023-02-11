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
                  simulation_start_epoch = 0.0,
                  simulation_end_epoch = 1000.0,
                  v = 250.0,  # Velocity profile array (North, East, Down)
                  # states_array = 0.0,
                  height = 10.0E3,
                  fixed_step_size = 10.0,
                  method = method_AC
                  ):

        # Propgation model for a straight flight with velocity profile v = [vx, vy, vz]
        # And with initial longitude, latitude and height
        if method == "straight":
            time = np.arange(simulation_start_epoch, simulation_end_epoch, fixed_step_size)
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

            return pos, heights, lat, lon, time

        # Other propagation methods can be added to this class
        elif method == "coordinate_based":
            # Propagation model ...
            # Amsterdam coord: lat, lon (52.3676, 4.9041)
            # New York: lat, lon (40.7128, -74.0060)

            self.lat_init = 52.3676
            self.lon_init = 4.9041
            self.lat_fin = 40.7128
            self.lon_fin = -74.0060
            dt = fixed_step_size



        elif method == "OpenSky_database":
            import csv

            icao24_code = "ab58b2"
            OpenSky_filename = r"C:\Users\wiege\Documents\TUDelft_Spaceflight\Thesis\aircraft_data\states\states_2022-06-27-23.csv"

            print('OpenSky')
            with open(OpenSky_filename, newline='') as csvfile:
                # Read the csv file
                csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                # Convert the csv file to a list of rows, then to an array
                rows = np.array(list(csv_reader))
                # Filter the array with only rows that contain the selected icao24 code
                aircraft_data = rows[rows[:, 1] == icao24_code]

            self.time = (aircraft_data[:, 0]).astype(float) - (aircraft_data[0, 0]).astype(float)
            self.lat = np.deg2rad(aircraft_data[:, 2].astype(float))
            self.lon = np.deg2rad(aircraft_data[:, 3].astype(float))
            self.speed = aircraft_data[:, 4].astype(float)
            self.heights = aircraft_data[:, -4].astype(float)
            self.R = aircraft_data[:, -3].astype(float) + R_earth
            self.pos = np.transpose(np.array((np.cos(self.lat) * np.cos(self.lon),
                                              np.cos(self.lat) * np.sin(self.lon),
                                                                 np.sin(self.lat))) * self.R)

            # fig = plt.figure(figsize=(6, 6), dpi=125)
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot(self.pos[:, 0],
            #         self.pos[:, 1],
            #         self.pos[:, 2], color='black', label='aircraft')
            # # Draw Earth as a blue dot in origin
            # ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue', s=100)
            # plt.show()

            print(aircraft_data[0, :])
            return self.pos, self.heights, self.lat, self.lon, self.time


        def interpolate(self):
            return

# aircraft = aircraft()
# aircraft.propagate(method="OpenSky_database")
