import numpy as np
from matplotlib import pyplot as plt

import constants as cons

class aircraft:
    def __init__(self,
                 t,
                 states_array,
                 height = 10.0E3,
                 lat_init = 0.0,
                 lon_init = 0.0,
                 fixed_step_size = 10.0,
                 ):

        # Initial position
        self.R   = states_array[:,0] * 0
        self.lat = states_array[:,0] * 0
        self.lon = states_array[:,0] * 0
        self.pos = np.zeros((len(states_array[:,0]), 3))

        self. R[0] = cons.R_earth + height
        self.lat[0] = np.deg2rad(lat_init)
        self.lon[0] = np.deg2rad(lon_init)
        self.pos[0] = np.transpose(np.array((np.cos(self.lat[0]) * np.cos(self.lon[0]),
                                             np.cos(self.lat[0]) * np.sin(self.lon[0]),
                                             np.sin(self.lat[0]))) * self.R[0])
        self.t = t

    def propagate(self,
                  v = np.array([0.0, 0.0, 0.0]), # Velocity profile array (North, East, Down)
                  propagator_type = "straight"  # Type of propagation
                  ):

        # Propgation model for a straight flight with velocity profile v = [vx, vy, vz]
        # And with initial longitude, latitude and height
        if propagator_type == "straight":
            for i in range(1, len(self.pos)):

                latdot = v[0] /  self.R[i-1]
                londot = v[1] / (self.R[i-1] * np.cos(self.lat[i-1]))
                Rdot   = v[2]

                dt = self.t[i] - self.t[i-1]
                dlat = latdot * dt
                dlon = londot * dt
                dR   = Rdot * dt

                self.lat[i] = self.lat[i-1] + dlat
                self.lon[i] = self.lon[i-1] + dlon
                self.R[i]   = self.R[i-1]   + dR
                self.pos[i] = np.transpose(np.array((np.cos(self.lat[i]) * np.cos(self.lon[i]),
                                                     np.cos(self.lat[i]) * np.sin(self.lon[i]),
                                                     np.sin(self.lat[i]))) * self.R[i])

            self.heights = self.R - cons.R_earth

            # Correct for BC conditions (-pi < lat < pi)
            for i in range(len(self.lat)):
                if self.lat[i] > np.pi:
                    self.lat[i] -= 2*np.pi
                elif self.lat[i] < -np.pi:
                    self.lat[i] += 2*np.pi

            # Correct for BC conditions (-pi < lon < pi)
            for i in range(len(self.lon)):
                if self.lon[i] > np.pi:
                    self.lon[i] -= 2*np.pi
                elif self.lon[i] < -np.pi:
                    self.lon[i] += 2*np.pi

            return self.pos, self.heights, self.lat, self.lon, self.R

        # Other propagation methods can be added to this class
        elif propagator_type == "other":
            # Propagation model ...
            return



# #------------------------------------------------------------------------
# #------------------------POST-PROCESS-&-VISUALIZATION--------------------
# #------------------------------------------------------------------------
#
# fig = plt.figure(figsize=(6,6), dpi=125)
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title(f'AC trajectory around Earth')
#
# # Plot the positional state history
# ax.plot(pos_AC[:, 0], pos_AC[:, 1], pos_AC[:, 2], color='orange', label='aircraft')
# ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
#
# # Add the legend and labels, then show the plot
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.legend()
# plt.show()



