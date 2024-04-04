# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_applicable_links import applicable_links




class BER_performance():
    def __init__(self, time, link_geometry):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.Links_applicable = applicable_links(time=time)
        self.applicable_output, self.sats_applicable = self.Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
        self.time = time
        self.speed_of_light = speed_of_light

    def calculate_BER_performance(self):
      

        return self.BER_performance
    


    def calculate_normalized_BER_performance(self, data, potential_linktime):
        max_time = np.max(potential_linktime, axis=1)
        self.normalized_BER_performance = data / max_time[:, np.newaxis]

        return self.normalized_BER_performance
    

