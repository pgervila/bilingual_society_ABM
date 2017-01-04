# IMPORT RELEVANT LIBRARIES
from importlib import reload
from math import ceil
import random
import itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist
import networkx as nx
# IMPORT MESA LIBRARIES
from mesa import Model
from mesa.time import RandomActivation, SimultaneousActivation, StagedActivation
from mesa.space import MultiGrid, ContinuousSpace
from mesa.datacollection import DataCollector

class Language_Model(Model):
    def __init__(self, num_people, width=100, height=100, 
                 max_people_factor=5):
        # initial number of people
        self.num_people = num_people
        self.alpha = alpha
        self.grid_width = width
        self.grid_height = height
        
        self.grid = MultiGrid(height, width, False)
        self.schedule = RandomActivation(self)
        