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

# IMPORT FROM simp_agent.py
from simp_agent import Simple_Language_Agent

# IMPORT MESA LIBRARIES
from mesa import Model
from mesa.time import RandomActivation, SimultaneousActivation, StagedActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

class Simple_Language_Model(Model):
    def __init__(self, num_people, width=10, height=10, alpha=1.1,
                 max_people_factor=5):
        self.num_people = num_people
        self.grid_width = width
        self.grid_height = height
        self.alpha = alpha
        # define grid and schedule
        self.grid = MultiGrid(height, width, False)
        self.schedule = RandomActivation(self)

        # INITIALIZE KNOWN PEOPLE NETWORK => label is lang spoken
        self.known_people_network = nx.DiGraph()
        #        self.known_people_network.add_edge('A','B', lang_spoken = 'cat')
        #        self.known_people_network.add_edge('A','C', lang_spoken = 'spa')
        #        self.known_people_network['A']['C']['lang_spoken']
        #        self.known_people_network['B']['A'] = 'cat'

        # INITIALIZE FRIENDSHIP NETWORK
        self.friendship_network = nx.Graph()  # sort by friendship intensity
        #       sorted(self.friendship_network[n_1].items(),
        #                    key=lambda edge: edge[1]['link_strength'],
        #                    reverse = True)

        # INITIALIZE FAMILY NETWORK
        self.family_networks = nx.DiGraph()

        S = 0.5
        for id_, coord in zip(range(num_people), itertools.product(range(self.grid_width),
                                                                   range(self.grid_height))
                              ):
            lang = np.random.choice([0,1,2], p=[0.25,0.65,0.1])
            a = Simple_Language_Agent(id_, lang, S)
            self.add_agent(a, coord)

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={"count_lang": lambda m: m.get_lang_stats()}
        )


    def add_agent(self, a, coords):
        # add agent to grid and schedule
        self.schedule.add(a)
        self.grid.place_agent(a, (coords[0], coords[1]))
        ## add agent node to all networks
        self.known_people_network.add_node(a)
        self.friendship_network.add_node(a)
        self.family_networks.add_node(a)


    def visualize_agents_attrs(self, ag_attr):
        """Plots linguistic agents attributes on a 2D grid

        Args:
            * ag_attr : a quantifiable attribute of the lang agent class
        """

        ag_and_coords = [(getattr(ag, ag_attr), ag.pos[0], ag.pos[1])
                         for ag in self.schedule.agents]
        ag_and_coords = np.array(ag_and_coords)
        df_exper = pd.DataFrame({'x': ag_and_coords[:, 1],
                                 'y': ag_and_coords[:, 2]})
        df_exper['values'] = ag_and_coords[:, 0]
        df_avg = df_exper.groupby(['x', 'y']).mean()
        masked_array = np.ma.array(df_avg.unstack(), mask=np.isnan(df_avg.unstack()))
        cmap = matplotlib.cm.viridis
        cmap.set_bad('white', 1.)
        plt.pcolor(masked_array, cmap=cmap)
        plt.colorbar()
        plt.title(ag_attr)
        plt.show()

    def get_lang_stats(self):
        ag_lang_list = [ag.language for ag in self.schedule.agents]
        lang_counts = Counter(ag_lang_list)
        return lang_counts[1]


    def step(self):
        #self.get_lang_stats()
        self.datacollector.collect(self)
        self.schedule.step()


    def run_model(self, steps):
        for _ in range(steps):
            self.step()





        