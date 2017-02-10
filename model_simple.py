# IMPORT RELEVANT LIBRARIES
import os
from importlib import reload
from math import ceil
import random
import itertools
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pylab as plt
import matplotlib.animation as animation
from scipy.spatial.distance import pdist
import networkx as nx

#import library to save any python data type to HDF5
import deepdish as dd

# import progress bar
import pyprind


# IMPORT FROM simp_agent.py
from agent_simple import Simple_Language_Agent

# IMPORT MESA LIBRARIES
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# need to subclass because of bug in last available mesa release
class RandomActivation_modif(RandomActivation):
    def step(self):
        random.shuffle(self.agents)
        for agent in self.agents[:]:
            agent.step()
        self.steps += 1
        self.time += 1

class Simple_Language_Model(Model):
    def __init__(self, num_people, avg_max_mem=20, width=5, height=5, max_people_factor=5,
                 init_lang_distrib=[0.25, 0.65, 0.1], num_cities=10, lang_ags_sorted_by_dist=True,
                 lang_ags_sorted_in_clust=True):
        self.num_people = num_people
        self.avg_max_mem = avg_max_mem
        self.grid_width = width
        self.grid_height = height
        self.max_people_factor = max_people_factor
        self.init_lang_distrib = init_lang_distrib
        self.num_cities = num_cities
        self.lang_ags_sorted_by_dist = lang_ags_sorted_by_dist
        self.lang_ags_sorted_in_clust = lang_ags_sorted_in_clust
        self.clust_centers = None
        self.cluster_sizes = None

        # define grid and schedule
        self.grid = MultiGrid(height, width, False)
        self.schedule = RandomActivation_modif(self)

        ## RANDOMLY DEFINE ALL CITY-CENTERS COORDS (CITY == HOMES, JOB CENTERS and SCHOOLS)
        # first define available points as pct of squared grid length
        grid_pct_list = np.linspace(0.1, 0.9, 100) # avoid edges
        # now generate the cluster centers (CITIES-VILLAGES)
        self.clust_centers = np.random.choice(grid_pct_list,size=(self.num_cities, 2),replace=False)
        if lang_ags_sorted_by_dist:
            self.clust_centers = sorted(self.clust_centers,
                                        key=lambda x:pdist([x,[0,0]])
                                       )

        #define container for available ids
        self.set_available_ids = set(range(num_people,
                                           self.max_people_factor * self.num_people)
                                     )


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
        self.family_network = nx.DiGraph()

        # ADD ALL AGENTS TO GRID AND SCHEDULE
        S = 0.5
        if (not lang_ags_sorted_by_dist) and (not lang_ags_sorted_in_clust):
            for id_ in range(self.num_people):
                x = random.randrange(self.grid_width)
                y = random.randrange(self.grid_height)
                coord = (x,y)
                lang = np.random.choice([0,1,2], p=self.init_lang_distrib)
                ag = Simple_Language_Agent(self, id_, lang, S)
                self.add_agent(ag, coord)
        else:
            self.create_lang_agents()

        # DATA COLLECTOR
        self.datacollector = DataCollector(
            model_reporters={"count_spa": lambda m: m.get_lang_stats(0),
                             "count_bil": lambda m: m.get_lang_stats(1),
                             "count_cat": lambda m: m.get_lang_stats(2),
                             "total_num_agents": lambda m:len(m.schedule.agents),
                             "biling_evol_h": lambda m:m.get_bilingual_global_evol('heard'),
                             "biling_evol_s": lambda m: m.get_bilingual_global_evol('spoken')
                             },
            agent_reporters={"pct_cat_in_biling": lambda a: a.lang_freq['cat_pct_h'],
                             "pct_spa_in_biling": lambda a: 1 - a.lang_freq['cat_pct_h']}
        )

    def add_agent(self, a, coords):
        """Method to add a given agent to grid, schedule and system networks

        Arguments:
            * a : agent class instance
            * coords : agent location on grid (2-D tuple of integers)

        """
        # add agent to grid and schedule
        self.schedule.add(a)
        self.grid.place_agent(a, (coords[0], coords[1]))
        ## add agent node to all networks
        self.known_people_network.add_node(a)
        self.friendship_network.add_node(a)
        self.family_network.add_node(a)

    def compute_cluster_sizes(self, min_size=20, small_large_pcts=[0.6, 0.4]):
        """ Method to compute sizes of each agent cluster

        Arguments:
            * min_size: minimum accepted cluster size ( integer)
            * small_large_pcts: percentages of small and large cities over total ( list of floats  0<x<1)

        Returns:
            * list of integers representing cluster sizes

        """
        if min_size * self.num_cities >= self.num_people:
            raise ValueError('num_people should be greater than min_size * num_cities ')
        size_choices = [max(int(self.num_people / (10 * self.num_cities)), min_size),
                        max(int(self.num_people / self.num_cities), min_size)]
        city_sizes = np.random.choice(size_choices, p=small_large_pcts, size=self.num_cities - 1)
        last_city_size = self.num_people - city_sizes.sum()
        city_sizes = np.append(city_sizes, last_city_size)
        pcts = np.random.dirichlet(city_sizes)
        return np.random.multinomial(city_sizes.sum(), pcts)

    def generate_cluster_points_coords(self, pct_grid_w, pct_grid_h, clust_size):
        """ Using binomial ditribution, this method generates initial coordinates
            for a given cluster, defined via its center and its size.
            Cluster size as well as cluster center coords
            (in grid percentage) must be provided

        Arguments:
            * pct_grid_w: positive float < 1 to define clust_center along grid width
            * pct_grid_h: positive float < 1 to define clust_center along grid height
            * clust_size: desired size of the cluster being generated

        Returns:
            * cluster_coordinates: two numpy arrays with x and y coordinates
            respectively

        """
        ## use binomial generator to get clusters in width * height grid
        ## n = grid_width, p = pct_grid, size = num_experim
        x_coords = np.random.binomial(self.grid_width,
                                      pct_grid_w,
                                      size=clust_size)
        for idx, elem in enumerate(x_coords):
            if elem >= self.grid_width:
                x_coords[idx] = self.grid_width - 1

        y_coords = np.random.binomial(self.grid_height,
                                      pct_grid_h,
                                      size=clust_size)
        for idx, elem in enumerate(y_coords):
            if elem >= self.grid_width:
                y_coords[idx] = self.grid_height - 1
        return x_coords, y_coords

    def create_lang_agents(self):
        """ Method to instantiate all agents

        Arguments:
            * sort_lang_types_by_dist: boolean to specify
                if agents must be sorted by distance to global origin
            * sort_sub_types_within_clust: boolean to specify
                if agents must be sorted by distance to center of cluster they belong to

            """

        self.cluster_sizes = self.compute_cluster_sizes()
        array_langs = np.random.choice([0, 1, 2], p=self.init_lang_distrib, size=self.num_people)
        if self.lang_ags_sorted_by_dist:
            array_langs.sort()
        idxs_to_split = self.cluster_sizes.cumsum()
        langs_per_clust = np.split(array_langs, idxs_to_split)
        if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
            for subarray in langs_per_clust:
                subarray.sort()  # invert if needed
        ids = set(range(self.num_people))

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes, self.clust_centers)):
            x_cs, y_cs = self.generate_cluster_points_coords(clust_c_coords[0], clust_c_coords[1], clust_size)
            if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
                clust_p_coords = sorted(list(zip(x_cs, y_cs)),
                                        key=lambda x:pdist([x, [self.grid_width*self.clust_centers[clust_idx][0],
                                                                self.grid_height*self.clust_centers[clust_idx][1]]
                                                            ])
                                        )
                x_cs, y_cs = list(zip(*clust_p_coords))
            for ag_lang, x, y in zip(langs_per_clust[clust_idx], x_cs, y_cs):
                ag = Simple_Language_Agent(self, ids.pop(), ag_lang, 0.5)
                self.add_agent(ag, (x, y))


    def get_lang_stats(self, i):
        """Method to get counts of each type of lang agent

        Arguments:
            * i : integer from [0,1,2] hat specifies agent lang type

        Returns:
            * lang type count as percentage of total

        """
        ag_lang_list = [ag.language for ag in self.schedule.agents]
        num_ag = len(ag_lang_list)
        lang_counts = Counter(ag_lang_list)
        return lang_counts[i]/num_ag

    def get_bilingual_global_evol(self, lang_typology):
        """Method to compute internal linguistic structure of all bilinguals,
        expressed as average amount of Catalan heard or spoken as % of total

         Arguments:
             * lang_typology: string that can take either of two values 'heard' or 'spoken'

         Returns:
             * float representing the AVERAGE percentage of Catalan in bilinguals

        """
        list_biling = [(ag.lang_freq['cat_pct_h'], ag.lang_freq['cat_pct_s'])
                       for ag in self.schedule.agents if ag.language == 1]
        if lang_typology == 'heard':
            if list_biling:
                return np.array(list(zip(*list_biling))[0]).mean()
            else:
                if self.get_lang_stats(2) > self.get_lang_stats(0):
                    return 1
                else:
                    return 0
        else:
            if list_biling:
                return np.array(list(zip(*list_biling))[1]).mean()
            else:
                if self.get_lang_stats(2) > self.get_lang_stats(0):
                    return 1
                else:
                    return 0

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def run_model(self, steps, save_frames_freq=0, save_dir=''):
        pbar = pyprind.ProgBar(steps)
        if save_frames_freq:
            self.save_dir = save_dir
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, save_dir)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
        for _ in range(steps):
            self.step()
            if save_frames_freq:
                if not self.schedule.steps%save_frames_freq:
                    self.show_results(step=self.schedule.steps, plot_results=False, save_fig=True)
            pbar.update()

    def create_agents_attrs_data(self, ag_attr, plot=False):
        ag_and_coords = [(getattr(ag, ag_attr), ag.pos[0], ag.pos[1])
                         for ag in self.schedule.agents]
        ag_and_coords = np.array(ag_and_coords)
        df_attrs = pd.DataFrame({'values': ag_and_coords[:, 0],
                                 'x': ag_and_coords[:, 1],
                                 'y': ag_and_coords[:, 2]})
        self.df_attrs_avg = df_attrs.groupby(['x', 'y']).mean()

        if plot:
            s = plt.scatter(self.df_attrs_avg.reset_index()['x'],
                            self.df_attrs_avg.reset_index()['y'],
                            c=self.df_attrs_avg.reset_index()['values'],
                            vmin=0, vmax=2, s=30,
                            cmap='viridis')
            plt.colorbar(s)
            plt.show()


    def show_results(self, ag_attr='language', step=None,
                     plot_results=True, plot_type='imshow', save_fig=False):

        grid_size = (3, 5)
        self.create_agents_attrs_data(ag_attr)

        data_2_plot = self.datacollector.get_model_vars_dataframe()[:step]
        data_2D = self.df_attrs_avg.reset_index()

        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        data_2_plot[["count_bil", "count_cat", "count_spa"]].plot(ax=ax1, title='lang_groups')
        ax1.xaxis.tick_bottom()
        ax1.legend(loc='best', prop={'size': 8})
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        data_2_plot['total_num_agents'].plot(ax=ax2, title='num_agents')
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        data_2_plot[['biling_evol_h', 'biling_evol_s']].plot(ax=ax3, title='biling_quality')
        ax3.legend(loc='best', prop={'size': 8})
        ax4 = plt.subplot2grid(grid_size, (0, 0), rowspan=3, colspan=3)
        if plot_type == 'imshow':
            s = ax4.imshow(self.df_attrs_avg.unstack('x'), vmin=0, vmax=2, cmap='viridis',
                           interpolation='nearest', origin='lower')
        else:
            s = ax4.scatter(data_2D['x'],
                            data_2D['y'],
                            c=data_2D['values'],
                            vmin=0, vmax=2, s=35,
                            cmap='viridis')
        ax4.text(0.02, 0.95, 'time = %.1f' % self.schedule.steps, transform=ax4.transAxes)
        plt.colorbar(s)
        plt.suptitle(self.save_dir)
        plt.tight_layout()
        if save_fig:
            if self.save_dir:
                plt.savefig(self.save_dir + '/step' + str(step) + '.png')
                plt.close()
            else:
                plt.savefig('step' + str(step) + '.png')

        if plot_results:
            plt.show()

    def run_and_animate(self, steps, plot_type='imshow'):
        fig = plt.figure()
        grid_size = (3, 5)
        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        ax1.set_xlim(0, steps)
        ax1.set_ylim(0, 1)
        ax1.xaxis.tick_bottom()
        line10, = ax1.plot([], [], lw=2, label='count_spa')
        line11, = ax1.plot([], [], lw=2, label='count_bil')
        line12, = ax1.plot([], [], lw=2, label='count_cat')
        ax1.legend(loc='best', prop={'size': 8})
        ax1.set_title("lang_groups")
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        ax2.set_xlim(0, steps)
        ax2.set_ylim(0, self.max_people_factor * self.num_people)
        line2, = ax2.plot([], [], lw=2, label = "total_num_agents")
        ax2.legend(loc='best', prop={'size': 8})
        ax2.set_title("num_agents")
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        ax3.set_xlim(0, steps)
        ax3.set_ylim(0, 1)
        line30, = ax3.plot([], [], lw=2, label='biling_evol_h')
        line31, = ax3.plot([], [], lw=2, label='biling_evol_s')
        ax3.legend(loc='best', prop={'size': 8})
        ax3.set_title("biling_quality")
        ax4 = plt.subplot2grid(grid_size, (0, 0), rowspan=3, colspan=3)
        ax4.set_xlim(0, self.grid_width-1)
        ax4.set_ylim(0, self.grid_height-1)
        if plot_type == 'imshow':
            im_2D = ax4.imshow(np.zeros((self.grid_width, self.grid_height)),
                               vmin=0, vmax=2, cmap='viridis',
                               interpolation='nearest', origin='lower')
            fig.colorbar(im_2D)
        elif plot_type == 'scatter':
            dots = ax4.scatter([], [], c=[], vmin=0, vmax=2, cmap='viridis')
            fig.colorbar(dots)
        time_text = ax4.text(0.02, 0.95, '', transform=ax4.transAxes)

        def init_show():
            if plot_type == 'imshow':
                im_2D.set_array(np.random.choice([np.nan, 0], p=[1, 0], size=(self.grid_width, self.grid_height)))
                return im_2D,
            elif plot_type == 'scatter':
                dots.set_offsets([0,0])
                return dots,

        def run_and_update(i):
            #run model step
            self.step()

            #create plots and data for 1D plots
            data = self.datacollector.get_model_vars_dataframe()
            line10.set_data(data.index, data['count_spa'])
            line11.set_data(data.index, data['count_bil'])
            line12.set_data(data.index, data['count_cat'])

            line2.set_data(data.index, data['total_num_agents'])

            line30.set_data(data.index, data['biling_evol_h'])
            line31.set_data(data.index, data['biling_evol_s'])
            # generate data for 2D representation
            self.create_agents_attrs_data('language')
            # create 2D plot
            time_text.set_text('time = %.1f' % i)
            if plot_type == 'imshow':
                im_2D.set_array(self.df_attrs_avg.unstack('x'))
                return line10, line11, line12, line2, line30, line31, im_2D, time_text
            else:
                data = np.hstack((self.df_attrs_avg.reset_index()['x'][:, np.newaxis],
                                  self.df_attrs_avg.reset_index()['y'][:, np.newaxis]))
                dots.set_offsets(data)
                dots.set_array(self.df_attrs_avg.reset_index()['values'])
                return line10, line11, line12, line2, line30, line31, dots, time_text

        # generate persistent animation object
        ani = animation.FuncAnimation(fig, run_and_update,init_func=init_show,
                                      frames=steps, interval=100, blit=True, repeat=False)
        plt.tight_layout()
        plt.show()

    def save_model_data(self):
        self.model_data = {'initial_conditions':{'cluster_sizes': self.cluster_sizes,
                                                 'cluster_centers': self.clust_centers,
                                                 'init_num_people': self.num_people,
                                                 'grid_width': self.grid_width,
                                                 'grid_height': self.grid_height,
                                                 'init_lang_distrib': self.init_lang_distrib,
                                                 'num_cities': self.num_cities,
                                                 'sort_by_dist': self.lang_ags_sorted_by_dist,
                                                 'sort_within_clust': self.lang_ags_sorted_in_clust},
                           'model_results': self.datacollector.get_model_vars_dataframe(),
                           'agent_results': self.datacollector.get_agent_vars_dataframe()}
        if self.save_dir:
            dd.io.save(self.save_dir + '/model_data.h5', self.model_data)
        else:
            dd.io.save('model_data.h5', self.model_data)

    def load_model_data(self, data_filename, key='/' ):
        return dd.io.load(data_filename,key)







        