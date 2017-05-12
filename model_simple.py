# IMPORT RELEVANT LIBRARIES
import os
from importlib import reload
from math import ceil
import random
from itertools import product
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

#import private library to model lang zipf CDF
from zipf_generator import Zipf_Mand_CDF_compressed, randZipf


# import progress bar
import pyprind


# IMPORT FROM simp_agent.py
from agent_simple import Simple_Language_Agent, Home, School, Job

# IMPORT MESA LIBRARIES
from mesa import Model
from mesa.time import RandomActivation, StagedActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class StagedActivation_modif(StagedActivation):

    def step(self):
        """ Executes all the stages for all agents. """
        if self.shuffle:
            random.shuffle(self.agents)
        for stage in self.stage_list:
            for agent in self.agents[:]:
                getattr(agent, stage)()  # Run stage
            if self.shuffle_between_stages:
                random.shuffle(self.agents)
            self.time += self.stage_time
        self.steps += 1


class Simple_Language_Model(Model):
    def __init__(self, num_people, vocab_red=1000, num_words_conv=(3, 25), width=5, height=5, max_people_factor=5,
                 init_lang_distrib=[0.25, 0.65, 0.1], num_cities=10, max_run_steps=1000,
                 lang_ags_sorted_by_dist=True, lang_ags_sorted_in_clust=True):
        self.num_people = num_people
        self.vocab_red = vocab_red
        self.num_words_conv = num_words_conv
        self.grid_width = width
        self.grid_height = height
        self.max_people_factor = max_people_factor
        self.init_lang_distrib = init_lang_distrib
        self.num_cities = num_cities
        self.max_run_steps = max_run_steps
        self.lang_ags_sorted_by_dist = lang_ags_sorted_by_dist
        self.lang_ags_sorted_in_clust = lang_ags_sorted_in_clust
        self.clust_centers = None
        self.cluster_sizes = None

        # import lang ICs and lang CDFs data as step function
        self.lang_ICs = dd.io.load('IC_lang_1500_steps.h5')
        self.cdf_data = dd.io.load('cdfs_3R_vs_step.h5')

        # define grid and schedule
        self.grid = MultiGrid(height, width, False)
        self.schedule = StagedActivation_modif(self,
                                               stage_list=["stage_1", "stage_2",
                                                           "stage_3", "stage_4"],
                                               shuffle=True,
                                               shuffle_between_stages=True)

        ## define clusters and add jobs, schools, agents
        self.compute_cluster_centers()
        self.compute_cluster_sizes()
        self.set_clusters_info()


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
        self.generate_jobs()
        self.generate_schools()
        self.generate_agents()

        # DATA COLLECTOR
        self.datacollector = DataCollector(
            model_reporters={"count_spa": lambda m: m.get_lang_stats(0),
                             "count_bil": lambda m: m.get_lang_stats(1),
                             "count_cat": lambda m: m.get_lang_stats(2),
                             "total_num_agents": lambda m:len(m.schedule.agents),
                             "biling_evol_h": lambda m:m.get_bilingual_global_evol('heard'),
                             "biling_evol_s": lambda m: m.get_bilingual_global_evol('spoken')
                             },
            agent_reporters={"pct_cat_in_biling": lambda a:a.lang_stats['l']['LT']['L2_pct'],
                             "pct_spa_in_biling": lambda a: 1 - a.lang_stats['l']['LT']['L2_pct']}
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


    def compute_cluster_centers(self, min_dist=0.20):
        """ Generate 2D coordinates for all cluster centers (in percentage of grid dimensions) """

        # Define available points as pct of squared grid length
        grid_pct_list = np.linspace(0.1, 0.9, 100) # avoid edges
        # Assure min distance btwn clusters
        s1 = set(product(grid_pct_list, grid_pct_list))
        self.clust_centers = []
        p = random.sample(s1, 1)[0]
        self.clust_centers.append(p)
        s1.remove(p)
        for _ in range(self.num_cities - 1):
            for point in set(s1):
                if pdist([point, p]) < min_dist * (grid_pct_list.max() - grid_pct_list.min()):
                    s1.remove(point)
            try:
                p = random.sample(s1, 1)[0]
            except ValueError:
                print('INPUT ERROR: Reduce either number of cities or minimum distance '
                      'in order to meet distance constraint')
                raise
            self.clust_centers.append(p)
            s1.remove(p)
        self.clust_centers = np.array(self.clust_centers)
        if self.lang_ags_sorted_by_dist:
            self.clust_centers = sorted(self.clust_centers, key=lambda x:pdist([x,[0,0]]))


    def compute_cluster_sizes(self, min_size=20):
        """ Method to compute sizes of each agent cluster.

        Arguments:
            * min_size: minimum accepted cluster size ( integer)

        Returns:
            * list of integers representing cluster sizes

        """
        p = np.random.pareto(1.25, size=self.num_cities)
        pcts = p / p.sum()
        self.cluster_sizes = np.random.multinomial(self.num_people, pcts)
        tot_sub_val = 0
        for idx in np.where(self.cluster_sizes < min_size)[0]:
            tot_sub_val += min_size - self.cluster_sizes[idx]
            self.cluster_sizes[idx] = min_size
        self.cluster_sizes[np.argmax(self.cluster_sizes)] -= tot_sub_val


    def generate_cluster_points_coords(self, pct_grid_w, pct_grid_h, clust_size):
        """ Using binomial distribution, this method generates initial coordinates
            for a given cluster, defined via its center and its size.
            Cluster size as well as cluster center coords (in grid percentage) must be provided

        Arguments:
            * pct_grid_w: positive float < 1 to define clust_center along grid width
            * pct_grid_h: positive float < 1 to define clust_center along grid height
            * clust_size: desired size of the cluster being generated

        Returns:
            * cluster_coordinates: two numpy arrays with x and y coordinates
            respectively

        """
        x_coords = np.random.binomial(self.grid_width,  pct_grid_w, size=clust_size)
        x_coords = np.clip(x_coords, 1, self.grid_width - 1)

        y_coords = np.random.binomial(self.grid_height, pct_grid_h, size=clust_size)
        y_coords = np.clip(y_coords, 1, self.grid_height - 1)

        return x_coords, y_coords


    def set_clusters_info(self):
        """ Create dict container for all cluster info"""

        self.clusters_info = defaultdict(dict)
        for idx, clust_size in enumerate(self.cluster_sizes):
            self.clusters_info[idx]['num_agents'] = clust_size
            self.clusters_info[idx]['jobs'] = []
            self.clusters_info[idx]['schools'] = []
            self.clusters_info[idx]['agents_id'] = []


    def generate_jobs(self, job_cent_per_agent=0.1, pct_agents_with_job=5,
                      min_places=2, max_places=200):
        """ Generates job centers coordinates and num places per center"""
        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_j, y_j = self.generate_cluster_points_coords(clust_c_coords[0],
                                                           clust_c_coords[1],
                                                           int(clust_size * job_cent_per_agent))

            if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
                c_coords = self.grid_width * self.clust_centers[clust_idx]
                job_p_coords = sorted(list(zip(x_j, y_j)), key=lambda p:pdist([p, c_coords]))
                x_j, y_j = list(zip(*job_p_coords))

            # compute places per each job in cluster using lognormal distrib
            p = np.random.lognormal(1, 1, size=int(clust_size * job_cent_per_agent))
            pcts = p / p.sum()
            job_cent_places = np.random.multinomial(int(clust_size * pct_agents_with_job), pcts)
            job_cent_places = np.clip(job_cent_places, min_places, max_places)

            for x, y, num_places in zip(x_j, y_j, job_cent_places):
                self.clusters_info[clust_idx]['jobs'].append(Job((x,y), num_places))

    def generate_schools(self, schools_per_agent=0.05, school_size=100):
        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_s, y_s = self.generate_cluster_points_coords(clust_c_coords[0],
                                                           clust_c_coords[1],
                                                           int(clust_size * schools_per_agent))
            if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
                c_coords = self.grid_width * self.clust_centers[clust_idx]
                school_p_coords = sorted(list(zip(x_s, y_s)), key=lambda p:pdist([p, c_coords]))
                x_s, y_s = list(zip(*school_p_coords))

            for x,y in zip(x_s,y_s):
                self.clusters_info[clust_idx]['schools'].append(School((x,y), school_size))

    def generate_agents(self):
        """ Method to instantiate all agents

        Arguments:
            * sort_lang_types_by_dist: boolean to specify
                if agents must be sorted by distance to global origin
            * sort_sub_types_within_clust: boolean to specify
                if agents must be sorted by distance to center of cluster they belong to

            """

        langs_per_ag_array = np.random.choice([0, 1, 2], p=self.init_lang_distrib, size=self.num_people)
        idxs_to_split_by_clust = self.cluster_sizes.cumsum()[:-1]
        langs_per_clust = np.split(langs_per_ag_array, idxs_to_split_by_clust)
        if self.lang_ags_sorted_by_dist:
            langs_per_ag_array.sort()
        if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
            for subarray in langs_per_clust:
                subarray.sort()  # invert if needed
        ids = set(range(self.num_people))

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_ags, y_ags = self.generate_cluster_points_coords(clust_c_coords[0],
                                                               clust_c_coords[1],
                                                               clust_size)
            if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
                c_coords = self.grid_width * self.clust_centers[clust_idx]
                clust_p_coords = sorted(list(zip(x_ags, y_ags)), key=lambda p:pdist([p, c_coords]))
                x_ags, y_ags = list(zip(*clust_p_coords))
            clust_schools_coords = [sc.pos for sc in self.clusters_info[clust_idx]['schools']]
            for ag_lang, x, y in zip(langs_per_clust[clust_idx], x_ags, y_ags):
                closest_school = np.argmin([pdist([(x,y), sc_coord])
                                            for sc_coord in clust_schools_coords])
                xs, ys = self.clusters_info[clust_idx]['schools'][closest_school].pos
                job = random.choice(self.clusters_info[clust_idx]['jobs'])
                ag = Simple_Language_Agent(self, ids.pop(), ag_lang, home_coords=(x, y), school_coords=(xs, ys),
                                           home_coords=(x, y), school_coords=(xs, ys), job_coords=job.pos)
                self.clusters_info[clust_idx]['agents_id'].append(ag.unique_id)
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
        list_biling = [(ag.lang_stats['l']['LT']['L2_pct'], ag.lang_stats['s']['LT']['L2_pct'])
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

    def run_model(self, steps, recording_steps_period=None, save_dir=''):
        """ Run model and save frames if required
            Args
                * steps: integer. Total steps to run
                * recording_steps_period : integer. Save frames every specified number of steps
                * save_dir : string. It specifies directory where frames will be saved
        """
        pbar = pyprind.ProgBar(steps)
        self.save_dir = save_dir
        if recording_steps_period:
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, save_dir)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
        for _ in range(steps):
            self.step()
            if recording_steps_period:
                if not self.schedule.steps%recording_steps_period:
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
        data_2_plot[["count_bil", "count_cat", "count_spa"]].plot(ax=ax1,
                                                                  title='lang_groups',
                                                                  color=['darkgreen','y','darkblue'])
        ax1.xaxis.tick_bottom()
        ax1.tick_params('x', labelsize='small')
        ax1.tick_params('y', labelsize='small')
        ax1.legend(loc='best', prop={'size': 8})
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        data_2_plot['total_num_agents'].plot(ax=ax2, title='num_agents')
        ax2.tick_params('x', labelsize='small')
        ax2.tick_params('y', labelsize='small')
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        data_2_plot[['biling_evol_h', 'biling_evol_s']].plot(ax=ax3, title='biling_quality')
        ax3.tick_params('x', labelsize='small')
        ax3.tick_params('y', labelsize='small')
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
        ax4.text(0.02, 1.04, 'time = %.1f' % self.schedule.steps, transform=ax4.transAxes)
        plt.colorbar(s)
        plt.suptitle(self.save_dir)
        plt.tight_layout()
        if save_fig:
            if self.save_dir:
                plt.savefig(self.save_dir + '/step' + str(step) + '.png')
                plt.close()
            else:
                plt.savefig('step' + str(step) + '.png')
                plt.close()

        if plot_results:
            plt.show()

    def run_and_animate(self, steps, plot_type='imshow'):
        fig = plt.figure()
        grid_size = (3, 5)
        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        ax1.set_xlim(0, steps)
        ax1.set_ylim(0, 1)
        ax1.xaxis.tick_bottom()
        line10, = ax1.plot([], [], lw=2, label='count_spa', color='darkblue')
        line11, = ax1.plot([], [], lw=2, label='count_bil', color='g')
        line12, = ax1.plot([], [], lw=2, label='count_cat', color='y')
        ax1.tick_params('x', labelsize='small')
        ax1.tick_params('y', labelsize='small')
        ax1.legend(loc='best', prop={'size': 8})
        ax1.set_title("lang_groups")
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        ax2.set_xlim(0, steps)
        ax2.set_ylim(0, self.max_people_factor * self.num_people)
        line2, = ax2.plot([], [], lw=2, label = "total_num_agents", color='k')
        ax2.tick_params('x', labelsize='small')
        ax2.tick_params('y', labelsize='small')
        ax2.legend(loc='best', prop={'size': 8})
        ax2.set_title("num_agents")
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        ax3.set_xlim(0, steps)
        ax3.set_ylim(0, 1)
        ax3.tick_params('x', labelsize='small')
        ax3.tick_params('y', labelsize='small')
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







        