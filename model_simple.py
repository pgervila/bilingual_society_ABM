# IMPORT RELEVANT LIBRARIES
import os
from importlib import reload
from math import ceil
import random
from itertools import product
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use("TKAgg")
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
        for agent in self.agents[:]:
            agent.age += 1
            agent.reproduce()
            for lang in ['L1', 'L2']:
                # update last-time word use vector
                agent.lang_stats[lang]['t'][~agent.day_mask[lang]] += 1
                # set current lang knowledge
                agent.lang_stats[lang]['pct'][agent.age] = (np.where(agent.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                            agent.model.vocab_red)
                agent.day_mask[lang] = np.zeros(agent.model.vocab_red, dtype=np.bool)
            # Update lang switch
            agent.update_lang_switch()
        if self.shuffle:
            random.shuffle(self.agents)
        for stage in self.stage_list:
            for agent in self.agents[:]:
                getattr(agent, stage)()  # Run stage
            if self.shuffle_between_stages:
                random.shuffle(self.agents)
            self.time += self.stage_time
        # simulate death chance
        for agent in self.agents[:]:
            agent.simulate_random_death()
        self.steps += 1


class Simple_Language_Model(Model):
    def __init__(self, num_people, spoken_only=True, num_words_conv=(3, 25), width=100, height=100, max_people_factor=5,
                 init_lang_distrib=[0.25, 0.65, 0.1], num_cities=10, max_run_steps=1000, lang_ags_sorted_by_dist=True,
                 lang_ags_sorted_in_clust=True):
        self.num_people = num_people
        if spoken_only:
            self.vocab_red = 500
        else:
            self.vocab_red = 1000
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
        self.lang_ICs = dd.io.load('IC_lang_ONLY_SPOKEN_1000_steps.h5')
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
        self.generate_jobs()
        self.generate_schools()
        self.map_lang_agents()


        # DEFINE FAMILY NETWORKS
        #self.define_family_networks()


        # DATA COLLECTOR
        self.datacollector = DataCollector(
            model_reporters={"count_spa": lambda m: m.get_lang_stats(0),
                             "count_bil": lambda m: m.get_lang_stats(1),
                             "count_cat": lambda m: m.get_lang_stats(2),
                             "total_num_agents": lambda m:len(m.schedule.agents),
                             "biling_evol": lambda m:m.get_bilingual_global_evol()
                             },
            agent_reporters={"pct_cat_in_biling": lambda a:a.lang_stats['L2']['pct'][a.age],
                             "pct_spa_in_biling": lambda a:a.lang_stats['L1']['pct'][a.age]}
        )

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
        # If requested, sort centers based on their distance to grid origin
        if self.lang_ags_sorted_by_dist:
            self.clust_centers = sorted(self.clust_centers, key=lambda x:pdist([x,[0,0]]))


    def compute_cluster_sizes(self, min_size=20):
        """ Method to compute sizes of each agent cluster.
        Cluster size equals number of language agents that live in this cluster

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


    def generate_cluster_points_coords(self, pcts_grid, clust_size):
        """ Using binomial distribution, this method generates initial coordinates
            for a given cluster, defined via its center and its size.
            Cluster size as well as cluster center coords (in grid percentage) must be provided

        Arguments:
            * pcts_grid: 2-D tuple with positive floats < 1 to define clust_center along grid width and height
            * clust_size: desired size of the cluster being generated

        Returns:
            * cluster_coordinates: two numpy arrays with x and y coordinates
            respectively

        """
        x_coords = np.random.binomial(self.grid_width,  pcts_grid[0], size=clust_size)
        # limit coords values to grid boundaries
        x_coords = np.clip(x_coords, 1, self.grid_width - 1)

        y_coords = np.random.binomial(self.grid_height, pcts_grid[1], size=clust_size)
        # limit coords values to grid boundaries
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

    def generate_jobs(self, num_job_c_per_agent=0.1, pct_agents_with_job=5, min_places=2, max_places=200):
        """ Generates job centers coordinates and num places per center
        Args:
            * job_c_per_agent: float. Number of jobs centers expressed as percentage of number of agents
            * min_places: integer. Minimum number of places for each job center
            * max_places: integer. Maximum number of places for each job center

        """
        # iterate to generate job center coordinates
        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_j, y_j = self.generate_cluster_points_coords(clust_c_coords,
                                                           int(clust_size * num_job_c_per_agent))
            if (not self.lang_ags_sorted_by_dist) and self.lang_ags_sorted_in_clust:
                c_coords = self.grid_width * self.clust_centers[clust_idx]
                job_p_coords = sorted(list(zip(x_j, y_j)), key=lambda p:pdist([p, c_coords]))
                x_j, y_j = list(zip(*job_p_coords))

            # compute percentage number places per each job center in cluster using lognormal distrib
            p = np.random.lognormal(1, 1, size=int(clust_size * num_job_c_per_agent))
            pcts = p / p.sum()
            # compute num of places out of percentages
            num_places_job_c = np.random.multinomial(int(clust_size * self.max_people_factor), pcts)
            num_places_job_c = np.clip(num_places_job_c, min_places, max_places)

            for x, y, num_places in zip(x_j, y_j, num_places_job_c):
                self.clusters_info[clust_idx]['jobs'].append(Job((x,y), num_places))

    def generate_schools(self, schools_per_agent=0.05, school_size=100):
        """ Generate coordinates for school centers and instantiate school objects
            Args:
                * Schools_per_agent: number of schools in cluster as percentage of num agents in cluster
                * school_size: fixed size of all schools generated
        """
        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_s, y_s = self.generate_cluster_points_coords(clust_c_coords,
                                                           int(clust_size * schools_per_agent))
            if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
                c_coords = self.grid_width * self.clust_centers[clust_idx]
                school_p_coords = sorted(list(zip(x_s, y_s)), key=lambda p:pdist([p, c_coords]))
                x_s, y_s = list(zip(*school_p_coords))

            for x,y in zip(x_s,y_s):
                self.clusters_info[clust_idx]['schools'].append(School((x,y), school_size))

    def map_lang_agents(self):
        """ Method to instantiate all agents

        Arguments:
            * sort_lang_types_by_dist: boolean to specify
                if agents must be sorted by distance to global origin
            * sort_sub_types_within_clust: boolean to specify
                if agents must be sorted by distance to center of cluster they belong to

            """
        # generate random array with lang labels ( no sorting at all )
        langs_per_ag_array = np.random.choice([0, 1, 2], p=self.init_lang_distrib, size=self.num_people)
        idxs_to_split_by_clust = self.cluster_sizes.cumsum()[:-1]
        # check if agent sorting by distance to origin is requested
        if self.lang_ags_sorted_by_dist:
            langs_per_ag_array.sort()
        # split lang labels by cluster
        langs_per_clust = np.split(langs_per_ag_array, idxs_to_split_by_clust)
        if (not self.lang_ags_sorted_by_dist) and self.lang_ags_sorted_in_clust:
            for subarray in langs_per_clust:
                subarray.sort()  # invert if needed
        ids = set(range(self.num_people))
        #start mapping
        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            # generate agent coordinates (cluster points)
            x_ags, y_ags = self.generate_cluster_points_coords(clust_c_coords, clust_size)
            # Sort clust points by dist to clust center if requested
            if (not self.lang_ags_sorted_by_dist) and self.lang_ags_sorted_in_clust:
                c_coords = self.grid_width * self.clust_centers[clust_idx]
                clust_p_coords = sorted(list(zip(x_ags, y_ags)), key=lambda p:pdist([p, c_coords]))
                x_ags, y_ags = list(zip(*clust_p_coords))
            # get list of tuples with school coords
            clust_schools_coords = [sc.pos for sc in self.clusters_info[clust_idx]['schools']]
            #
            for ag_lang, x, y in zip(langs_per_clust[clust_idx], x_ags, y_ags):
                # get closest school given agent coordinates
                closest_school = np.argmin([pdist([(x,y), sc_coord])
                                            for sc_coord in clust_schools_coords])
                xs, ys = self.clusters_info[clust_idx]['schools'][closest_school].pos
                # get random job
                job = random.choice(self.clusters_info[clust_idx]['jobs'])
                # instantiate agent
                ag = Simple_Language_Agent(self, ids.pop(), ag_lang, age=1000, home_coords=(x, y),
                                           school_coords=(xs, ys), job_coords=job.pos, city_idx=clust_idx,
                                           import_IC=True)
                self.clusters_info[clust_idx]['agents_id'].append(ag.unique_id)
                # add agent
                self.add_agent(ag)

    def add_agent(self, a):
        """Method to add a given agent to grid, schedule and system networks

        Arguments:
            * a : agent class instance
            * coords : agent location on grid (2-D tuple of integers)
        """
        # add agent to grid and schedule
        self.schedule.add(a)
        self.grid.place_agent(a, a.home_coords)
        ## add agent node to all networks
        self.known_people_network.add_node(a)
        self.friendship_network.add_node(a)
        self.family_network.add_node(a)

    def define_family_networks(self):
        # define families
        # marriage, to make things simple, only allowed for combinations  0-1, 1-1, 1-2
        family_size = 4
        for family in zip(*[iter(sorted(self.schedule.agents, key=lambda ag:ag.language))] *
                          family_size):
            graph_fam = self.family_network
            family[0].age, family[1].age = np.random.randint(40, 60, size=2)
            family[2].age, family[3].age = np.random.randint(10, 20, size=2)

            # find out lang btw family members
            # consorts
            if (family[0].language, family[1].language) in [(0, 0), (0, 1), (1, 0)]:
                lang_consorts = 0
            elif (family[0].language, family[1].language) in [(2, 1), (1, 2), (2, 2)]:
                lang_consorts = 1
            elif (family[0].language, family[1].language) == (1, 1):
                p1 = family[0].lang_freq['cat_pct_s']
                p2 = family[1].lang_freq['cat_pct_s']
                if all(x >= 0.5 for x in (p1, p2)):
                    if p1 > 0.5 or p2 > 0.5:
                        lang_consorts = 1
                    else:
                        lang_consorts = random.choice([0, 1])
                elif all(x <= 0.5 for x in (p1, p2)):
                    lang_consorts = 0
                else:
                    delta_abs = np.abs([p1 - 0.5, p2 - 0.5])
                    most_distant_agent = np.argmax(delta_abs)
                    if family[most_distant_agent].lang_freq['cat_pct_s'] > 0.5:
                        lang_consorts = 1
                    else:
                        lang_consorts = 0

            # parent, mother
            lang_with_father = np.random.binomial(1,family[0].lang_freq['cat_pct_s'])
            lang_with_mother = np.random.binomial(1,family[1].lang_freq['cat_pct_s'])

            #siblings
            avg_lang = (lang_with_father + lang_with_mother) / 2
            if avg_lang == 0:
                lang_siblings = 0
            elif avg_lang == 1:
                lang_siblings = 1
            else:
                avg = (family[0].lang_freq['cat_pct_s'] + family[1].lang_freq['cat_pct_s']) / 2
                lang_siblings = np.random.binomial(1, avg)

            graph_fam.add_edge(family[0], family[1], fam_link='consort', lang=lang_consorts)
            graph_fam.add_edge(family[1], family[0], fam_link='consort', lang=lang_consorts)
            graph_fam.add_edge(family[0], family[2], fam_link='child', lang=lang_with_father)
            graph_fam.add_edge(family[2], family[0], fam_link='father', lang=lang_with_father)
            graph_fam.add_edge(family[0], family[3], fam_link='child', lang=lang_with_father)
            graph_fam.add_edge(family[3], family[0], fam_link='father', lang=lang_with_father)
            graph_fam.add_edge(family[1], family[2], fam_link='child', lang=lang_with_mother)
            graph_fam.add_edge(family[2], family[1], fam_link='mother', lang=lang_with_mother)
            graph_fam.add_edge(family[1], family[3], fam_link='child', lang=lang_with_mother)
            graph_fam.add_edge(family[3], family[1], fam_link='mother', lang=lang_with_mother)
            graph_fam.add_edge(family[2], family[3], fam_link='sibling', lang=lang_siblings)
            graph_fam.add_edge(family[3], family[2], fam_link='sibling', lang=lang_siblings)

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

    def get_bilingual_global_evol(self):
        """Method to compute internal linguistic structure of all bilinguals,
        expressed as average amount of Catalan heard or spoken as % of total

         Returns:
             * float representing the AVERAGE percentage of Catalan in bilinguals

        """
        list_biling = [ag.lang_stats['L2']['pct'][ag.age] for ag in self.schedule.agents
                       if ag.language == 1]
        if list_biling:
            return np.array(list_biling).mean()
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
                            marker='s',
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
        data_2_plot['biling_evol'].plot(ax=ax3, title='biling_quality')
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
                            marker='s',
                            vmin=0, vmax=2, s=35,
                            cmap='viridis')
        ax4.text(0.02, 1.04, 'time = %.1f' % self.schedule.steps, transform=ax4.transAxes)
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        plt.colorbar(s)
        plt.suptitle(self.save_dir)
        #plt.tight_layout()
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
        line3, = ax3.plot([], [], lw=2, label='biling_evol')
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

            line3.set_data(data.index, data['biling_evol'])
            # generate data for 2D representation
            self.create_agents_attrs_data('language')
            # create 2D plot
            time_text.set_text('time = %.1f' % i)
            if plot_type == 'imshow':
                im_2D.set_array(self.df_attrs_avg.unstack('x'))
                return line10, line11, line12, line2, line3, im_2D, time_text
            else:
                data = np.hstack((self.df_attrs_avg.reset_index()['x'][:, np.newaxis],
                                  self.df_attrs_avg.reset_index()['y'][:, np.newaxis]))
                dots.set_offsets(data)
                dots.set_array(self.df_attrs_avg.reset_index()['values'])
                return line10, line11, line12, line2, line3, dots, time_text

        # generate persistent animation object
        ani = animation.FuncAnimation(fig, run_and_update,init_func=init_show,
                                      frames=steps, interval=100, blit=True, repeat=False)
        #plt.tight_layout()
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

    def plot_family_networks(self):
        """PLOT NETWORK with lang colors and position"""
        #        people_pos = [elem.pos for elem in self.schedule.agents if elem.agent_type == 'language']
        #        # Following works because lang agents are added before other agent types
        #        people_pos_dict= dict(zip(self.schedule.agents, people_pos))
        people_pos_dict = {elem: elem.pos
                           for elem in self.schedule.agents}
        people_color = [elem.language for elem in self.family_network]
        nx.draw(self.family_network, pos=people_pos_dict, node_color=people_color)







        