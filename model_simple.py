# IMPORT RELEVANT LIBRARIES
import os
from importlib import reload
from math import ceil
import random
import bisect
import itertools
from itertools import product
from collections import defaultdict, Counter
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
# import from clusters.py
from clusters import BuildClusters

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
            # simulate chance to reproduce
            agent.reproduce()
            for lang in ['L1', 'L2']:
                # update last-time word use vector
                agent.lang_stats[lang]['t'][~agent.day_mask[lang]] += 1
                # set current lang knowledge
                agent.lang_stats[lang]['pct'][agent.age] = (np.where(agent.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                            agent.model.vocab_red)
                # reset day mask
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
    ic_pct_keys = [10, 25, 50, 75, 90]
    def __init__(self, num_people, spoken_only=True, num_words_conv=(3, 25, 250), width=100, height=100,
                 max_people_factor=5, init_lang_distrib=[0.25, 0.65, 0.1], num_clusters=10, max_run_steps=1000,
                 lang_ags_sorted_by_dist=True, lang_ags_sorted_in_clust=True):
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
        self.num_clusters = num_clusters
        self.max_run_steps = max_run_steps
        self.lang_ags_sorted_by_dist = lang_ags_sorted_by_dist
        self.lang_ags_sorted_in_clust = lang_ags_sorted_in_clust
        self.clust_centers = None
        self.cluster_sizes = None

        # import lang ICs and lang CDFs data as function of steps. Use directory of executed file
        self.lang_ICs = dd.io.load(os.path.join(os.path.dirname(__file__), 'lang_spoken_ics_vs_step.h5'))
        self.cdf_data = dd.io.load(os.path.join(os.path.dirname(__file__), 'lang_cdfs_vs_step.h5'))

        # define model grid and schedule
        self.grid = MultiGrid(height, width, False)
        self.schedule = StagedActivation_modif(self,
                                               stage_list=["stage_1", "stage_2",
                                                           "stage_3", "stage_4"],
                                               shuffle=True,
                                               shuffle_between_stages=True)

        ## Define clusters parameters
        self.compute_cluster_centers()
        self.compute_cluster_sizes()
        self.set_clusters_info()
        # TODO : clusters = BuildClusters(self)


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
        self.map_jobs()
        self.map_schools()
        self.map_homes()
        self.map_lang_agents()
        self.define_family_networks()


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

    def compute_cluster_centers(self, min_dist=0.20, min_grid_pct_val=0.1, max_grid_pct_val=0.9):
        """ Generate 2D coordinates for all cluster (towns/villages) centers
            Args:
                * min_dist: float. Minimum distance between cluster centers
                            expressed as percentage of square grid AVAILABLE side dimension

            Returns:
                * A numpy array with 2D coordinates of cluster centers
        """

        # Define available points as pct of squared grid length
        grid_side_pcts = np.linspace(min_grid_pct_val, max_grid_pct_val, 100) # avoid edges
        # Define a set of all available gridpoints coordinates
        set_av_grid_pts = set(product(grid_side_pcts, grid_side_pcts))
        # initiate list of cluster centers and append random point from set
        self.clust_centers = []
        p = random.sample(set_av_grid_pts, 1)[0]
        self.clust_centers.append(p)
        # remove picked point from availability list
        set_av_grid_pts.remove(p)
        # Ensure min distance btwn cluster centers
        # Iterate over available points until all requested centers are found
        for _ in range(self.num_clusters - 1):
            # Before picking new point, remove all points too-close to existing centers
            # from availability set
            # Iterate over copy of original set since elements are being removed during iteration
            for point in set(set_av_grid_pts):
                if pdist([point, p]) < min_dist * (grid_side_pcts.max() - grid_side_pcts.min()):
                # if pdist([point, p]) < min_dist:
                    set_av_grid_pts.remove(point)
            # Try picking new center from availability set ( if it's not already empty...)
            try:
                p = random.sample(set_av_grid_pts, 1)[0]
            except ValueError:
                print('INPUT ERROR: Reduce either number of clusters or minimum distance '
                      'in order to meet distance constraint')
                raise
            # Add new center to return list and remove it from availability set
            self.clust_centers.append(p)
            set_av_grid_pts.remove(p)
        self.clust_centers = np.array(self.clust_centers)
        # If requested, sort cluster centers based on their distance to grid origin
        if self.lang_ags_sorted_by_dist:
            self.clust_centers = sorted(self.clust_centers, key=lambda x:pdist([x,[0,0]]))

    def compute_cluster_sizes(self, min_size=20):
        """ Method to compute sizes of each cluster in model.
            Cluster size equals number of language agents that live in this cluster

            Arguments:
                * min_size: minimum accepted cluster size ( integer)

            Returns:
                * list of integers representing cluster sizes

        """
        p = np.random.pareto(1.25, size=self.num_clusters)
        pcts = p / p.sum()
        self.cluster_sizes = np.random.multinomial(self.num_people, pcts)
        tot_sub_val = 0
        for idx in np.where(self.cluster_sizes < min_size)[0]:
            tot_sub_val += min_size - self.cluster_sizes[idx]
            self.cluster_sizes[idx] = min_size
        self.cluster_sizes[np.argmax(self.cluster_sizes)] -= tot_sub_val
        if np.min(self.cluster_sizes) < min_size:
            raise ValueError('INPUT ERROR: Impossible to build all clusters within minimum size. Make sure ratio '
                          'num_people/num_clusters >= min_size')

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
        x_coords = np.random.binomial(self.grid_width, pcts_grid[0], size=clust_size)
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
            self.clusters_info[idx]['homes'] = []
            self.clusters_info[idx]['agents'] = []

    def sort_coords_in_clust(self, x_coords, y_coords, clust_idx):
        """ Method that sorts a collection of x and y cluster coordinates
        according to distance from cluster center

        Args:
            * x_coords: list of floats. It represents x component of cluster points
            * y_coords: list of floats. It represents y component of cluster points
            * clust_idx: integer. It identifies cluster from list of clusters(towns) in model
        Returns:
            * sorted x_coords, y_coords lists ( 2 separate lists )
        """
        # compute coordinates of cluster center
        c_coords = self.grid_width * self.clust_centers[clust_idx]
        # sort list of coordinate tuples by distance
        sorted_coords = sorted(list(zip(x_coords, y_coords)), key=lambda pts:pdist([pts, c_coords]))
        # unzip list of tuples
        x_coords, y_coords = list(zip(*sorted_coords))
        return x_coords, y_coords

    def map_jobs(self, min_places=2, max_places=200):
        """ Generates job centers coordinates and num places per center
            Instantiates job objects
        Args:
            * min_places: integer. Minimum number of places for each job center
            * max_places: integer. Maximum number of places for each job center
        """
        # iterate to generate job center coordinates

        num_job_cs_per_agent = self.max_people_factor / ((max_places - min_places)/2)

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_j, y_j = self.generate_cluster_points_coords(clust_c_coords,
                                                           ceil(clust_size * num_job_cs_per_agent))
            if (not self.lang_ags_sorted_by_dist) and self.lang_ags_sorted_in_clust:
                x_j, y_j = self.sort_coords_in_clust(x_j, y_j, clust_idx)

            # compute percentage number places per each job center in cluster using lognormal distrib
            p = np.random.lognormal(1, 1, size=int(clust_size * num_job_cs_per_agent))
            pcts = p / p.sum()
            # compute num of places out of percentages
            num_places_job_c = np.random.multinomial(int(clust_size * self.max_people_factor), pcts)
            num_places_job_c = np.clip(num_places_job_c, min_places, max_places)

            for x, y, num_places in zip(x_j, y_j, num_places_job_c):
                self.clusters_info[clust_idx]['jobs'].append(Job((x,y), num_places))

    def map_schools(self, school_size=100):
        """ Generate coordinates for school centers and instantiate school objects
            Args:
                * school_size: fixed size of all schools generated
        """
        num_schools_per_agent = self.max_people_factor / school_size

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_s, y_s = self.generate_cluster_points_coords(clust_c_coords,
                                                           ceil(clust_size * num_schools_per_agent))
            if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
                x_s, y_s = self.sort_coords_in_clust(x_s, y_s, clust_idx)
            for x,y in zip(x_s, y_s):
                self.clusters_info[clust_idx]['schools'].append(School((x,y), school_size))

    def map_homes(self, num_people_per_home=4):
        """ Generate coordinates for agent homes and instantiate Home objects"""

        num_homes_per_agent = self.max_people_factor / num_people_per_home

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_h, y_h = self.generate_cluster_points_coords(clust_c_coords,
                                                           ceil(clust_size * num_homes_per_agent))
            # check if sorting within cluster is requested
            if (not self.lang_ags_sorted_by_dist) and (self.lang_ags_sorted_in_clust):
                x_h, y_h = self.sort_coords_in_clust(x_h, y_h, clust_idx)
            for x, y in zip(x_h, y_h):
                self.clusters_info[clust_idx]['homes'].append(Home((x,y)))

    def generate_lang_distrib(self):
        """ Method that generates a list of lists of lang labels in the requested order
            Returns:
                * A list of lists where each list contains lang labels per cluster
        """
        # generate random array with lang labels ( no sorting at all )
        langs_per_ag_array = np.random.choice([0, 1, 2], p=self.init_lang_distrib, size=self.num_people)
        idxs_to_split_by_clust = self.cluster_sizes.cumsum()[:-1]
        # check if agent-sorting by distance to origin is requested
        if self.lang_ags_sorted_by_dist:
            langs_per_ag_array.sort()
            # split lang labels by cluster
            langs_per_clust = np.split(langs_per_ag_array, idxs_to_split_by_clust)
            if not self.lang_ags_sorted_in_clust:
                for clust in langs_per_clust:
                    clust.sort()
                    clust_subsorted_by_fam = [fam for fam in zip(*[iter(clust)] * 4)]
                    random.shuffle(clust_subsorted_by_fam)
                    clust_subsorted_by_fam = [val for group in clust_subsorted_by_fam
                                                 for val in group]
                    clust[:len(clust_subsorted_by_fam)] = clust_subsorted_by_fam
        else:
            # split lang labels by cluster
            langs_per_clust = np.split(langs_per_ag_array, idxs_to_split_by_clust)
            # check if sorting within cluster is requested
            if self.lang_ags_sorted_in_clust:
                for clust in langs_per_clust:
                    clust.sort()  # invert if needed
            # if no clust sorting, need to cluster langs by groups of 4 for easier family definition
            else:
                for clust in langs_per_clust:
                    clust.sort()
                    clust_subsorted_by_fam = [fam for fam in zip(*[iter(clust)] * 4)]
                    random.shuffle(clust_subsorted_by_fam)
                    clust_subsorted_by_fam = [val for group in clust_subsorted_by_fam
                                                 for val in group]
                    clust[:len(clust_subsorted_by_fam)] = clust_subsorted_by_fam
        return langs_per_clust

    def map_lang_agents(self):
        """ Method to instantiate all agents according to requested linguistic order """
        # get lang distribution for each cluster
        langs_per_clust = self.generate_lang_distrib()
        # set agents ids
        ids = set(range(self.num_people))

        family_size = 4
        for clust_idx, clust_info in self.clusters_info.items():
            for idx, family_langs in enumerate(zip(*[iter(langs_per_clust[clust_idx])] * family_size)):
                # instantiate 2 adults with neither job nor home assigned
                ag1 = Simple_Language_Agent(self, ids.pop(), family_langs[0], city_idx=clust_idx)
                ag2 = Simple_Language_Agent(self, ids.pop(), family_langs[1], city_idx=clust_idx)

                # instantiate 2 adolescents with neither school nor home assigned
                ag3 = Simple_Language_Agent(self, ids.pop(), family_langs[2], city_idx=clust_idx)
                ag4 = Simple_Language_Agent(self, ids.pop(), family_langs[3], city_idx=clust_idx)

                # add agents to clust_info, schedule, grid and networks
                clust_info['agents'].extend([ag1, ag2, ag3, ag4])
                self.add_agent_to_grid_sched_networks(ag1, ag2, ag3, ag4)

            # set up agents left out of family partition of cluster
            len_clust = clust_info['num_agents']
            num_left_agents = len_clust%family_size
            if num_left_agents:
                for lang in clust_info['agents'][-num_left_agents:]:
                    ag = Simple_Language_Agent(self, ids.pop(), lang, city_idx=clust_idx)
                    clust_info['agents'].append(ag)
                    self.add_agent_to_grid_sched_networks(ag)

    def add_agent_to_grid_sched_networks(self, ag, *more_ags):
        """ Method to add a number of agents to grid, schedule and system networks
            Arguments:
                * a : agent class instance
                * more_ags : optional. A tuple of agent class instances if
                             more than one agent needs to be added at once
        """
        # make an iterable list out of all inputs
        agents = [ag, *more_ags]
        # add agent to grid and schedule
        for a in agents:
            self.schedule.add(a)
            if a.loc_info['home']:
                self.grid.place_agent(a, a.loc_info['home'].pos)
            else:
                self.grid.place_agent(a, (0, 0))
        ## add agent node to all networks
        self.known_people_network.add_nodes_from(agents)
        self.friendship_network.add_nodes_from(agents)
        self.family_network.add_nodes_from(agents)

    def define_family_networks(self):
        # Method to define families and also adds relatives to known_people_network
        # marriage, to make things simple, only allowed for combinations  0-1, 1-1, 1-2
        family_size = 4

        for clust_idx, clust_info in self.clusters_info.items():
            for idx, family in enumerate(zip(*[iter(clust_info['agents'])] * family_size)):
                # set ages of family members
                steps_per_year = 36
                min_age, max_age = 40 * steps_per_year, 50 * steps_per_year
                family[0].age, family[1].age = np.random.randint(min_age, max_age, size=2)
                min_age, max_age = 10 * steps_per_year, 20 * steps_per_year
                family[2].age, family[3].age = np.random.randint(min_age, max_age, size=2)

                # assign same home to all family members
                home = clust_info['homes'][idx]
                # import ICs and assign home
                # apply correlation between parents' and children's lang knowledge if parents bilinguals
                if 1 in [family[0].language, family[1].language]:
                    key_parents = [] # define list to store parents' percentage knowledge
                    for ix, member in enumerate(family):
                        if ix <2 and member.language == 1:
                            key = np.random.choice(self.ic_pct_keys)
                            key_parents.append(key)
                            member.set_lang_ics(biling_key=key)
                        elif ix < 2:
                            lang_mono = member.language
                            member.set_lang_ics()
                        elif ix >= 2:
                            if len(key_parents) == 1:
                                if not lang_mono: # mono in lang 0
                                    key = (key_parents[0] + 100) / 2
                                else: # mono in lang 1
                                    key = key_parents[0] / 2
                            else:
                                key = sum(key_parents) / len(key_parents)
                            key = self.ic_pct_keys[
                                bisect.bisect_left(self.ic_pct_keys, key, hi=len(self.ic_pct_keys)-1)
                            ]
                            member.set_lang_ics(biling_key=key)
                        member.loc_info['home'] = home
                        home.agents_in.add(member)
                else: # monolingual parents
                    # check if children are bilingual
                    if 1 in [family[2].language, family[3].language]:
                        for ix, member in enumerate(family):
                            if ix < 2:
                                member.set_lang_ics()
                            else:
                                if member.language == 1:
                                    # logical that child has much better knowledge of parents lang
                                    member.set_lang_ics(biling_key=90)
                                else:
                                    member.set_lang_ics()
                            member.loc_info['home'] = home
                            home.agents_in.add(member)
                    else:
                        for member in family:
                            member.set_lang_ics()
                            member.loc_info['home'] = home
                            home.agents_in.add(member)
                # assign job to parents
                for parent in family[:2]:
                    while True:
                        job = np.random.choice(clust_info['jobs'])
                        if job.num_places:
                            job.num_places -= 1
                            parent.loc_info['job'] = job
                            break
                # assign school to children
                # find closest school
                idx_school = np.argmin([pdist([home.pos, school.pos])
                                        for school in clust_info['schools']])
                school = clust_info['schools'][idx_school]
                for child in family[2:]:
                    child.loc_info['school'] = school
                # find out lang of interaction btw family members
                # consorts
                pct11 = family[0].lang_stats['L1']['pct'][family[0].age]
                pct12 = family[0].lang_stats['L2']['pct'][family[0].age]
                pct21 = family[1].lang_stats['L1']['pct'][family[1].age]
                pct22 = family[1].lang_stats['L2']['pct'][family[1].age]
                if (family[0].language, family[1].language) in [(0, 0), (0, 1), (1, 0)]:
                    lang_consorts = 0
                elif (family[0].language, family[1].language) in [(2, 1), (1, 2), (2, 2)]:
                    lang_consorts = 1
                elif (family[0].language, family[1].language) == (1, 1):
                    # Find weakest combination lang-agent, pick other language as common
                    idx_weakest = np.argmin([pct11, pct12, pct21, pct22])
                    if idx_weakest in [0, 2]:
                        lang_consorts = 1
                    else:
                        lang_consorts = 0
                # language with father, mother
                lang_with_father = np.argmax([pct11, pct12])
                lang_with_mother = np.argmax([pct21, pct22])
                # siblings
                avg_lang = (lang_with_father + lang_with_mother) / 2
                if avg_lang == 0:
                    lang_siblings = 0
                elif avg_lang == 1:
                    lang_siblings = 1
                else:
                    # find weakest, pick opposite
                    pct11 = family[2].lang_stats['L1']['pct'][family[2].age]
                    pct12 = family[2].lang_stats['L2']['pct'][family[2].age]
                    pct21 = family[3].lang_stats['L1']['pct'][family[3].age]
                    pct22 = family[3].lang_stats['L2']['pct'][family[3].age]
                    idx_weakest = np.argmin([pct11, pct12, pct21, pct22])
                    if idx_weakest in [0, 2]:
                        lang_siblings = 1
                    else:
                        lang_siblings = 0
                # add family edges in family and known_people networks
                for (i, j) in [(0, 1), (1, 0)]:
                    self.family_network.add_edge(family[i], family[j], fam_link='consort', lang=lang_consorts)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_consorts)
                for (i, j, link) in [(0, 2, 'child'), (2, 0, 'father'), (0, 3, 'child'), (3, 0, 'father')]:
                    self.family_network.add_edge(family[i], family[j], fam_link=link, lang=lang_with_father)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_with_father)
                for (i, j, link) in [(1, 2, 'child'), (2, 1, 'mother'), (1,3, 'child'), (3,1,'mother')]:
                    self.family_network.add_edge(family[i], family[j], fam_link=link, lang=lang_with_mother)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_with_mother)
                for (i, j) in [(2, 3), (3, 2)]:
                    self.family_network.add_edge(family[i], family[j], fam_link='sibling', lang=lang_siblings)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_siblings)

            # set up agents left out of family partition of cluster
            len_clust = len(clust_info['agents'])
            num_left_agents = len_clust%family_size
            if num_left_agents:
                for ag in clust_info['agents'][-num_left_agents:]:
                    min_age, max_age = 40 * steps_per_year, 60 * steps_per_year
                    ag.age = np.random.randint(min_age, max_age)
                    ag.set_lang_ics()
                    home = clust_info['homes'][idx + 1]
                    ag.loc_info['home'] = home
                    home.agents_in.add(ag)
                    while True:
                        job = np.random.choice(clust_info['jobs'])
                        if job.num_places:
                            job.num_places -= 1
                            ag.loc_info['job'] = job
                            break

    def define_friendship_networks(self):
        graph_friends = self.friendship_network

        # Apply small world graph to relevant nodes using networkx

        # how many friends per agent originally ??
        # Let's start with 2


        # Simple idea:
        # parents - > pick friends from job
        # children -> pick friends from school


    def run_conversation(self, ag_init, others):
        """ Method that models conversation between ag_init and others
            Calls method to determine conversation parameters
            Then makes each speaker speak and the rest listen (loop through all involved agents)
            Args:
                * ag_init : agent object instance. Agent that starts conversation
                * others : list of agent object instances. Rest of agents that take part in conversation
                           It can be a single agent object that will be automatically converted into a list
        """
        ags, conv_params = self.get_conv_params(ag_init, others)
        num_ags = len(ags)
        if not conv_params['bilingual']:
            conv_params['lang_group'] = itertools.repeat(conv_params['lang_group'], num_ags)
        for ix, (ag, lang) in enumerate(zip(ags, conv_params['lang_group'])):
            if ag.language != conv_params['mute_type']:
                spoken_words = ag.vocab_choice_model(lang, long=conv_params['long'])
                # call 'self' agent update
                ag.update_lang_arrays(lang, spoken_words)
                # call listeners' updates
                listeners = ags[:ix] + ags[ix + 1:]
                for listener in listeners:
                    listener.update_lang_arrays(lang, spoken_words, speak=False)
        # TODO : add option 'return ALL spoken words' so that bystanders can get words too ( agent 'listen' method)
        # need to split in 2 lang arrays
        # all_spoken_words = np.zeros(self.vocab_red, dtype=np.int64) # check new 1.11 numpy to set output type
        # if ret_spoken_words:
        #     all_spoken_words[spoken_words[0]] += spoken_words[1]  # list_words[act] += act_c
        # if ret_spoken_words:
        #     return all_spoken_words


    def get_conv_params(self, ag_init, others):
        """
        Method to find out parameters of conversation between 2 or more agents:
            conversation lang or lang spoken by each involved speaker,
            conversation type(mono or bilingual),
            mute agents (agents that only listen),
            conversation length.
        It implements MAXIMIN language rule from Van Parijs
        Args:
            * ag_init : agent object instance .Agent that starts conversation
            * others : list of agent object instances. Rest of agents that take part in conversation
                       It can be a single agent object that will be automatically converted into a list
        Returns:
            * lang: integer in [0, 1] if unique lang conv or list of integers in [0, 1] if multilang conversation
            * mute_type: integer. agent lang type that is unable to speak in conversation
        """

        # define list with all agents involved in conversation
        ags = [ag_init]
        try:
            iter(others)
        except TypeError:
            others = [others]
        ags.extend(others)
        num_ags = len(ags)
        conv_params = dict(bilingual=False, mute_type=None, long=True)
        # define lists with agent competences and preferences in each language
        l1_pcts = [ag.lang_stats['L1']['pct'][ag.age] for ag in ags]
        l2_pcts = [ag.lang_stats['L2']['pct'][ag.age] for ag in ags]
        fav_lang_per_agent = list(np.argmax([l1_pcts, l2_pcts], axis=0))
        ag_langs = set([ag.language for ag in ags])

        # define current case
        if ag_langs in [{0}, {0, 1}]: # TODO: need to save info of how init wanted to talk-> Feedback for AI learning
            lang_group = 0
            conv_params.update({'lang_group':0})
            #run_conversation(lang_group)
        elif ag_langs == {1}:
            # simplified PRELIMINARY NEUTRAL assumption: ag_init will start speaking the language they speak best
            # ( TODO : at this stage no modeling of place bias !!!!)
            # who starts conversation matters, but also average lang spoken with already known agents
            if l1_pcts[0] == l2_pcts[0]:
                lang_init = 1 if random.random() > 0.5 else 0
            else:
                lang_init = np.argmax([l1_pcts[0], l2_pcts[0]])
            langs_with_known_agents = [self.known_people_network[ag_init][ag]['lang']
                                       for ag in others
                                       if ag in self.known_people_network[ag_init]]
            if langs_with_known_agents:
                av_k_lang = round(sum(langs_with_known_agents) / len(langs_with_known_agents))
                lang_group = av_k_lang
            else:
                lang_group = lang_init
            conv_params.update({'lang_group': lang_group})
            #run_conversation(lang_group)
        elif ag_langs in [{1, 2}, {2}]:
            lang_group = 1
            conv_params.update({'lang_group': lang_group})
            #run_conversation(lang_group)
        else:
            # monolinguals on both linguistic sides => VERY SHORT CONVERSATION
            # get agents on both lang sides unable to speak in other lang
            idxs_real_monolings_l1 = [idx for idx, pct in enumerate(l2_pcts) if pct < 0.025]
            idxs_real_monolings_l2 = [idx for idx, pct in enumerate(l1_pcts) if pct < 0.025]

            if not idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # No complete monolinguals on either side
                # All agents partially understand each other langs, but some can't speak l1 and some can't speak l2
                # Conversation is possible when each agent picks their favorite lang
                lang_group = fav_lang_per_agent
                conv_params.update({'lang_group': lang_group, 'bilingual':True, 'long':False})

            elif idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # There are real L1 monolinguals in the group
                # Everybody partially understands L1, but some agents don't understand L2 at all
                # Some agents only understand and speak L1, while others partially understand but can't speak L1
                # slight bias towards l1 => conversation in l1 but some speakers will stay mute = > short conversation
                mute_type = 2
                if ag_init.language != mute_type:
                    lang_group = 0
                else:
                    lang_group, mute_type = 1, 0
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'long': False})

            elif not idxs_real_monolings_l1 and idxs_real_monolings_l2:
                # There are real L2 monolinguals in the group
                # Everybody partially understands L2, but some agents don't understand L1 at all
                # Some agents only understand and speak l2, while others partially understand but can't speak l2
                # slight bias towards l2 => conversation in L2 but some speakers will stay mute = > short conversation
                mute_type = 0
                if ag_init.language != mute_type:
                    lang_group = 1
                else:
                    lang_group, mute_type = 0, 2
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'long': False})

            else:
                # There are agents on both lang sides unable to follow other's speech.
                # Initiator agent will speak with whom understands him, others will listen but understand nothing

                if ag_init.language == 1:
                    # init agent is bilingual
                    # pick majority lang
                    num_l1_speakers = sum([1 if pct >= 0.1 else 0 for pct in l1_pcts])
                    num_l2_speakers = sum([1 if pct >= 0.1 else 0 for pct in l2_pcts])
                    if num_l1_speakers > num_l2_speakers:
                        lang_group, mute_type = 0, 2
                    elif num_l1_speakers < num_l2_speakers:
                        lang_group, mute_type = 2, 0
                    else:
                        lang_group = fav_lang_per_agent[0]
                        mute_type = 2 if lang_group == 0 else 0
                else:
                    # init agent is monolang
                    lang_group = fav_lang_per_agent[0]
                    mute_type = 2 if lang_group == 0 else 0
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'long': False})


        return ags, conv_params


    def get_lang_stats(self, i):
        """Method to get counts of each type of lang agent

        Arguments:
            * i : integer from [0,1,2] that specifies agent lang type

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
        """ Get value of specific attribute for all lang agents in model """
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
        ani = animation.FuncAnimation(fig, run_and_update, init_func=init_show,
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
                                                 'num_clusters': self.num_clusters,
                                                 'sort_by_dist': self.lang_ags_sorted_by_dist,
                                                 'sort_within_clust': self.lang_ags_sorted_in_clust},
                           'model_results': self.datacollector.get_model_vars_dataframe(),
                           'agent_results': self.datacollector.get_agent_vars_dataframe()}
        if self.save_dir:
            dd.io.save(self.save_dir + '/model_data.h5', self.model_data)
        else:
            dd.io.save('model_data.h5', self.model_data)

    def load_model_data(self, data_filename, key='/' ):
        return dd.io.load(data_filename, key)

    def plot_family_networks(self):
        """PLOT NETWORK with lang colors and position"""
        #        people_pos = [elem.pos for elem in self.schedule.agents if elem.agent_type == 'language']
        #        # Following works because lang agents are added before other agent types
        #        people_pos_dict= dict(zip(self.schedule.agents, people_pos))
        people_pos_dict = {elem: elem.pos
                           for elem in self.schedule.agents}
        people_color = [elem.language for elem in self.family_network]
        nx.draw(self.family_network, pos=people_pos_dict, node_color=people_color)

