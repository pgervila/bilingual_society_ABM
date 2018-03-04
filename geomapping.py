import sys
from importlib import reload
import numpy as np
import random
from scipy.spatial.distance import pdist
from itertools import product
from collections import defaultdict
from math import ceil

# IMPORT AGENTS AND  ENTITIES
import agent, city_objects
reload(sys.modules['agent'])
reload(sys.modules['city_objects'])
from agent import Child, Adult
from city_objects import Job, School, University, Home


class GeoMapper:
    def __init__(self, model, num_clusters):
        self.model = model
        self.num_clusters = num_clusters
        self.random_seeds = np.random.randint(1, 10000, size=2)
        # setup clusters
        self.compute_cluster_centers()
        self.compute_cluster_sizes()
        self.set_clusters_info()

    def map_model_objects(self):
        """ Instantiate and position all model objects on 2-D grid """
        self.map_jobs()
        self.map_schools()
        self.map_universities()
        self.map_homes()
        self.map_lang_agents()

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
        pt = random.choice(list(set_av_grid_pts))
        self.clust_centers.append(pt)
        # remove picked point from availability list
        set_av_grid_pts.remove(pt)
        # Ensure min distance btwn cluster centers
        # Iterate over available points until all requested centers are found
        for _ in range(self.num_clusters - 1):
            # Before picking new point, remove all points too-close to existing centers
            # from availability set
            # Iterate over copy of original set since elements are being removed during iteration
            for point in set(set_av_grid_pts):
                if pdist([point, pt]) < min_dist * (grid_side_pcts.max() - grid_side_pcts.min()):
                # if pdist([point, pt]) < min_dist:
                    set_av_grid_pts.remove(point)
            # Try picking new center from availability set ( if it's not already empty...)
            try:
                pt = random.choice(list(set_av_grid_pts))
            except ValueError:
                print('INPUT ERROR: Reduce either number of clusters or minimum distance '
                      'in order to meet distance constraint')
                raise
            # Add new center to return list and remove it from availability set
            self.clust_centers.append(pt)
            set_av_grid_pts.remove(pt)
        self.clust_centers = np.array(self.clust_centers)
        # If requested, sort cluster centers based on their distance to grid origin
        if self.model.lang_ags_sorted_by_dist:
            self.clust_centers = sorted(self.clust_centers, key=lambda x:pdist([x,[0,0]]))

    def compute_cluster_sizes(self, min_size=100):
        """ Method to compute sizes of each cluster in model.
            Cluster size equals number of language agents that live in this cluster

            Arguments:
                * min_size: minimum accepted cluster size ( integer)

            Returns:
                * list of integers representing cluster sizes

        """
        np.random.seed(self.random_seeds[0])
        p = np.random.pareto(1.25, size=self.num_clusters)
        pcts = p / p.sum()
        np.random.seed(self.random_seeds[1])
        self.cluster_sizes = np.random.multinomial(self.model.num_people, pcts)
        tot_sub_val = 0
        for idx in np.where(self.cluster_sizes < min_size)[0]:
            tot_sub_val += min_size - self.cluster_sizes[idx]
            self.cluster_sizes[idx] = min_size
        self.cluster_sizes[np.argmax(self.cluster_sizes)] -= tot_sub_val
        if np.min(self.cluster_sizes) < min_size:
            raise ValueError('INPUT ERROR: Impossible to build all clusters within minimum size. Make sure ratio '
                             'num_people/num_clusters >= min_size')

    def sort_clusts_by_dist(self, ix):
        return sorted(range(len(self.clust_centers)),
                      key=lambda x: pdist([self.clust_centers[x],
                                           self.clust_centers[ix]])[0])

    def set_clusters_info(self):
        """ Create dict container for all cluster info"""
        self.clusters_info = defaultdict(dict)
        for idx, clust_size in enumerate(self.cluster_sizes):
            self.clusters_info[idx]['num_agents'] = clust_size
            self.clusters_info[idx]['closest_clusters'] = self.sort_clusts_by_dist(idx)
            self.clusters_info[idx]['jobs'] = []
            self.clusters_info[idx]['schools'] = []
            self.clusters_info[idx]['homes'] = []
            self.clusters_info[idx]['agents'] = []

    def generate_points_coords(self, pcts_grid, clust_size, clust_idx):
        """ Using binomial distribution, this method generates initial coordinates
            for a given cluster, defined via its center and its size.
            Cluster size as well as cluster center coords (in grid percentage) must be provided
        Arguments:
            * pcts_grid: 2-D tuple with positive floats < 1 to define clust_center along grid width and height
            * clust_size: desired size of the cluster being generated
            * clust_idx : integer. Cluster index for which points have to be generated
        Returns:
            * cluster_coordinates: two numpy arrays with x and y coordinates
            respectively
        """
        x_coords = np.random.binomial(self.model.grid.width, pcts_grid[0], size=clust_size)
        # limit coords values to grid boundaries
        x_coords = np.clip(x_coords, 1, self.model.grid.width - 1)

        y_coords = np.random.binomial(self.model.grid.height, pcts_grid[1], size=clust_size)
        # limit coords values to grid boundaries
        y_coords = np.clip(y_coords, 1, self.model.grid.height - 1)

        if (not self.model.lang_ags_sorted_by_dist) and self.model.lang_ags_sorted_in_clust:
            x_coords, y_coords = self.sort_coords_in_clust(x_coords, y_coords, clust_idx)

        return x_coords, y_coords

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
        c_coords = self.model.grid.width * self.clust_centers[clust_idx]
        # sort list of coordinate tuples by distance
        sorted_coords = sorted(list(zip(x_coords, y_coords)), key=lambda pts: pdist([pts, c_coords]))
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

        num_job_cs_per_agent = self.model.max_people_factor / ((max_places - min_places)/2)

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_j, y_j = self.generate_points_coords(clust_c_coords,
                                                   ceil(clust_size * num_job_cs_per_agent),
                                                   clust_idx)
            # compute percentage number places per each job center in cluster using lognormal distrib
            p = np.random.lognormal(1, 1, size=int(clust_size * num_job_cs_per_agent))
            pcts = p / p.sum()
            # compute num of places out of percentages
            num_places_job_c = np.random.multinomial(int(clust_size * self.model.max_people_factor), pcts)
            num_places_job_c = np.clip(num_places_job_c, min_places, max_places)

            for x, y, num_places in zip(x_j, y_j, num_places_job_c):
                self.clusters_info[clust_idx]['jobs'].append(Job(clust_idx, (x, y), num_places, lang_policy=[0, 1]))

    def map_schools(self, max_school_size=100, min_school_size=40, buffer_factor=1.2):
        """ Generate coordinates for school centers and instantiate school objects
            Args:
                * max_school_size: integer. Maximum size of all schools generated
                * min_school_size: integer. Minimum size of all schools generated. Must be < max_school_size
                * buffer_factor: float > 1. Allows to define buffer wrt current number of student population
        """

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            school_places_per_cluster = int(buffer_factor * clust_size / 2)
            num_schools_per_cluster = ceil(school_places_per_cluster / max_school_size)
            # generate school coords
            x_s, y_s = self.generate_points_coords(clust_c_coords, num_schools_per_cluster, clust_idx)
            if (not self.model.lang_ags_sorted_by_dist) and self.model.lang_ags_sorted_in_clust:
                x_s, y_s = self.sort_coords_in_clust(x_s, y_s, clust_idx)
            for x, y in zip(x_s, y_s):
                if school_places_per_cluster < max_school_size:
                    school_size = max(min_school_size, school_places_per_cluster)
                else:
                    school_size = max_school_size
                school = School(self.model, (x, y), clust_idx, school_size,
                                lang_policy=self.model.school_lang_policy)
                self.clusters_info[clust_idx]['schools'].append(school)
                school_places_per_cluster -= school_size

    def map_universities(self, pct_univ_towns = 0.2):
        num_univ = ceil(pct_univ_towns * self.num_clusters)
        ixs_sorted = np.argsort(self.cluster_sizes)[::-1][:num_univ]
        for clust_idx in ixs_sorted:
            x_u, y_u = self.generate_points_coords(self.clust_centers[clust_idx], 1, clust_idx)
            univ = University(self.model, (x_u[0], y_u[0]), clust_idx)
            self.clusters_info[clust_idx]['university'] = univ

    def map_homes(self, num_people_per_home=4):
        """ Generate coordinates for agent homes and instantiate Home objects"""

        num_homes_per_agent = self.model.max_people_factor / num_people_per_home

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_h, y_h = self.generate_points_coords(clust_c_coords,
                                                   ceil(clust_size * num_homes_per_agent),
                                                   clust_idx)
            for x, y in zip(x_h, y_h):
                self.clusters_info[clust_idx]['homes'].append(Home(clust_idx, (x, y)))

    def generate_lang_distrib(self):
        """ Method that generates a list of lists of lang labels in the requested order
            Returns:
                * A list of lists where each list contains lang labels per cluster
        """
        # generate random array with lang labels ( no sorting at all )
        langs_per_ag_array = np.random.choice([0, 1, 2],
                                              p=self.model.init_lang_distrib,
                                              size=self.model.num_people)
        idxs_to_split_by_clust = self.cluster_sizes.cumsum()[:-1]
        # check if agent-sorting by distance to origin is requested
        if self.model.lang_ags_sorted_by_dist:
            langs_per_ag_array.sort()
            # split lang labels by cluster
            langs_per_clust = np.split(langs_per_ag_array, idxs_to_split_by_clust)
            if not self.model.lang_ags_sorted_in_clust:
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
            if self.model.lang_ags_sorted_in_clust:
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
        lang_ags_ids = set(range(self.model.num_people))

        for clust_idx, clust_info in self.clusters_info.items():
            for idx, family_langs in enumerate(zip(*[iter(langs_per_clust[clust_idx])] * self.model.family_size)):
                # family sexes
                family_sexes = ['M', 'F'] + ['M' if random.random() < 0.5 else 'F' for _ in range(2)]
                # instantiate 2 adults with neither job nor home assigned
                ag1 = Adult(self.model, lang_ags_ids.pop(), family_langs[0], family_sexes[0])
                ag2 = Adult(self.model, lang_ags_ids.pop(), family_langs[1], family_sexes[1])

                # instantiate 2 children with neither school nor home assigned
                ag3 = Child(self.model, lang_ags_ids.pop(), family_langs[2], family_sexes[2])
                ag4 = Child(self.model, lang_ags_ids.pop(), family_langs[3], family_sexes[3])

                # add agents to clust_info, schedule, grid and networks
                clust_info['agents'].extend([ag1, ag2, ag3, ag4])
                self.add_agents_to_grid_and_schedule([ag1, ag2, ag3, ag4])
                import ipdb; ipdb.set_trace()

                # TODO : need to delete all original references to agent instances except for
                # networks, grid, schedule

            # set up agents left out of family partition of cluster
            len_clust = clust_info['num_agents']
            num_left_agents = len_clust % self.model.family_size
            if num_left_agents:
                for lang in langs_per_clust[clust_idx][-num_left_agents:]:
                    sex = ['M' if random.random() < 0.5 else 'F']
                    ag = Adult(self.model, lang_ags_ids.pop(), lang, sex)
                    clust_info['agents'].append(ag)
                    self.add_agents_to_grid_and_schedule(ag)


    def add_agents_to_grid_and_schedule(self, ags):
        """ Method to add a number of agents to grid, schedule and system networks
            Arguments:
                * ags : a single agent class instance or a list of class instances. In either case,
                    ags will be transformed into an iterable
        """
        # construct an iterable list out of all possible input structure
        ags = [ags] if type(ags) != list else ags
        # add agent to grid and schedule
        for ag in ags:
            self.model.schedule.add(ag)
            if ag.loc_info['home']:
                self.model.grid.place_agent(ag, ag.loc_info['home'].pos)
            else:
                self.model.grid.place_agent(ag, (0, 0))

