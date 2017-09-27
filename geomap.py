import numpy as np
import random
from scipy.spatial.distance import pdist
from itertools import product
from collections import defaultdict
from math import ceil


from agent_simple import Language_Agent, Job, School, Home

class GeoMap:
    def __init__(self, model):
        self.model = model

    def compute_cluster_centers(self, min_dist=0.20, min_grid_pct_val=0.1, max_grid_pct_val=0.9):
        """ Generate 2D coordinates for all cluster( towns/villages )centers
            Args:
                * min_dist: float. Minimum distance between cluster centers
                            expressed as percentage of square grid AVAILABLE side dimension
                * min_grid_pct_val: float
                * max_grid_pct_val: float

            Returns:
                * A numpy array with 2D coordinates of cluster centers
        """
        # Define available points as pct of squared grid length
        grid_side_pcts = np.linspace(min_grid_pct_val, max_grid_pct_val, 100)  # avoid edges
        # Define a set of all available gridpoints coordinates
        set_av_grid_pts = set(product(grid_side_pcts, grid_side_pcts))
        # initiate list of cluster centers and append random point from set
        self.model.clust_centers = []
        p = random.sample(set_av_grid_pts, 1)[0]
        self.model.clust_centers.append(p)
        # remove picked point from availability list
        set_av_grid_pts.remove(p)
        # Ensure min distance btwn cluster centers
        # Iterate over available points until all requested centers are found
        for _ in range(self.model.num_clusters - 1):
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
            self.model.clust_centers.append(p)
            set_av_grid_pts.remove(p)
        self.model.clust_centers = np.array(self.model.clust_centers)
        # If requested, sort cluster centers based on their distance to grid origin
        if self.model.lang_ags_sorted_by_dist:
            self.model.clust_centers = sorted(self.model.clust_centers, key=lambda x: pdist([x, [0, 0]]))

    def compute_cluster_sizes(self, min_size=20):
        """ Method to compute sizes of each cluster in model.
            Cluster size equals number of language agents that live in this cluster

            Arguments:
                * min_size: minimum accepted cluster size ( integer)

            Returns:
                * list of integers representing cluster sizes

        """
        p = np.random.pareto(1.25, size=self.model.num_clusters)
        pcts = p / p.sum()
        self.model.cluster_sizes = np.random.multinomial(self.model.num_people, pcts)
        tot_sub_val = 0
        for idx in np.where(self.model.cluster_sizes < min_size)[0]:
            tot_sub_val += min_size - self.model.cluster_sizes[idx]
            self.model.cluster_sizes[idx] = min_size
        self.model.cluster_sizes[np.argmax(self.model.cluster_sizes)] -= tot_sub_val
        if np.min(self.model.cluster_sizes) < min_size:
            raise ValueError(
                'INPUT ERROR: Impossible to build all clusters within minimum size. Make sure ratio '
                'num_people/num_clusters >= min_size')

    def set_clusters_info(self):
        """ Create dict container for all cluster info"""
        self.model.clusters_info = defaultdict(dict)

        for idx, clust_size in enumerate(self.model.cluster_sizes):
            self.model.clusters_info[idx]['num_agents'] = clust_size
            self.model.clusters_info[idx]['jobs'] = []
            self.model.clusters_info[idx]['schools'] = []
            self.model.clusters_info[idx]['homes'] = []
            self.model.clusters_info[idx]['agents'] = []

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
        x_coords = np.random.binomial(self.model.grid_width, pcts_grid[0], size=clust_size)
        # limit coords values to grid boundaries
        x_coords = np.clip(x_coords, 1, self.model.grid_width - 1)

        y_coords = np.random.binomial(self.model.grid_height, pcts_grid[1], size=clust_size)
        # limit coords values to grid boundaries
        y_coords = np.clip(y_coords, 1, self.model.grid_height - 1)

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
        c_coords = self.model.grid_width * self.model.clust_centers[clust_idx]
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

        num_job_cs_per_agent = self.model.max_people_factor / ((max_places - min_places)/2)

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.model.cluster_sizes,
                                                                     self.model.clust_centers)):
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
                self.model.clusters_info[clust_idx]['jobs'].append(Job((x,y), num_places))

    def map_schools(self, school_size=100):
        """ Generate coordinates for school centers and instantiate school objects
            Args:
                * school_size: fixed size of all schools generated
        """
        num_schools_per_agent = self.model.max_people_factor / school_size

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.model.cluster_sizes,
                                                                     self.model.clust_centers)):
            x_s, y_s = self.generate_points_coords(clust_c_coords,
                                                   ceil(clust_size * num_schools_per_agent),
                                                   clust_idx)
            for x, y in zip(x_s, y_s):
                self.model.clusters_info[clust_idx]['schools'].append(School((x,y), school_size))

    def map_homes(self, num_people_per_home=4):
        """ Generate coordinates for agent homes and instantiate Home objects"""

        num_homes_per_agent = self.model.max_people_factor / num_people_per_home

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.model.cluster_sizes,
                                                                     self.model.clust_centers)):
            x_h, y_h = self.generate_points_coords(clust_c_coords,
                                                   ceil(clust_size * num_homes_per_agent),
                                                   clust_idx)
            for x, y in zip(x_h, y_h):
                self.model.clusters_info[clust_idx]['homes'].append(Home((x,y)))

    def generate_lang_distrib(self):
        """ Method that generates a list of lists of lang labels in the requested order
            Returns:
                * A list of lists where each list contains lang labels per cluster
        """
        # generate random array with lang labels ( no sorting at all )
        langs_per_ag_array = np.random.choice([0, 1, 2],
                                              p=self.model.init_lang_distrib,
                                              size=self.model.num_people)
        idxs_to_split_by_clust = self.model.cluster_sizes.cumsum()[:-1]
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
        langs_per_clust = self.model.generate_lang_distrib()
        # set agents ids
        ids = set(range(self.model.num_people))

        for clust_idx, clust_info in self.model.clusters_info.items():
            for idx, family_langs in enumerate(zip(*[iter(langs_per_clust[clust_idx])] * self.model.family_size)):
                # instantiate 2 adults with neither job nor home assigned
                ag1 = Language_Agent(self, ids.pop(), family_langs[0], city_idx=clust_idx)
                ag2 = Language_Agent(self, ids.pop(), family_langs[1], city_idx=clust_idx)

                # instantiate 2 adolescents with neither school nor home assigned
                ag3 = Language_Agent(self, ids.pop(), family_langs[2], city_idx=clust_idx)
                ag4 = Language_Agent(self, ids.pop(), family_langs[3], city_idx=clust_idx)

                # add agents to clust_info, schedule, grid and networks
                clust_info['agents'].extend([ag1, ag2, ag3, ag4])
                self.add_agent_to_grid_sched_networks(ag1, ag2, ag3, ag4)

            # set up agents left out of family partition of cluster
            len_clust = clust_info['num_agents']
            num_left_agents = len_clust % self.model.family_size
            if num_left_agents:
                for lang in langs_per_clust[clust_idx][-num_left_agents:]:
                    ag = Language_Agent(self, ids.pop(), lang, city_idx=clust_idx)
                    clust_info['agents'].append(ag)
                    self.add_agent_to_grid_sched_networks(ag)

    def setup_GeoMap(self):
        # clusters
        self.compute_cluster_centers()
        self.compute_cluster_sizes()
        self.set_clusters_info()
        # mapping
        self.map_jobs()
        self.map_schools()
        self.map_homes()
        self.map_lang_agents()
