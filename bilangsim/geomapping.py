import itertools as it
import numpy as np
import random
from scipy.spatial.distance import pdist
from itertools import product
from collections import defaultdict, Counter
from math import ceil

# IMPORT AGENTS AND  ENTITIES
# import city_objects
# reload(sys.modules['city_objects'])
from .agent import Child, Adolescent, Young, Adult
from .places import Job, School, University, Home


class GeoMapper:
    def __init__(self, model, num_clusters):
        self.model = model
        self.num_clusters = num_clusters
        self.random_seeds = np.random.randint(1, 10000, size=2)
        # setup clusters
        self.compute_cluster_centers()
        self.compute_cluster_sizes()
        self.set_clusters_info()
        # generate lang distribution for each cluster
        self.generate_langs_per_clust()

    def import_geo(self):
        """ Idea: import coordinates and assign them to model current object definition """
        # import clusters' centers and sizes
        # import jobs, schools, univs locations
        # import agents
        pass

    def import_model_ics(self):
        """ Import data from save model computation and map it to current model """
        pass

    def map_model_objects(self):
        """ Instantiate and locate all model objects on 2-D grid """
        self.map_jobs()
        self.map_schools()
        self.map_universities()
        self.map_homes()
        self.map_lang_agents()

    def compute_cluster_centers(self, min_dist=0.10, min_grid_pct_val=0.1, max_grid_pct_val=0.9):

        """
            Generate 2D coordinates for all cluster (towns/villages) centers
            Args:
                * min_dist: float. Minimum distance between cluster centers
                            expressed as percentage of square grid AVAILABLE side dimension
                * min_grid_pct_val: float
                * max_grid_pct_val: float

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

    def generate_dim_coord(self, dim, pct_grid, clust_size=None):
        """ Generate coordinate along a given dimension for a given cluster
            Args:
                * dim: integer. Total length of chosen dimension
                * pct_grid: float (positive, <1). Coordinate of cluster center
                * clust_size: integer. Number of requested coordinates
        """
        coord = np.random.binomial(dim, pct_grid, size=clust_size)
        # limit coordinate value to grid boundaries
        return np.clip(coord, 1, dim - 1)

    def generate_points_coords(self, pcts_grid, clust_size, clust_idx):
        """
            Using binomial distribution, this method generates initial coordinates
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
        # x_coords = np.random.binomial(self.model.grid.width, pcts_grid[0], size=clust_size)
        # # limit coords values to grid boundaries
        # x_coords = np.clip(x_coords, 1, self.model.grid.width - 1)
        #
        # y_coords = np.random.binomial(self.model.grid.height, pcts_grid[1], size=clust_size)
        # # limit coords values to grid boundaries
        # y_coords = np.clip(y_coords, 1, self.model.grid.height - 1)


        x_coords = self.generate_dim_coord(self.model.grid.width, pcts_grid[0],
                                           clust_size=clust_size)
        y_coords = self.generate_dim_coord(self.model.grid.height, pcts_grid[1],
                                           clust_size=clust_size)

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
        """
            Generates job centers coordinates and num places per center
            Instantiates job objects
            Args:
                * min_places: integer. Minimum number of places for each job center
                * max_places: integer. Maximum number of places for each job center
        """
        # iterate to generate job center coordinates

        num_job_cs_per_agent = self.model.max_people_factor / ((max_places - min_places)/2)

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            # adults account for approximately half of the cluster size
            clust_size = int(clust_size / 2)
            x_j, y_j = self.generate_points_coords(clust_c_coords,
                                                   ceil(clust_size * num_job_cs_per_agent),
                                                   clust_idx)
            # compute percentage number places per each job center in cluster using lognormal distrib
            # many small companies and a few of very large ones
            p = np.random.lognormal(1, 1, size=int(clust_size * num_job_cs_per_agent))
            pcts = p / p.sum()
            # compute num of places out of percentages
            num_places_job_c = np.random.multinomial(int(clust_size * self.model.max_people_factor), pcts)
            # adapt array to minimum and maximum number of places
            num_places_job_c = np.clip(num_places_job_c, min_places, max_places)

            for x, y, num_places in zip(x_j, y_j, num_places_job_c):
                if self.model.jobs_lang_policy:
                    self.clusters_info[clust_idx]['jobs'].append(Job(self.model, clust_idx, (x, y), num_places,
                                                                     lang_policy=self.model.jobs_lang_policy))
                else:
                    self.clusters_info[clust_idx]['jobs'].append(Job(self.model, clust_idx, (x, y), num_places))

    def map_schools(self, max_school_size=400, min_school_size=40, buffer_factor=1.2):
        """ Generate coordinates for school centers and instantiate school objects
            Args:
                * max_school_size: integer. Maximum size of all schools generated
                * min_school_size: integer. Minimum size of all schools generated. Must be < max_school_size
                * buffer_factor: float > 1. Allows to define buffer wrt current number of student population
        """

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            if not self.model.school_system['split']:
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
                                    lang_system=1)
                    self.clusters_info[clust_idx]['schools'].append(school)
                    school_places_per_cluster -= school_size
            else:
                langs = [0, 2]
                school_places_per_cluster = int(buffer_factor * clust_size / 2)
                pcts = self.get_lang_distrib_per_clust(clust_idx)
                num_schools_per_cl_and_lang = []
                for lang in langs:
                    pct = pcts[lang] + pcts[1]/2
                    if pct >= self.model.school_system['min_pct']:
                        num_schools_per_cl_and_lang.append(ceil(pct * school_places_per_cluster /
                                                                max_school_size))
                    else:
                        num_schools_per_cl_and_lang.append(0)
                num_schools_per_cluster = sum(num_schools_per_cl_and_lang)

                # generate school coords
                x_s, y_s = self.generate_points_coords(clust_c_coords,
                                                       num_schools_per_cluster,
                                                       clust_idx)
                if (not self.model.lang_ags_sorted_by_dist) and self.model.lang_ags_sorted_in_clust:
                    x_s, y_s = self.sort_coords_in_clust(x_s, y_s, clust_idx)
                zipped_coords = zip(x_s, y_s)
                slices = (num_schools_per_cl_and_lang[0], num_schools_per_cl_and_lang[1])
                iterable = iter(zipped_coords)
                sliced_zipped_coords = [list(it.islice(iterable, sl)) for sl in slices]
                for coords, lang in zip(sliced_zipped_coords, langs):
                    for x, y in coords:
                        school = School(self.model, (x, y), clust_idx,
                                        max_school_size, lang_system=lang)
                        self.clusters_info[clust_idx]['schools'].append(school)











    def map_universities(self, pct_univ_towns=0.2):
        num_univ = ceil(pct_univ_towns * self.num_clusters)
        ixs_sorted = np.argsort(self.cluster_sizes)[::-1][:num_univ]
        for clust_idx in ixs_sorted:
            x_u, y_u = self.generate_points_coords(self.clust_centers[clust_idx], 1, clust_idx)
            univ = University(self.model, (x_u[0], y_u[0]), clust_idx,
                              lang_policy=self.model.univ_lang_policy)
            self.clusters_info[clust_idx]['university'] = univ

    def map_homes(self, num_people_per_home=4):
        """ Generate coordinates for agent's homes and instantiate Home objects"""

        num_homes_per_agent = self.model.max_people_factor / num_people_per_home

        for clust_idx, (clust_size, clust_c_coords) in enumerate(zip(self.cluster_sizes,
                                                                     self.clust_centers)):
            x_h, y_h = self.generate_points_coords(clust_c_coords,
                                                   ceil(clust_size * num_homes_per_agent),
                                                   clust_idx)
            for x, y in zip(x_h, y_h):
                self.clusters_info[clust_idx]['homes'].append(Home(clust_idx, (x, y)))

    def create_new_agent(self, ag_class, ag_id, lang, sex=None, age=None, age_range=None):
        if not sex:
            sex = 'M' if random.random() < 0.5 else 'F'
        else:
            sex = sex
        ag = ag_class(self.model, ag_id, lang, sex)
        if not age:
            if age_range:
                ag.info['age'] = self.model.steps_per_year * np.random.randint(age_range[0],
                                                                               age_range[1])
            else:
                ag.info['age'] = self.model.steps_per_year * np.random.randint(0, 100)
        else:
            ag.info['age'] = age
        return ag

    def create_new_family(self, family_langs, parents_age_range=(25, 50),
                          children_age_range=(2, 17), max_age_diff=35):
        """ Method to create 4 new agent instances: 2 parents and two children
            Args:
                * family_langs: list of integers.
                * parents_age_range: tuple. Minimum and maximum parent's age
                * children_age_range: tuple. Minimum and maximum children's age
                * max_age_diff: integer. Maximum allowed age difference between parent and child"""
        av_ags_ids = self.model.set_available_ids
        # family sexes
        family_sexes = ['M', 'F'] + ['M' if random.random() < 0.5 else 'F' for _ in range(2)]
        # get parents ages
        age1 = np.random.randint(*parents_age_range)
        age2 = max(parents_age_range[0],
                   min(parents_age_range[1], age1 + np.random.randint(-5, 5)))
        # instantiate 2 adults with neither job nor home assigned
        # lang ics will be set later on when defining family network
        ag_type = Adult if age1 >= 30 else Young
        ag1 = ag_type(self.model, av_ags_ids.pop(), family_langs[0], family_sexes[0],
                      age=self.model.steps_per_year * age1, married=True, num_children=2)
        ag_type = Adult if age2 >= 30 else Young
        ag2 = ag_type(self.model, av_ags_ids.pop(), family_langs[1], family_sexes[1],
                      age=self.model.steps_per_year * age2, married=True, num_children=2)
        # instantiate 2 children with neither school nor home assigned
        # lang ics will be set later on when defining family network
        # get children ages
        age3, age4 = np.random.randint(max(2, age2 - max_age_diff),
                                       min(children_age_range[1],
                                           age2 - parents_age_range[0] + children_age_range[0] + 1),
                                       size=2)
        if age3 == age4:
            if age4 + 2 <= children_age_range[1]:
                age4 += 2
            elif age4 - 2 >= children_age_range[0]:
                age4 -= 2

        ag_type = Adolescent if age3 >= 12 else Child
        ag3 = ag_type(self.model, av_ags_ids.pop(), family_langs[2], family_sexes[2],
                      age=self.model.steps_per_year * age3)
        ag_type = Adolescent if age4 >= 12 else Child
        ag4 = ag_type(self.model, av_ags_ids.pop(), family_langs[3], family_sexes[3],
                      age=self.model.steps_per_year * age4)
        family = [ag1, ag2, ag3, ag4]
        # set linguistic ICs to family
        self.model.set_lang_ics_in_family(family)
        return family

    def add_new_home(self, clust_id):
        """ Method to add a random new home to a given cluster
            Args:
                * clust_id: integer. Cluster index
        """
        x_c, y_c = self.model.geo.clust_centers[clust_id]
        x_nh = self.model.geo.generate_dim_coord(self.model.grid.width, x_c)
        y_nh = self.model.geo.generate_dim_coord(self.model.grid.height, y_c)
        new_home = Home(clust_id, (x_nh, y_nh))
        self.model.geo.clusters_info[clust_id]['homes'].append(new_home)
        return new_home

    def add_new_school(self, clust_id, lang_system):
        """ Method to add a new school to a given cluster
            Args:
                * clust_id: integer. Cluster index
        """
        x_c, y_c = self.model.geo.clust_centers[clust_id]
        x_nsc = self.model.geo.generate_dim_coord(self.model.grid.width, x_c)
        y_nsc = self.model.geo.generate_dim_coord(self.model.grid.height, y_c)
        new_school = School(self.model, (x_nsc, y_nsc), clust_id, num_places=400,
                            lang_system=lang_system)
        self.clusters_info[clust_id]['schools'].append(new_school)
        return new_school

    def generate_langs_per_clust(self):
        """ Method that generates a list of lists of lang labels.
            Each sublist represents a cluster
            Output:
                * Sets value of attribute 'langs_per_clust'.
                    A list of lists where each list contains lang labels per cluster
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

        self.langs_per_clust = langs_per_clust

    def map_lang_agents(self, parents_age_range=(25, 50), children_age_range=(2, 17),
                        max_age_diff=35):
        """
            Method to instantiate all agents grouped by families of 4 members and
            according to requested linguistic order.
            It also assigns a home and an occupation to each agent
            Args:
                * parents_age_range: tuple. Minimum and maximum parent's age
                * children_age_range: tuple. Minimum and maximum children's age
                * max_age_diff: integer. Maximum allowed age difference between parent and child
            Output:
                * Assigns home and job/school to all agents grouped by families of four members.
                    Assigns home and job to lonely agents
        """

        # set available agents' ids
        av_ags_ids = self.model.set_available_ids
        for clust_idx, clust_info in self.clusters_info.items():
            for idx, family_langs in enumerate(zip(*[iter(self.langs_per_clust[clust_idx])] * self.model.family_size)):
                family_agents = self.create_new_family(family_langs, parents_age_range,
                                                       children_age_range, max_age_diff)
                # assign same home to all family members to locate them at step = 0
                home = clust_info['homes'][idx]
                home.assign_to_agent(family_agents)
                # add agents to clust_info, schedule and grid, but not yet to networks
                clust_info['agents'].extend(family_agents)
                self.add_agents_to_grid_and_schedule(family_agents)

                # assign job in current cluster to parents, without moving to a new home
                for parent in family_agents[:2]:
                    parent.get_job(keep_cluster=True, move_home=False)

                # assign school to children ( but not yet course !!!!)
                # find closest school to home
                # TODO : introduce also University for age > 18
                home = clust_info['homes'][idx]
                idx_school = np.argmin([pdist([home.pos, school.pos])
                                        for school in clust_info['schools']])
                school = clust_info['schools'][idx_school]
                for child in family_agents[2:]:
                    child.loc_info['school'] = [school, None]
                    school.info['students'].add(child)

            # set up agents left out of family partition of cluster
            len_clust = clust_info['num_agents']
            num_left_agents = len_clust % self.model.family_size
            if num_left_agents:
                for idx2, lang in enumerate(self.langs_per_clust[clust_idx][-num_left_agents:]):
                    sex = 'M' if random.random() < 0.5 else 'F'
                    ag = Adult(self.model, av_ags_ids.pop(), lang, sex)
                    ag.info['age'] = self.model.steps_per_year * np.random.randint(parents_age_range[0],
                                                                                   parents_age_range[1])
                    clust_info['agents'].append(ag)
                    self.add_agents_to_grid_and_schedule(ag)
                    # assign home
                    home = clust_info['homes'][idx + idx2 + 1]
                    home.assign_to_agent(ag)
                    # assign job
                    ag.get_job(keep_cluster=True, move_home=False)

    def assign_school_jobs(self):
        """
            Method to set up all courses in all schools at initialization.
            It calls 'set_up_courses' school method for each school,
            that hires teachers for all courses after grouping students
            by age
        """
        # Loop over all model schools to assign teachers
        for clust_idx, clust_info in self.clusters_info.items():
            for school in clust_info['schools']:
                school.set_up_courses()

        # check for model inconsistencies in school assignment
        # check if there is a teacher for each created course
        error_message = ("""MODELING ERROR: not all school courses have a
                            teacher assigned. The specified school lang policy
                            cannot be met by current population language knowledge.""")
        try:
            teachers_per_course = [course['teacher'] if 'teacher' in course else False
                                   for cl in range(self.num_clusters)
                                   for sch in self.clusters_info[cl]['schools']
                                   for ck, course in sch.grouped_studs.items()]
            assert all(teachers_per_course)
        except AssertionError:
            raise Exception(error_message)

    def add_agents_to_grid_and_schedule(self, ags):
        """ Method to add a number of agents to grid, schedule and system networks
            Arguments:
                * ags : a single agent class instance or a list of class instances. In either case,
                    ags will be transformed into an iterable
        """
        # construct an iterable list out of all possible input structure
        ags = [ags] if not isinstance(ags, list) else ags
        # add agent to grid and schedule
        for ag in ags:
            self.model.schedule.add(ag)
            if ag.loc_info['home']:
                self.model.grid.place_agent(ag, ag.loc_info['home'].pos)
            else:
                self.model.grid.place_agent(ag, (0, 0))

    def update_agent_clust_info(self, agent, curr_clust, update_type='remove',
                                new_clust=None, grown_agent=None):
        """ Update agent reference in clusters info """

        self.clusters_info[curr_clust]['agents'].remove(agent)
        if update_type == 'replace':
            if grown_agent:
                self.clusters_info[curr_clust]['agents'].append(grown_agent)
            else:
                raise Exception('grown_agent must be specified for replace option')
        elif update_type == 'switch': # need curr cluster, new cluster
            if new_clust is not None:
                self.clusters_info[new_clust]['agents'].append(agent)
            else:
                raise Exception('new cluster must be specified for switch option')

    def get_clusters_with_univ(self):
        """ Convenience method to get ids of clusters with university """
        sorted_clusts = np.argsort(self.cluster_sizes)[::-1]
        for ix, clust in enumerate(sorted_clusts):
            if 'university' not in self.clusters_info[clust]:
                ix_max_univ = ix
                break
        return sorted_clusts[:ix_max_univ]

    def get_current_clust_size(self, clust_ix):
        if self.clusters_info[clust_ix]['agents']:
            curr_clust_size = len(self.clusters_info[clust_ix]['agents'])
        else:
            curr_clust_size = self.cluster_sizes[clust_ix]
        return curr_clust_size

    def get_lang_distrib_per_clust(self, clust_ix):
        """ Method to find language percentages for each language cathegory
            in a given cluster 0-> mono L1, 1-> biling, 2-> mono L2
            Args:
                * clust_ix: integer. Identifies cluster by its index in model
            Output:
                * numpy array where indices are lang cathegories and values are percentages
        """
        if self.clusters_info[clust_ix]['agents']:
            clust_lang_cts = Counter([ag.info['lang_type'] for ag
                                      in self.clusters_info[clust_ix]['agents']])
        else:
            clust_lang_cts = Counter(self.langs_per_clust[clust_ix])
        clust_size = self.get_current_clust_size(clust_ix)
        return np.array([clust_lang_cts[lang] / clust_size for lang in range(3)])

    def get_dominant_lang_per_clust(self, clust_ix):
        """
            Method to compute the dominant language per cluster
            as measured by the average percentage knowledge of
            cluster inhabitants.
            Args:
                * clust_ix: integer. Identifies cluster by its index in model
            Output:
                * integer that identifies dominant language or None if no language is dominant
        """

        L1_pcts, L2_pcts = list(zip(*[[ag.lang_stats['L1']['pct'][ag.info['age']],
                                       ag.lang_stats['L2']['pct'][ag.info['age']]]
                                       for ag in self.clusters_info[clust_ix]['agents']
                                     ]))
        L1_pct, L2_pct = np.array(L1_pcts).mean(), np.array(L2_pcts).mean()

        # For a lang to be dominant, its quality must be on average 10% higher than the other's

        if L1_pct >= L2_pct + 0.1:
            return 0
        elif L2_pct >= L1_pct + 0.1:
            return 1
        else:
            return None




