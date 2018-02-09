import random
import numpy as np
from scipy.spatial.distance import pdist
import networkx as nx
import bisect


class NetworkBuilder:
    """ Class to deal with everything that has to do with networks in model"""

    def __init__(self, model):

        self.adj_mat_fam_nw = None
        self.adj_mat_friend_nw = None

        self.model = model
        # setup
        self.create_networks()
        self.add_ags_to_networks(self.model.schedule.agents)


    def create_networks(self):
        # INITIALIZE KNOWN PEOPLE NETWORK => label is lang spoken
        self.known_people_network = nx.DiGraph()
        # INITIALIZE FRIENDSHIP NETWORK
        self.friendship_network = nx.Graph()
        # sort by friendship intensity
        #       sorted(self.friendship_network[n_1].items(),
        #              key=lambda edge: edge[1]['link_strength'],
        #              reverse = True)
        # INITIALIZE FAMILY NETWORK
        self.family_network = nx.DiGraph()

    def add_ags_to_networks(self, ags, *more_ags):
        """Method that adds agents to all networks in model
           Args:
               * ags: single agent instance or list of agents instances
        """
        ags = [ags] if type(ags) != list else ags
        for nw in [self.known_people_network,
                   self.friendship_network,
                   self.family_network]:
            nw.add_nodes_from(ags)

    def build_networks(self):
        self.define_family_networks()
        self.define_friendship_networks()

    def get_lang_fam_members(self, family):
        """ Find out lang of interaction btw family members in a 4-members family
            Args:
                * family: list of family agents
            Output:
                * lang_consorts, lang_with_father, lang_with_mother, lang_siblings: list of integers
        """
        # language between consorts
        consorts_lang_params = self.model.get_conv_params([family[0], family[1]])
        lang_consorts = consorts_lang_params['lang_group']

        # language of children with parents
        lang_with_father = consorts_lang_params['fav_langs'][0]
        lang_with_mother = consorts_lang_params['fav_langs'][1]

        # siblings
        avg_lang = (lang_with_father + lang_with_mother) / 2
        if avg_lang == 0:
            lang_siblings = 0
        elif avg_lang == 1:
            lang_siblings = 1
        else:
            siblings_lang_params = self.model.get_conv_params([family[2], family[3]])
            lang_siblings = siblings_lang_params['lang_group']

        return lang_consorts, lang_with_father, lang_with_mother, lang_siblings

    def define_family_networks(self, parents_age_range=(32, 42), children_age_range=(2, 11)):
        """
            Method to define family links between agents. It also adds relatives to known_people_network
            It assumes distribution of languages in clusters has already been sorted at cluster level or
            by groups of four to ensure linguistic affinity within families.
            Marriage, to make things simple, only allowed for combinations  0-1, 1-1, 1-2
        """
        for clust_idx, clust_info in self.model.geo.clusters_info.items():
            for idx, family in enumerate(zip(*[iter(clust_info['agents'])] * self.model.family_size)):
                # set ages of family members
                min_age, max_age = parents_age_range
                (family[0].info['age'],
                 family[1].info['age']) = np.random.randint(min_age, max_age, 2) * self.model.steps_per_year
                min_age, max_age = children_age_range
                (family[2].info['age'],
                 family[3].info['age']) = np.random.randint(min_age, max_age, 2) * self.model.steps_per_year
                # assign same home to all family members
                home = clust_info['homes'][idx]
                # import ICs and assign home
                # apply correlation between parents' and children's lang knowledge if parents bilinguals
                if 1 in [m.info['language'] for m in family[:2]]:
                    key_parents = [] # define list to store parents' percentage knowledge
                    for ix, member in enumerate(family):
                        if ix < 2 and member.info['language'] == 1:
                            key = np.random.choice(self.model.ic_pct_keys)
                            key_parents.append(key)
                            member.set_lang_ics(biling_key=key)
                        elif ix < 2:
                            lang_mono = member.info['language']
                            member.set_lang_ics()
                        elif ix >= 2:
                            if len(key_parents) == 1:
                                if not lang_mono: # mono in lang 0
                                    key = (key_parents[0] + 100) / 2
                                else: # mono in lang 1
                                    key = key_parents[0] / 2
                            else:
                                key = sum(key_parents) / len(key_parents)
                            key = self.model.ic_pct_keys[
                                bisect.bisect_left(self.model.ic_pct_keys, key,
                                                   hi=len(self.model.ic_pct_keys) - 1)
                            ]
                            member.set_lang_ics(biling_key=key)
                        home.assign_to_agent(member)
                else: # monolingual parents
                    # check if children are bilingual
                    if 1 in [m.info['language'] for m in family[2:4]]:
                        for ix, member in enumerate(family):
                            if ix < 2:
                                member.set_lang_ics()
                            else:
                                if member.info['language'] == 1:
                                    # logical that child has much better knowledge of parents lang
                                    member.set_lang_ics(biling_key=90)
                                else:
                                    member.set_lang_ics()
                            home.assign_to_agent(member)
                    else:
                        for member in family:
                            member.set_lang_ics()
                            home.assign_to_agent(member)

                # assign job to parents
                for parent in family[:2]:
                    while True:
                        job = np.random.choice(clust_info['jobs'])
                        if job.num_places:
                            job.num_places -= 1
                            parent.loc_info['job'] = job
                            job.info['employees'].add(parent)
                            break

                # assign school to children
                # find closest school
                # TODO : introduce also University for age > 18
                idx_school = np.argmin([pdist([home.pos, school.pos])
                                        for school in clust_info['schools']])
                school = clust_info['schools'][idx_school]
                for child in family[2:]:
                    child.loc_info['school'] = school
                    school.info['students'].add(child)

                # find out lang of interaction btw family members
                (lang_consorts, lang_with_father,
                 lang_with_mother, lang_siblings) = self.get_lang_fam_members(family)

                # initialize family network
                # add family edges in family and known_people networks ( both are DIRECTED networks ! )
                for (i, j) in [(0, 1), (1, 0)]:
                    self.family_network.add_edge(family[i], family[j], fam_link='consort', lang=lang_consorts)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_consorts)
                for (i, j, link) in [(0, 2, 'child'), (2, 0, 'father'), (0, 3, 'child'), (3, 0, 'father')]:
                    self.family_network.add_edge(family[i], family[j], fam_link=link, lang=lang_with_father)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_with_father)
                for (i, j, link) in [(1, 2, 'child'), (2, 1, 'mother'), (1, 3, 'child'), (3, 1,'mother')]:
                    self.family_network.add_edge(family[i], family[j], fam_link=link, lang=lang_with_mother)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_with_mother)
                for (i, j) in [(2, 3), (3, 2)]:
                    self.family_network.add_edge(family[i], family[j], fam_link='sibling', lang=lang_siblings)
                    self.known_people_network.add_edge(family[i], family[j], family=True, lang=lang_siblings)

            # set up agents left out of family partition of cluster
            len_clust = len(clust_info['agents'])
            num_left_agents = len_clust % self.model.family_size
            if num_left_agents:
                for ag in clust_info['agents'][-num_left_agents:]:
                    min_age, max_age = 40 * self.model.steps_per_year, 60 * self.model.steps_per_year
                    ag.info['age']= np.random.randint(min_age, max_age)
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
        # assign school jobs
        # Loop over schools to assign teachers
        for clust_idx, clust_info in self.model.geo.clusters_info.items():
            for school in clust_info['schools']:
                school.set_up_courses()

    def define_friendship_networks(self):
        # TODO : Apply small world graph to relevant nodes using networkx
        friends_per_agent = np.random.randint(1, 5, size=self.model.num_people)
        for ag, num_friends in zip(self.model.schedule.agents, friends_per_agent):
            if 'job' in ag.loc_info and len(self.friendship_network[ag]) < num_friends:
                ag_occupation = ag.loc_info['job']
                colleagues = 'employees'
            elif 'school' in ag.loc_info and len(self.friendship_network[ag]) < num_friends:
                ag_occupation = ag.loc_info['school']
                colleagues = 'students'
            else:
                continue
            for coll in getattr(ag_occupation, 'info')[colleagues].difference({ag}):
                # check colleague lang distance and all frienship conditions
                if (abs(coll.info['language'] - ag.info['language']) <= 1 and
                            len(self.friendship_network[coll]) < friends_per_agent[coll.unique_id] and
                            coll not in self.friendship_network[ag] and
                            coll not in self.family_network[ag]):
                    friends = [ag, coll]
                    # who speaks first may determine communication lang
                    random.shuffle(friends)
                    lang = self.model.get_conv_params(friends)['lang_group']
                    self.friendship_network.add_edge(ag, coll, lang=lang, weight=np.random.randint(1, 10))
                    # known people network is directed graph !
                    self.known_people_network.add_edge(ag, coll, friends=True, lang=lang)
                    self.known_people_network.add_edge(coll, ag, friends=True, lang=lang)
                if len(self.friendship_network[ag]) > num_friends - 1:
                    break

    def plot_family_networks(self):
        """PLOT NETWORK with lang colors and position"""
        #        people_pos = [elem.pos for elem in self.schedule.agents if elem.agent_type == 'language']
        #        # Following works because lang agents are added before other agent types
        #        people_pos_dict= dict(zip(self.schedule.agents, people_pos))
        people_pos_dict = {elem: elem.pos
                           for elem in self.model.schedule.agents}
        people_color = [elem.language for elem in self.family_network]
        nx.draw(self.family_network, pos=people_pos_dict, node_color=people_color)




