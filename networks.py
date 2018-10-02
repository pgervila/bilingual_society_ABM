import random
import numpy as np
import networkx as nx

from sklearn.preprocessing import normalize
import warnings
from sklearn.utils import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)

from agent import Child, Young, Adult, Teacher


class NetworkBuilder:
    """ Class to deal with everything that has to do with networks in model """

    family_mirror = {'consort': 'consort', 'father': 'child', 'mother': 'child',
                     'grandfather': 'grandchild', 'grandmother': 'grandchild',
                     'uncle': 'nephew', 'aunt': 'nephew', 'sibling': 'sibling',
                     'cousin': 'cousin'}

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
        # INITIALIZE FAMILY NETWORK
        self.family_network = nx.DiGraph()
        # INITIALIZE JOBS NETWORK
        self.jobs_network = nx.Graph()

    def add_ags_to_networks(self, ags, *more_ags):
        """
            Method that adds agents to all networks in model
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
        self.define_jobs_network()

    def define_family_networks(self):
        """
            Method to define family links between agents. It also adds relatives to known_people_network
            It assumes that the distribution of languages in clusters has already been sorted at cluster level or
            by groups of four to ensure linguistic affinity within families.
            Marriage, to make things simple, only allowed for linguistic combinations ->  0-1, 1-1, 1-2
            Once all family links are defined, method also assigns school jobs to parents so that entire family
            will move if needed
        """
        for clust_idx, clust_info in self.model.geo.clusters_info.items():
            # trick to iterate over groups of agents of size = self.model.family_size
            for idx, family in enumerate(zip(*[iter(clust_info['agents'])] * self.model.family_size)):
                # set linguistic ICs to family
                self.model.set_lang_ics_in_family(family)
                # find out lang of interaction btw family members
                (lang_consorts, lang_with_father,
                 lang_with_mother, lang_siblings) = self.model.get_lang_fam_members(family)
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
                for idx2, ag in enumerate(clust_info['agents'][-num_left_agents:]):
                    ag.set_lang_ics()
        # assign school jobs
        self.model.geo.assign_school_jobs()

    def define_friendship_networks(self):
        """ """
        # TODO : Apply small world graph to relevant nodes using networkx
        # TODO : change distribution to allow for extreme number of friends for a few agents
        friends_per_agent = np.random.randint(1, 5, size=self.model.num_people)
        for ag, num_friends in zip(self.model.schedule.agents, friends_per_agent):
            if type(ag) == Adult and len(self.friendship_network[ag]) < num_friends:
                if ag.loc_info['job']:
                    info_occupation = ag.loc_info['job'].info
                else:
                    clust = ag['clust']
                    info_occupation = random.choice(self.model.geo.clusters_info[clust]['jobs']).info
                colleagues = 'employees'
            elif isinstance(ag, Teacher):
                if ag.loc_info['job'][0]:
                    info_occupation = ag.loc_info['job'][0].info
                colleagues = 'employees'
            elif isinstance(ag, Child) and len(self.friendship_network[ag]) < num_friends:
                school, course_key = ag.loc_info['school']
                info_occupation = school.grouped_studs[course_key]
                colleagues = 'students'
            else:
                continue
            for coll in info_occupation[colleagues].difference({ag}):
                # check colleague lang distance and all friendship conditions
                if ag.check_friend_conds(coll, num_friends):
                    ag.make_friend(coll)
                if len(self.friendship_network[ag]) == num_friends:
                    break

    def define_jobs_network(self, p=0.3):
        """ Defines random graph with probability p where nodes are job instances"""
        jobs = [j for cl in range(self.model.geo.num_clusters)
                for j in self.model.geo.clusters_info[cl]['jobs']]
        self.jobs_network = nx.gnp_random_graph(len(jobs), p)
        nodes_mapping = {i: j for i, j in enumerate(jobs)}
        self.jobs_network = nx.relabel_nodes(self.jobs_network, nodes_mapping)

    def set_family_links(self, agent, father, mother, lang_with_father, lang_with_mother):
        """
            Method to define family links and interaction language of a newborn agent.
            Corresponding edges are created in family and known people networks
            Args:
                * agent: agent instance. The agent on which links have to be defined
                * father: agent instance. The agent's father
                * mother: agent instance. The agent's mother
                * lang_with_father: integer [0, 1]. Defines lang of agent with father
                * lang_with_mother: integer [0, 1]. Defines lang of agent with mother
        """
        fam_nw = self.family_network
        kn_people_nw = self.known_people_network

        # get agent siblings from mother BEFORE agent birth
        siblings = mother.get_family_relative('child')
        # Define family links with parents

        for (i, j, fam_link) in [(agent, father, 'father'), (father, agent, 'child')]:
            fam_nw.add_edge(i, j, fam_link=fam_link, lang=lang_with_father)
            kn_people_nw.add_edge(i, j, family=True, lang=lang_with_father)
        for (i, j, fam_link) in [(agent, mother, 'mother'), (mother, agent, 'child')]:
            fam_nw.add_edge(i, j, fam_link=fam_link, lang=lang_with_mother)
            kn_people_nw.add_edge(i, j, family=True, lang=lang_with_mother)

        # self.set_link_with_relatives(agent, (father, mother), ('father', 'mother'),
        #                              lang_with_relatives=(lang_with_father, lang_with_mother))

        # Define links with agent siblings if any
        for sibling in siblings:
            lang_with_sibling = sibling.get_dominant_lang()
            for (i, j, fam_link) in [(agent, sibling, 'sibling'), (sibling, agent, 'sibling')]:
                fam_nw.add_edge(i, j, fam_link=fam_link, lang=lang_with_sibling)
                kn_people_nw.add_edge(i, j, family=True, lang=lang_with_sibling)

        # langs_with_siblings = [sibling.get_dominant_lang() for sibling in siblings]
        # self.set_link_with_relatives(agent, siblings, 'sibling', lang_with_relatives=langs_with_siblings)

        # rest of family will if possible speak same language with baby as their link agent to the baby
        for elder, lang in zip([father, mother], [lang_with_father, lang_with_mother]):
            for relat, labels in fam_nw[elder].items():
                lang_label = 'L1' if lang == 0 else 'L2'
                if relat.lang_stats[lang_label]['pct'][relat.info['age']] > relat.lang_thresholds['speak']:
                    com_lang = lang
                else:
                    com_lang = 1 if lang == 0 else 0
                if labels["fam_link"] == 'father':

                    #self.set_link_with_relatives(relat, agent, 'grandfather', lang_with_relatives=com_lang)

                    for (i, j, fam_link) in [(agent, relat, 'grandfather'), (relat, agent, 'grandchild')]:
                        fam_nw.add_edge(i, j, fam_link=fam_link, lang=com_lang)
                        kn_people_nw.add_edge(i, j, family=True, lang=com_lang)
                elif labels["fam_link"] == 'mother':
                    self.set_link_with_relatives(relat, agent, 'grandmother', lang_with_relatives=com_lang)


                    for (i, j, fam_link) in [(agent, relat, 'grandmother'), (relat, agent, 'grandchild')]:
                        fam_nw.add_edge(i, j, fam_link=fam_link, lang=com_lang)
                        kn_people_nw.add_edge(i, j, family=True, lang=com_lang)
                elif labels["fam_link"] == 'sibling':
                    fam_nw.add_edge(agent, relat,
                                    fam_link='uncle' if relat.info['sex'] == 'M' else 'aunt',
                                    lang=com_lang)
                    fam_nw.add_edge(relat, agent, fam_link='nephew', lang=com_lang)
                    for (i, j) in [(agent, relat), (relat, agent)]:
                        kn_people_nw.add_edge(i, j, family=True, lang=com_lang)
                    if isinstance(relat, Young) and relat.info['married']:
                        consort = [key for key, value in fam_nw[relat].items()
                                   if value['fam_link'] == 'consort'][0]
                        fam_nw.add_edge(agent, consort,
                                        fam_link ='uncle' if consort.info['sex'] == 'M' else 'aunt')
                        fam_nw.add_edge(consort, agent, fam_link='nephew', lang=com_lang)
                        for (i, j) in [(agent, consort), (consort, agent)]:
                            kn_people_nw.add_edge(i, j, family=True, lang=com_lang)
                elif labels["fam_link"] == 'nephew':
                    fam_nw.add_edge(agent, relat, fam_link='cousin', lang=com_lang)
                    fam_nw.add_edge(relat, agent, fam_link='cousin', lang=com_lang)
                    for (i, j) in [(agent, relat), (relat, agent)]:
                        kn_people_nw.add_edge(i, j, family=True, lang=com_lang)

    def set_link_with_relatives(self, agent, relatives, labels, lang_with_relatives=None):
        """ Method to set network links with relatives
            Args:
                * agent: class instance
                * relatives: tuple of class instances.
                * labels: string or tuple of strings. Must have same length as 'relatives'
                * lang_with_relatives: tuple of integers. Optional. Default None
            Output:
                *
        """
        fam_nw = self.family_network
        kn_people_nw = self.known_people_network
        if not isinstance(relatives, (tuple, list)):
            relatives = [relatives]
        if isinstance(labels, str):
            labels = (labels, ) * len(relatives)
        if not lang_with_relatives:
            lang_with_relatives = [relative.get_dominant_lang() for relative in relatives]
        for relative, fam_link, lang in zip(relatives, labels, lang_with_relatives):
            for (i, j, link) in [(agent, relative, fam_link),
                                 (relative, agent, self.family_mirror[fam_link])]:
                fam_nw.add_edge(i, j, fam_link=link, lang=lang)
                kn_people_nw.add_edge(i, j, family=True, lang=lang)

    def compute_adj_matrices(self):
        """
            Method to compute adjacent matrices for family and friends
            Output:
                * Method sets value of adj_mat_fam_nw and adj_mat_friend_nw NetworkBuilder
                    class attributes
        """
        fam_graph = nx.adjacency_matrix(self.model.nws.family_network,
                                        nodelist=self.model.schedule.agents)
        self.model.nws.adj_mat_fam_nw = normalize(fam_graph, norm='l1', axis=1)
        # compute adjacent matrices for friends
        friend_graph = nx.adjacency_matrix(self.model.nws.friendship_network,
                                           nodelist=self.model.schedule.agents)
        self.model.nws.adj_mat_friend_nw = normalize(friend_graph, norm='l1', axis=1)

    def plot_family_networks(self):
        """ PLOT NETWORK with lang colors and position """
        #        people_pos = [elem.pos for elem in self.schedule.agents if elem.agent_type == 'language']
        #        # Following works because lang agents are added before other agent types
        #        people_pos_dict= dict(zip(self.schedule.agents, people_pos))
        people_pos_dict = {elem: elem.pos
                           for elem in self.model.schedule.agents}
        people_color = [elem.language for elem in self.family_network]
        nx.draw(self.family_network, pos=people_pos_dict, node_color=people_color)

    def plot_jobs_network(self):
        jobs_pos = {j: j.pos for cl in range(self.model.geo.num_clusters)
                    for j in self.model.geo.clusters_info[cl]['jobs']}
        nx.draw(self.jobs_network, pos=jobs_pos)

    def __getstate__(self):
        print('pickling NWS')
        state = self.__dict__.copy()
        for nw in ['known_people_network', 'family_network', 'friendship_network', 'jobs_network']:
            state[nw] = [tuple(state[nw].nodes()), tuple(state[nw].edges(data=True))]
        return state

    def __setstate__(self, state):
        print('unpickling NWS in')
        self.__dict__.update(state)
        self.known_people_network = nx.DiGraph()
        self.known_people_network.add_nodes_from(state['known_people_network'][0])
        self.known_people_network.add_edges_from(state['known_people_network'][1])
        self.family_network = nx.Graph()
        self.family_network.add_nodes_from(state['family_network'][0])
        self.family_network.add_edges_from(state['family_network'][1])
        self.friendship_network = nx.DiGraph()
        self.friendship_network.add_nodes_from(state['friendship_network'][0])
        self.friendship_network.add_edges_from(state['friendship_network'][1])
        self.jobs_network = nx.Graph()
        self.jobs_network.add_nodes_from(state['jobs_network'][0])
        self.jobs_network.add_edges_from(state['jobs_network'][1])
        print('unpickling NWS out')



