# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mesa.time import StagedActivation
from agent import IndepAgent, Young

import numpy as np
import networkx as nx
import random


class StagedActivationModif(StagedActivation):
    # TODO : add/separate agents by type ??? Is it a good idea ??

    def step(self):
        """ Executes all the stages for all agents. """
        for ag in self.agents[:]:
            ag.info['age'] += 1
            for lang in ['L1', 'L12', 'L21', 'L2']:
                # update last-time word use vector
                ag.lang_stats[lang]['t'][~ag.day_mask[lang]] += 1
                # set current lang knowledge
                # compute current language knowledge in percentage after 't' update
                pct_threshold = 0.9
                if lang in ['L1', 'L12']:
                    real_lang_knowledge = np.maximum(ag.lang_stats['L1']['R'], ag.lang_stats['L12']['R'])
                    ag.lang_stats['L1']['pct'][ag.info['age']] = (np.where(real_lang_knowledge > pct_threshold)[0].shape[0] /
                                                                        ag.model.vocab_red)
                else:
                    real_lang_knowledge = np.maximum(ag.lang_stats['L2']['R'], ag.lang_stats['L21']['R'])
                    ag.lang_stats['L2']['pct'][ag.info['age']] = (np.where(real_lang_knowledge > pct_threshold)[0].shape[0] /
                                                                        ag.model.vocab_red)
                # reset day mask
                ag.day_mask[lang] = np.zeros(ag.model.vocab_red, dtype=np.bool)
            # Update lang switch
            ag.update_lang_switch()
        if self.shuffle:
            random.shuffle(self.agents)
        # basic IDEA: network adj matrices will be fixed through all stages of one step
        # compute adjacent matrices for family and friends
        Fam_Graph = nx.adjacency_matrix(self.model.nws.family_network,
                                        nodelist=self.agents).toarray()
        self.model.nws.adj_mat_fam_nw = Fam_Graph / Fam_Graph.sum(axis=1, keepdims=True)

        Friend_Graph = nx.adjacency_matrix(self.model.nws.friendship_network,
                                           nodelist=self.agents).toarray()
        self.model.nws.adj_mat_friend_nw = Friend_Graph / Friend_Graph.sum(axis=1, keepdims=True)

        for stage in self.stage_list:
            for ix_ag, ag in enumerate(self.agents):
                # Run stage
                if isinstance(ag, IndepAgent):
                    getattr(ag, stage)(ix_ag)
                else:
                    getattr(ag, stage)()
            if self.shuffle_between_stages:
                random.shuffle(self.agents)
            self.time += self.stage_time
        # check reproduction, death : make shallow copy of agents list,
        # since we are potentially removing agents as we iterate
        for ag in self.agents[:]:
            if isinstance(ag, Young):
                ag.reproduce()
            ag.random_death()
        # loop and update courses in schools and universities year after year
        if not self.steps % 36:
            for clust_idx, clust_info in self.model.geo.clusters_info.items():
                if 'university' in clust_info:
                    for fac in clust_info['university'].faculties.values():
                        if fac.info['students']:
                            fac.update_courses()
                for school in clust_info['schools']:
                    school.update_courses()
                    if not self.steps % 72: # every 2 years only, teachers swap
                        school.swap_teachers_courses()
        self.steps += 1
