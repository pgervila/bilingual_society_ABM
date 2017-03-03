# IMPORT LIBS
import random
import numpy as np
import networkx as nx
from collections import deque

class Simple_Language_Agent:

    def __init__(self, model, unique_id, language, S, age=0,
                 home_coords=None, school_coords=None, job_coords=None):
        self.model = model
        self.unique_id = unique_id
        self.language = language # 0, 1, 2 => spa, bil, cat
        self.S = S
        self.age = age
        self.home_coords = home_coords
        self.school_coords = school_coords
        self.job_coords = job_coords

        self.lang_freq = dict()
        num_init_occur = 50
        if self.language == 0:
            self.lang_freq['spoken'] = [np.random.poisson(num_init_occur), 0] # 0, 2 => spa, cat
            self.lang_freq['heard'] = [np.random.poisson(num_init_occur), 0]
            self.lang_freq['cat_pct_s'] = 0
            self.lang_freq['cat_pct_h'] = 0
        elif self.language == 2:
            self.lang_freq['spoken'] = [0, np.random.poisson(num_init_occur)] # 0, 2 => spa, cat
            self.lang_freq['heard'] = [0, np.random.poisson(num_init_occur)]
            self.lang_freq['cat_pct_s'] = 1
            self.lang_freq['cat_pct_h'] = 1
        else:
            v1 = np.random.poisson(num_init_occur/2)
            v2 = np.random.poisson(num_init_occur/2)
            self.lang_freq['spoken'] = [v1, v2] # 0, 2 => spa, cat
            self.lang_freq['heard'] = [v1, v2]
            self.lang_freq['cat_pct_s'] = self.lang_freq['spoken'][1]/sum(self.lang_freq['spoken'])
            self.lang_freq['cat_pct_h'] = self.lang_freq['heard'][1]/sum(self.lang_freq['heard'])
        # initialize maxmem deque based on language spoken last maxmem lang encounters
        self.lang_freq['maxmem'] = np.random.poisson(self.model.avg_max_mem)
        self.lang_freq['maxmem_list'] = deque(maxlen=self.lang_freq['maxmem'])


    def move_random(self):
        """ Take a random step into any surrounding cell
            All eight surrounding cells are available as choices
            Current cell is not an output choice

            Returns:
                * modifies self.pos attribute
        """
        x, y = self.pos  # attr pos is defined when adding agent to schedule
        possible_steps = self.model.grid.get_neighborhood(
            (x, y),
            moore=True,
            include_center=False
        )
        chosen_cell = random.choice(possible_steps)
        self.model.grid.move_agent(self, chosen_cell)

    def speak(self, with_agent=None):
        """ Pick random lang_agent from current cell and start a conversation
            with it. It updates heard words in order to shape future vocab.
            Language of the conversation is determined by given laws,
            including probabilistic ones based on parameter self.S
            This method can also simulate distance contact e.g.
            phone, messaging, etc ... by specifying an agent through 'with_agent'

            Arguments:
                * with_agent : specify a specific agent with which conversation will take place
                  By default the agent will be picked randomly from all lang agents in current cell

            Returns:
                * Defines conversation and language(s) in which it takes place.
                  Updates heard/used stats
        """
        if not with_agent:
            pos = [self.pos]
            # get all other agents currently placed on chosen cell
            others = self.model.grid.get_cell_list_contents(pos)
            others.remove(self)
            ## linguistic model of encounter with another random agent
            if len(others) >= 1:
                other = random.choice(others)
                self.get_conversation_lang(self, other)
                # update lang status
                self.update_lang_status()
                other.update_lang_status()
        else:
            self.get_conversation_lang(self, with_agent)
            other = with_agent
            # update lang status
            self.update_lang_status()
            other.update_lang_status()

    def speak_in_group(self, first_speaker=True, group=None, group_max_size=5):
        """ Determine language spoken when a group meets
            Group is split between initiator and rest_of_group
        """

        if group:
            group_set = set(group)
        else:
            ags_in_cell = self.model.grid.get_cell_list_contents([self.pos])
            num_ags_in_cell = len(ags_in_cell)
            if num_ags_in_cell >= 3:
                if num_ags_in_cell >= group_max_size:
                    group_size = np.random.randint(3, group_max_size + 1)
                    group = list(np.random.choice(ags_in_cell, replace=False, size=group_size))
                    if self not in group:
                        group = [self] + group[:-1]
                else:
                    group_size = np.random.randint(3, num_ags_in_cell + 1)
                    group = np.random.choice(ags_in_cell, replace=False, size=group_size)
                group_set = set(group)
            else:
                group_set = None
        if group_set:
            if not first_speaker:
                initiator = np.random.choice(group_set.difference({self}))
                rest_of_group = list(group_set.difference({initiator}))
                self.get_group_conversation_lang(initiator, rest_of_group )
            else:
                rest_of_group = list(group_set.difference({self}))
                self.get_group_conversation_lang(self, rest_of_group)


    def listen(self):
        """Listen to random agents placed on the same cell as calling agent"""
        pos = [self.pos]
        # get all agents currently placed on chosen cell
        others = self.model.grid.get_cell_list_contents(pos)
        others.remove(self)
        ## linguistic model of encounter with another random agent
        if len(others) >= 2:
            ag_1, ag_2 = np.random.choice(others, size=2, replace=False)
            l1, l2 = self.get_conversation_lang(ag_1, ag_2, return_langs=True)
            self.lang_freq['heard'][l1] += 1
            self.lang_freq['heard'][l2] += 1
            # update lang status
            ag_1.update_lang_status()
            ag_2.update_lang_status()

    def update_lang_counter(self, ags_list, langs_list):
        for ag_idx, ag in enumerate(ags_list):
            for lang_idx, lang in enumerate(langs_list):
                if lang_idx != ag_idx:
                    ag.lang_freq['heard'][lang] += 1
                else:
                    ag.lang_freq['spoken'][lang] += 1


    def get_conversation_lang(self, ag_1, ag_2, return_langs=False):

        if (ag_1.language, ag_2.language) in [(0, 0), (0, 1), (1, 0)]:# spa-bilingual
            l1 = l2 = 0
            self.update_lang_counter([ag_1, ag_2], [l1, l2])
            ag_1.lang_freq['maxmem_list'].append(0)
            ag_2.lang_freq['maxmem_list'].append(0)

        elif (ag_1.language, ag_2.language) in [(2, 1), (1, 2), (2, 2)]:# bilingual-cat
            l1=l2=1
            self.update_lang_counter([ag_1, ag_2], [l1, l2])
            ag_1.lang_freq['maxmem_list'].append(1)
            ag_2.lang_freq['maxmem_list'].append(1)

        elif (ag_1.language, ag_2.language) == (1, 1): # bilingual-bilingual
            # TODO : NEED TO IMPROVE THIS CASE !! ( Avoid two bad speakers both choose weak lang)
            p11 = (0.5 * (ag_1.lang_freq['cat_pct_s']) +
                   0.5 * (ag_1.lang_freq['cat_pct_h']))
            # find out lang spoken by self ( self STARTS CONVERSATION !!)
            if sum(ag_1.lang_freq['spoken']) != 0:
                l1 = np.random.binomial(1, p11)
            else:
                l1 = random.choice([0,1])
            l2=l1
            self.update_lang_counter([ag_1, ag_2], [l1, l2])
            ag_1.lang_freq['maxmem_list'].append(l1)
            ag_2.lang_freq['maxmem_list'].append(l1)

        else: # spa-cat
            p11 = (0.5 * (ag_1.lang_freq['cat_pct_s']) +
                   0.5 * (ag_1.lang_freq['cat_pct_h']))
            p21 = (0.5 * (ag_2.lang_freq['cat_pct_s']) +
                   0.5 * (ag_2.lang_freq['cat_pct_h']))
            if ag_1.language == 0:
                l1 = 0
                if (1 - ag_2.lang_freq['cat_pct_s']) or (1 - ag_2.lang_freq['cat_pct_h']):
                    l2 = np.random.binomial(1, p21)
                    if l2 == 0:
                        self.update_lang_counter([ag_1, ag_2], [l1, l2])
                    elif l2 == 1:
                        l1 = np.random.binomial(1, p11)
                        self.update_lang_counter([ag_1, ag_2], [l1, l2])
                else:
                    l1 = np.random.binomial(1, p11)
                    l2 = 1
                    self.update_lang_counter([ag_1, ag_2], [l1, l2])

            elif ag_1.language == 2:
                l1 = 1
                if (ag_2.lang_freq['cat_pct_s']) or (ag_2.lang_freq['cat_pct_h']):
                    l2 = np.random.binomial(1, p21)
                    if l2 == 1:
                        self.update_lang_counter([ag_1, ag_2], [l1, l2])
                    elif l2 == 0:
                        l1 = np.random.binomial(1, p11)
                        self.update_lang_counter([ag_1, ag_2], [l1, l2])
                else:
                    l1 = np.random.binomial(1, p11)
                    l2 = 0
                    self.update_lang_counter([ag_1, ag_2], [l1, l2])
            ag_1.lang_freq['maxmem_list'].append(l1)
            ag_2.lang_freq['maxmem_list'].append(l2)
        if return_langs:
            return l1, l2

    def get_group_conversation_lang(self, initiator, rest_of_group):
        """ Method that allows to
        Initiator of conversation is seprated from rest_of_group
        """
        ags_lang_profile = [(ag.language, ag.lang_freq['cat_pct_s'], ag) for ag in rest_of_group]

        ags_lang_profile = sorted(ags_lang_profile, key=lambda elem: (elem[0], elem[1]))
        rest_of_group = [tup[2] for tup in ags_lang_profile]

        # map init lang profile
        def fun_map_init(x):
            if (x <= 0.2):
                return 0
            elif (0.2<x<=0.5):
                return 1
            elif (0.5<x<=0.8):
                return 2
            elif (x > 0.8):
                return 3

        # define inputs to group_lang_map_dict
        init = fun_map_init(initiator.lang_freq['cat_pct_s'])
        group_lang_min = rest_of_group[0].language
        group_lang_max = rest_of_group[-1].language
        # call group_lang_map_dict model method to get conversat layout
        init_lang, common_lang = self.model.group_lang_map_dict[(init,
                                                                 group_lang_min,
                                                                 group_lang_max)]
        if common_lang: # only one group conversation lang
            langs_list = [init_lang] + [init_lang for ag in rest_of_group]
            self.update_lang_counter([initiator] + rest_of_group, langs_list)
        else: # cases wherein group conversat lang is not unique
            langs_list = []
            if (init, group_lang_min, group_lang_max) in [(0, 0, 2), (1, 0, 2)] :
                for ag in [initiator] + rest_of_group:
                    if ag.language == 2:
                        p_ag = (0.5 * (ag.lang_freq['cat_pct_s']) +
                                0.5 * (ag.lang_freq['cat_pct_h']))
                        langs_list.append(np.random.binomial(1, p_ag))
                    else:
                        langs_list.append(0)
            elif (init, group_lang_min, group_lang_max) == (3, 0, 0):
                langs_list.append(1)
                for ag in rest_of_group:
                        p_ag = (0.5 * (ag.lang_freq['cat_pct_s']) +
                                0.5 * (ag.lang_freq['cat_pct_h']))
                        langs_list.append(np.random.binomial(1, p_ag))
            elif (init, group_lang_min, group_lang_max) in [(2, 0, 2), (3, 0, 1), (3, 0, 2)]:
                for ag in [initiator] + rest_of_group:
                    if ag.language == 0:
                        p_ag = (0.5 * (ag.lang_freq['cat_pct_s']) +
                                0.5 * (ag.lang_freq['cat_pct_h']))
                        langs_list.append(np.random.binomial(1, p_ag))
                    else:
                        langs_list.append(1)

            self.update_lang_counter([initiator] + rest_of_group, langs_list)


    def study_lang(self, lang):
        self.lang_freq['spoken'][lang] += np.random.binomial(1, p=0.25)
        self.lang_freq['heard'][lang] += 1

    def update_lang_pcts(self):
        if sum(self.lang_freq['spoken']) != 0:
            self.lang_freq['cat_pct_s'] = round(self.lang_freq['spoken'][1] / sum(self.lang_freq['spoken']), 2)
        else:
            self.lang_freq['cat_pct_s'] = 0
        if sum(self.lang_freq['heard']) != 0:
            self.lang_freq['cat_pct_h'] = round(self.lang_freq['heard'][1] / sum(self.lang_freq['heard']), 2)
        else:
            self.lang_freq['cat_pct_h'] = 0

    def update_lang_switch(self):
        """ Switch is driven and allowed by heard language """
        # TODO : add condition on cat_pct_s as well, redefine short-term impact. More consistence
        if self.model.schedule.steps > self.lang_freq['maxmem']:
            if self.language == 0:
                if self.lang_freq['cat_pct_h'] >= 0.2:
                    self.language = 1
            elif self.language == 2:
                if self.lang_freq['cat_pct_h'] <= 0.8:
                    self.language = 1
            elif self.language == 1:
                if self.lang_freq['cat_pct_h'] >= 0.8:
                    if 0 not in self.lang_freq['maxmem_list']:
                        self.language = 2
                elif self.lang_freq['cat_pct_h'] <= 0.2:
                    if 1 not in self.lang_freq['maxmem_list']:
                        self.language = 0

    def update_lang_status(self):
        # update lang experience
        self.update_lang_pcts()
        # check lang switch
        self.update_lang_switch()

    def stage_1(self):
        self.speak()

    def stage_2(self):
        self.move_random()
        self.speak()
        self.listen()

    def stage_3(self):
        if self.age < 100:
            self.model.grid.move_agent(self, self.school_coords)
            self.study_lang(0)
            self.study_lang(1)
        else:
            self.model.grid.move_agent(self, self.job_coords)
        self.speak()
        self.speak_in_group()

    def stage_4(self):
        self.move_random()
        self.speak()
        self.model.grid.move_agent(self, self.home_coords)
        self.speak()
        self.age += 1

    def __repr__(self):
        return 'Lang_Agent_{0.unique_id!r}'.format(self)



class Home:
    def __init__(self, pos):
        self.pos=pos
    def __repr__(self):
        return 'Home_{0.pos!r}'.format(self)

class School:
    def __init__(self, pos, num_places):
        self.pos=pos
        self.num_free=num_places
    def __repr__(self):
        return 'School_{0.pos!r}'.format(self)

class Job:
    def __init__(self, pos, num_places, skill_level=0):
        self.pos=pos
        self.num_places=num_places
        self.skill_level = skill_level
    def __repr__(self):
        return 'Job{0.pos!r}'.format(self)