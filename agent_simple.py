# IMPORT LIBS
import random
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from collections import deque, Counter, defaultdict

#import private library to model lang zipf CDF
from zipf_generator import Zipf_Mand_CDF_compressed, randZipf

class Simple_Language_Agent:

    #define memory retrievability constant
    k = np.log(10 / 9)

    def __init__(self, model, unique_id, language, lang_act_thresh=0.1, lang_passive_thresh=0.025, age=0,
                 num_children=0, home_coords=None, school_coords=None, job_coords=None, city_idx=None):
        self.model = model
        self.unique_id = unique_id
        self.language = language # 0, 1, 2 => spa, bil, cat
        self.lang_thresholds = {'speak':lang_act_thresh, 'understand':lang_passive_thresh}
        self.age = age
        self.num_children = num_children
        self.home_coords = home_coords
        self.school_coords = school_coords
        self.job_coords = job_coords
        self.city_idx = city_idx

        # define container for languages' tracking and statistics
        self.lang_stats = defaultdict(lambda:defaultdict(dict))
        self.day_mask = {'L1':np.zeros(self.model.vocab_red, dtype=np.bool),
                         'L2':np.zeros(self.model.vocab_red, dtype=np.bool)}


        # Add randomness to number of hours needed to learn second language
        #-> See min_mem_times arg in update_lang_arrays method
        # define hours needed for agent to be able to converse in other language
        # -> 10 % of vocabulary ??? ENOUGH??



        if self.language == 0:
            # numpy array(shape=vocab_size) that counts elapsed steps from last activation of each word
            self.lang_stats['L1']['t'] = np.copy(self.model.lang_ICs['100_pct']['t'])
            # S: numpy array(shape=vocab_size) that measures memory stability for each word
            self.lang_stats['L1']['S'] = np.copy(self.model.lang_ICs['100_pct']['S'])
            # compute R from t, S
            self.lang_stats['L1']['R'] = np.exp( - self.k *
                                                         self.lang_stats['L1']['t'] /
                                                         self.lang_stats['L1']['S']
                                                        ).astype(np.float64)
            # word counter
            self.lang_stats['L1']['wc'] = np.copy(self.model.lang_ICs['100_pct']['wc'])
            #L2
            self.lang_stats['L2']['S'] = np.full(self.model.vocab_red, 0.01)
            self.lang_stats['L2']['t'] = np.full(self.model.vocab_red, 1000)
            self.lang_stats['L2']['R'] = np.exp( - self.k *
                                                         self.lang_stats['L2']['t'] /
                                                         self.lang_stats['L2']['S']
                                                        ).astype(np.float32)
            self.lang_stats['L2']['wc'] = np.zeros(self.model.vocab_red)

            for lang in ['L1', 'L2']:
                self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
                self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                          self.model.vocab_red)



        elif self.language == 2:
            self.lang_stats['L2']['t'] = np.copy(self.model.lang_ICs['100_pct']['t'])
            self.lang_stats['L2']['S'] = np.copy(self.model.lang_ICs['100_pct']['S'])
            self.lang_stats['L2']['R'] = np.exp( - self.k *
                                                           self.lang_stats['L2']['t'] /
                                                           self.lang_stats['L2']['S']
                                                        ).astype(np.float64)
            self.lang_stats['L2']['wc'] = np.copy(self.model.lang_ICs['100_pct']['wc'])

            #L2
            self.lang_stats['L1']['S'] = np.full(self.model.vocab_red, 0.01)
            self.lang_stats['L1']['t'] = np.full(self.model.vocab_red, 1000)
            self.lang_stats['L1']['R'] = np.exp( - self.k *
                                                         self.lang_stats['L1']['t'] /
                                                         self.lang_stats['L1']['S']
                                                        ).astype(np.float64)
            self.lang_stats['L1']['wc'] = np.zeros(self.model.vocab_red)

            for lang in ['L1', 'L2']:
                self.lang_stats[lang]['pct'] = np.zeros(3600, dtype = np.float64)
                self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                          self.model.vocab_red)


        else: # BILINGUAL
            biling_key = np.random.choice([25,50,75])
            L1_key = str(biling_key) + '_pct'
            L2_key = str(100 - biling_key) + '_pct'
            for lang, key in zip(['L1', 'L2'], [L1_key, L2_key]):
                self.lang_stats[lang]['t'] = np.copy(self.model.lang_ICs[key]['t'])
                self.lang_stats[lang]['S'] = np.copy(self.model.lang_ICs[key]['S'])
                self.lang_stats[lang]['R'] = np.exp(- self.k *
                                                    self.lang_stats[lang]['t'] /
                                                    self.lang_stats[lang]['S']
                                                    ).astype(np.float64)
                self.lang_stats[lang]['wc'] = np.copy(self.model.lang_ICs[key]['wc'])
                self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
                self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                          self.model.vocab_red)


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

    def reproduce(self, age_1=20, age_2=40):
        if (age_1 * 36 <= self.age <= age_2 * 36) and (self.num_children < 1) and (random.random() > 1 - 5/20):
            id_ = self.model.set_available_ids.pop()
            lang = self.language
            # find closest school to parent home
            clust_schools_coords = [sc.pos for sc in self.clusters_info[self.city_idx]['schools']]
            closest_school_idx = np.argmin([pdist([self.home_coords, sc_coord])
                                        for sc_coord in clust_schools_coords])
            xs, ys = self.clusters_info[self.city_idx]['schools'][closest_school_idx].pos
            # instantiate agent
            a = Simple_Language_Agent(self.model, id_, lang,
                                      home_coords=self.home_coords, school_coords=(xs, ys),
                                      job_coords=None, city_idx=self.city_idx)
            # Add agent to model
            self.model.add_agent(a, self.pos)
            # Update num of children
            self.num_children += 1

    def simulate_random_death(self, age_1=20, age_2=75, age_3=90, prob_1=0.25, prob_2=0.7):  ##
        # transform ages to steps
        age_1, age_2, age_3 = age_1 * 36, age_2 * 36, age_3 * 36
        # define stochastic probability of agent death as function of age
        if (self.age > age_1) and (self.age <= age_2):
            if random.random() < prob_1 / (age_2 - age_1):  # 25% pop will die through this period
                self.remove_after_death()
        elif (self.age > age_2) and (self.age < age_3):
            if random.random() < prob_2 / (age_3 - age_2):  # 70% will die
                self.remove_after_death()
        elif self.age >= age_3:
            self.remove_after_death()

    def remove_after_death(self):
        """ call this function if death conditions
        for agent are verified """
        for network in [self.model.family_network,
                        self.model.known_people_network,
                        self.model.friendship_network]:
            try:
                network.remove_node(self)
            except nx.NetworkXError:
                pass
        # find agent coordinates
        x, y = self.pos
        # make id from deceased agent available
        self.model.set_available_ids.add(self.unique_id)
        # remove agent from grid and schedule
        self.model.grid._remove_agent((x,y), self)
        self.model.schedule.remove(self)

    def speak(self, with_agent=None):
        """ Pick random lang_agent from current cell and start a conversation
            with it. It updates heard words in order to shape future vocab.
            Language of the conversation is determined by given laws
            This method can also simulate distance contact e.g.
            phone, messaging, etc ... by specifying an agent through 'with_agent'

            Arguments:
                * with_agent : specify a specific agent with which conversation will take place
                  By default the agent will be picked randomly from all lang agents in current cell

            Returns:
                * Defines conversation and language(s) in which it takes place.
                  Updates heard/used stats
        """
        if with_agent is None:
            pos = [self.pos]
            # get all agents currently placed on chosen cell
            others = self.model.grid.get_cell_list_contents(pos)
            others.remove(self)
            ## linguistic model of encounter with another random agent
            if len(others) >= 1:
                other = random.choice(others)
                self.get_conversation_lang(self, other)
        else:
            self.get_conversation_lang(self, with_agent)

    def listen(self):
        """Listen to random agents placed on the same cell as calling agent"""
        pos = [self.pos]
        # get all agents currently placed on chosen cell
        others = self.model.grid.get_cell_list_contents(pos)
        others.remove(self)
        ## if two or more agents in cell, conversation is possible
        if len(others) >= 2:
            ag_1, ag_2 = np.random.choice(others, size=2, replace=False)
            l1, l2 = self.get_conversation_lang(ag_1, ag_2, return_values=True)
            k1, k2 = 'L' + str(l1), 'L' + str(l2)

            self.update_lang_arrays(l1, spoken_words, speak=False)
            self.update_lang_arrays(l2, spoken_words, speak=False)


            #OLD
            self.lang_stats[k1]['LT']['freqs'][l1] += 1
            self.lang_stats['l']['ST']['freqs'][l1][-1] += 1
            self.lang_stats['l']['LT']['freqs'][l2] += 1
            self.lang_stats['l']['ST']['freqs'][l2][-1] += 1

    def read(self):
        pass

    def get_conversation_lang(self, ag_1, ag_2, return_values=False):
        if (ag_1.language, ag_2.language) in [(0, 0), (0, 1), (1, 0)]:# spa-bilingual
            l1 = l2 = 0
            ag_1.speak_choice_model(l1, ag_2)
            ag_2.speak_choice_model(l2, ag_1)

        elif (ag_1.language, ag_2.language) in [(2, 1), (1, 2), (2, 2)]:# bilingual-cat
            l1 = l2 = 1
            ag_1.speak_choice_model(l1, ag_2)
            ag_2.speak_choice_model(l2, ag_1)

        elif (ag_1.language, ag_2.language) == (1, 1): # bilingual-bilingual
            # simplified PRELIMINARY assumption: ag_1 will start speaking the language they speak best
            # (at this stage no modeling of place, inertia, known-person influence)
            l1 = np.argmax([ag_1.lang_stats['L1']['pct'][self.age], ag_1.lang_stats['L2']['pct'][self.age]])
            l2 = l1
            ag_1.speak_choice_model(l1, ag_2)
            ag_2.speak_choice_model(l2, ag_1)

        else: # mono L1 vs mono L2 with relatively close languages -> SOME understanding is possible (LOW THRESHOLD)
            ag1_pcts = (ag_1.lang_stats['L1']['pct'][self.age], ag_2.lang_stats['L2']['pct'][self.age])
            ag2_pcts = (ag_2.lang_stats['L1']['pct'][self.age], ag_2.lang_stats['L2']['pct'][self.age])
            # if ag_2 can understand some of ag_1 lang
            if ag_1.language == 0:
                # if each agent can understand some of other agent's language
                if (ag1_pcts[1] >= ag_1.lang_thresholds['understand'] and
                ag2_pcts[0] >= ag_2.lang_thresholds['understand']):
                    l1, l2 = 0, 1
                # otherwise
                elif (ag1_pcts[1] < ag_1.lang_thresholds['understand'] and
                ag2_pcts[0] >= ag_2.lang_thresholds['understand']):
                    l1, l2 = 0, 0
                elif (ag1_pcts[1] >= ag_1.lang_thresholds['understand'] and
                ag2_pcts[0] < ag_2.lang_thresholds['understand']):
                    l1, l2 = 1, 1
                else:
                    l1, l2 = None, None # NO CONVERSATION POSSIBLE
            elif ag_1.language == 2:
                if (ag1_pcts[0] >= ag_1.lang_thresholds['understand'] and
                ag2_pcts[1] >= ag_2.lang_thresholds['understand']):
                    l1, l2 = 1, 0
                elif (ag1_pcts[0] < ag_1.lang_thresholds['understand'] and
                ag2_pcts[1] >= ag_2.lang_thresholds['understand']):
                    l1, l2 = 1, 1
                elif (ag1_pcts[0] >= ag_1.lang_thresholds['understand'] and
                ag2_pcts[1] < ag_2.lang_thresholds['understand']):
                    l1, l2 = 0, 0
                else:
                    l1, l2 = None, None # NO CONVERSATION POSSIBLE
            if l1 or l2: # call update only if conversation takes place
                ag_1.speak_choice_model(l1, ag_2, long=False)
                ag_2.speak_choice_model(l2, ag_1, long=False)
        if return_values:
            return l1, l2

    def study_lang(self, lang):
        pass

    def get_words_per_conv(self, long=True, age_1=14, age_2=65):
        """ Computes number of words spoken per conversation for a given age
            If conversation=False, computes average number of words per day,
            assuming 16000 tokens per adult per day as average """
        if self.age < 36 * age_1:
            factor = 2.5 + 100 * np.exp(-0.014 * self.age)
        elif 36 * age_1 <= self.age <= 36 * age_2:
            factor = 2.5
        else:
            factor = 1.5 + np.exp(0.002 * (self.age - 36 * age_2))

        if long:
            return self.model.num_words_conv[1] / factor
        else:
            return self.model.num_words_conv[0] / factor

    def update_lang_arrays(self, lang, sample_words, speak=True, a=7.6, b=0.023, c=-0.031, d=-0.2,
                           min_mem_times=5, pct_threshold=0.9):
        """ Function to compute and update main arrays that define agent linguistic knowledge
            Args:
                * lang :
                * sample_words:
                * speak
                * a, b, c, d: parameters to define memory function from SUPERMEMO by Piotr A. Wozniak
                * min_mem_times
                * max_age
                * pct_threshold

            MEMORY MODEL: https://www.supermemo.com/articles/stability.htm

            Assumptions ( see "HOW MANY WORDS DO WE KNOW ???" By Marc Brysbaert*,
            Michaël Stevens, Paweł Mandera and Emmanuel Keuleers):
                * ~16000 spoken tokens per day + 16000 heard tokens per day + TV, RADIO
                * 1min reading -> 220-300 tokens with large individual differences, thus
                  in 1 h we get ~ 16000 words"""

        # if not active_words:
        #     # get real number of words per conversation for a given age
        #     num_words = self.get_words_per_conv()
        #     # get conv samples
        #     zipf_samples = randZipf(self.model.cdf_data['s'][self.age], int(num_words * 10))
        #     # assess which words and how many of each were encountered in current step
        #     act, act_c = np.unique(zipf_samples, return_counts=True)
        # else:
        #     act, act_c = active_words

        act, act_c = sample_words
        # update word counter with newly active words
        self.lang_stats[lang]['wc'][act] += act_c
        # if words are from listening, they might be new to agent
        if not speak:
            # check which words are available for memorization (need minimum number of times)
            mem_availab_words = np.where(self.lang_stats[lang]['wc'] > min_mem_times)[0]
            # compute indices of active words that are available for memory
            idxs_act = np.nonzero(np.in1d(act, mem_availab_words, assume_unique=True))
            # get really activated words and apply indices to counts ( retrieve only counts of really active words)
            act, act_c = act[idxs_act], act_c[idxs_act]

        if act.any():  # check that there are any real active words
            # compute increase in memory stability S due to (re)activation
            # TODO : I think it should be dS[reading]  < dS[media_listening]  < dS[listen_in_conv] < dS[speaking]
            delta_S = a * (self.lang_stats[lang]['S'][act] ** (-b)) * np.exp(c * 100 * self.lang_stats[lang]['R'][act]) + d
            # update memory stability value
            self.lang_stats[lang]['S'][act] += delta_S

            # define mask to update elapsed-steps array t
            #mask = np.zeros(self.model.vocab_red, dtype=np.bool)
            self.day_mask[lang][act] = True
            #self.lang_stats[lang]['t'][~mask] += 1  # add ones to last activation time counter if word not act
            self.lang_stats[lang]['t'][self.day_mask[lang]] = 0  # set last activation time counter to zero if word act

            # discount one to counts
            act_c -= 1
            # Simplification with good approx : we apply delta_S without iteration !!
            delta_S = act_c * (a * (self.lang_stats[lang]['S'][act] ** (-b)) * np.exp(c * 100 * self.lang_stats[lang]['R'][act]) + d)
            # update
            self.lang_stats[lang]['S'][act] += delta_S

        else:
            # define mask to update elapsed-steps array t
            #mask = np.zeros(self.model.vocab_red, dtype=np.bool)
            self.day_mask[lang][act] = True
            #self.lang_stats[lang]['t'][~mask] += 1  # add ones to last activation time counter if word not act
            self.lang_stats[lang]['t'][self.day_mask[lang]] = 0  # set last activation time counter to one if word act


        # compute memory retrievability R from t, S
        self.lang_stats[lang]['R'] = np.exp(-self.k * self.lang_stats[lang]['t'] / self.lang_stats[lang]['S'])
        self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > pct_threshold)[0].shape[0] /
                                                  self.model.vocab_red)

    def speak_choice_model(self, lang, ag2, long=True, return_values=False):
        """ Method that models word choice by self agent in a conversation
            Word choice is governed by vocabulary knowledge constraints
            Both self and ag2 lang arrays are updated according to the sampled words

            Args:
                * lang: integer in [0, 1]
                * ag2: agent instance with which self is talking to
                * long: boolean that defines conversation length"""

        # sample must come from AVAILABLE words in R ( retrievability) !!!! This can be modeled in TWO STEPS
        # 1. First sample from lang CDF ( that encapsulates all to-be-known concepts at a given age-step)
        word_samples = randZipf(self.model.cdf_data['s'][self.age], int(self.get_words_per_conv(long) * 10))
        act, act_c = np.unique(word_samples, return_counts=True)
        # 2. Then assess which sampled words can be succesfully retrieved from memory
        # get mask for words successfully retrieved from memory
        if lang == 0:
            lang = 'L1'
        else:
            lang = 'L2'
        mask_R = np.random.rand(self.lang_stats[lang]['R'][act].shape[0]) <= self.lang_stats[lang]['R'][act]

        spoken_words = act[mask_R], act_c[mask_R]
        # call own update
        self.update_lang_arrays(lang, spoken_words)
        # call listener's update
        ag2.update_lang_arrays(lang, spoken_words, speak=False)


        # TODO: NOW NEED MODEL of how to deal with missed words = > L3, emergent lang with mixed vocab ???
        if return_values:
            return spoken_words

    def update_lang_pcts(self):

        freqs_s = self.lang_stats['s']['LT']['freqs']
        freqs_l = self.lang_stats['l']['LT']['freqs']

        self.lang_stats['s']['LT']['L2_pct'] = round(freqs_s[1] / sum(freqs_s), 2)
        self.lang_stats['l']['LT']['L2_pct'] = round(freqs_l[1] / sum(freqs_l), 2)

        a = np.array(self.lang_stats['s']['ST']['freqs'])
        np.average(np.around(a[0] / (a[0] + a[1]), 2), weights=a[0] + a[1])

    def update_lang_switch(self, switch_threshold=0.1):
        """Switch to a new linguistic regime whn threshold is reached
           If language knowledge falls below switch_threshold value, agent
           becomes monolingual"""

        # days_per_year = 365
        # max_lang_h_day = 16
        # max_words_per_day = 50 # saturation value
        # lang_hours_per_day = (max_lang_h_day *
        #                      (sum(self.lang_stats['l']['LT']['freqs']) + sum(self.lang_stats['s']['LT']['freqs'])) /
        #                      (self.model.schedule.steps + max_words_per_day))
        # steps_per_year = 36.5
        # pct = (self.lang_stats['l']['LT']['L2_pct'] + self.lang_stats['s']['LT']['L2_pct'])
        # if self.model.schedule.steps > self.lang_stats['maxmem']:
        #     if self.language == 0:
        #         num_hours_L2 = (pct * days_per_year *
        #                         lang_hours_per_day * self.model.schedule.steps) / steps_per_year
        #         if num_hours_L2 >= self.lang_stats['learning_hours']:
        #             self.language = 1
        #     elif self.language == 2:
        #         pct = 1 - pct
        #         num_hours_L1 = (pct * days_per_year *
        #                         lang_hours_per_day * self.model.schedule.steps) / steps_per_year
        #         if num_hours_L1 >= self.lang_stats['learning_hours']:
        #             self.language = 1
        #     else: # LANGUAGE ATTRITION
        #
        #         if pct:
        #             pass
        #
        #
        #
        #         if self.lang_stats['l']['LT']['L2_pct'] >= 0.8:
        #             if 0 not in self.lang_stats['s']['ST']['freqs']:
        #                 self.language = 2
        #         elif self.lang_stats['l']['LT']['L2_pct'] <= 0.2:
        #             if 1 not in self.lang_stats['s']['ST']['freqs']:
        #                 self.language = 0
        if self.language == 0:
            if self.lang_stats['L2']['pct'][self.age] >= switch_threshold:
                self.language = 1
        elif self.language == 2:
            if self.lang_stats['L1']['pct'][self.age] >= switch_threshold:
                self.language = 1
        elif self.language == 1:
            if self.lang_stats['L1']['pct'][self.age] < switch_threshold:
                self.language == 2
            elif self.lang_stats['L2']['pct'][self.age] < switch_threshold:
                self.language == 0









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
        self.move_random()
        #self.listen()

    def stage_3(self):
        if self.age < 100:
            self.model.grid.move_agent(self, self.school_coords)
            self.study_lang(0)
            self.study_lang(1)
            self.speak()
        else:
            self.model.grid.move_agent(self, self.job_coords)
            self.speak()

    def stage_4(self):
        self.move_random()
        self.speak()
        self.model.grid.move_agent(self, self.home_coords)
        self.speak()
        # memory becomes ever shakier after turning 65...
        if self.age > 65 * 36:
            for lang in ['L1', 'L2']:
                self.lang_stats[lang]['S'] = np.where(self.lang_stats[lang]['S'] >= 0.01,
                                                      self.lang_stats[lang]['S'] - 0.01,
                                                      0.000001)
        # update at the end of each step
        for lang in ['L1', 'L2']:
            self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                      self.model.vocab_red)

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