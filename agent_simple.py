# IMPORT LIBS
import random
import numpy as np
import networkx as nx
from collections import deque, Counter, defaultdict

#import private library to model lang zipf CDF
from zipf_generator import Zipf_Mand_CDF_compressed, randZipf

class Simple_Language_Agent:

    #define memory retrievability constant
    k = np.log(10 / 9)

    def __init__(self, model, unique_id, language, age=0,
                 home_coords=None, school_coords=None, job_coords=None):
        self.model = model
        self.unique_id = unique_id
        self.language = language # 0, 1, 2 => spa, bil, cat
        self.age = age
        self.home_coords = home_coords
        self.school_coords = school_coords
        self.job_coords = job_coords



        # define container for languages' tracking and statistics
        self.lang_stats = defaultdict(lambda:defaultdict(dict))


        # Add randomness to number of hours needed to learn second language
        #-> See min_mem_times arg in get_lang_stats method
        # define hours needed for agent to be able to converse in other language
        # -> 10 % of vocabulary ??? ENOUGH??



        if self.language == 0:
            self.lang_stats['L1']['t'] = self.model.lang_ICs['100_pct']['t']
            self.lang_stats['L1']['S'] = self.model.lang_ICs['100_pct']['S']
            self.lang_stats['L1']['R'] = np.exp( - self.k *
                                                         self.lang_stats['L1']['t'] /
                                                         self.lang_stats['L1']['S']
                                                        ).astype(np.float32)
            self.lang_stats['L1']['wc'] = np.zeros(self.model.vocab_red, dtype=np.int32)
            self.lang_stats['L1']['pct'] = np.zeros(3600, dtype = np.float64)



        elif self.language == 2:
            self.lang_stats['L2']['t'] = self.model.lang_ICs['100_pct']['t']
            self.lang_stats['L2']['S'] = self.model.lang_ICs['100_pct']['S']
            self.lang_stats['L2']['R'] = np.exp( - self.k *
                                                           self.lang_stats['L2']['t'] /
                                                           self.lang_stats['L2']['S']
                                                        ).astype(np.float32)
            self.lang_stats['L2']['wc'] = np.zeros(self.model.vocab_red, dtype=np.int32)
            self.lang_stats['L2']['pct'] = np.zeros(3600, dtype = np.float64)

        else: # BILINGUAL
            biling_key = np.random.choice([25,50,100])
            L1_key = str(biling_key) + '_pct'
            L2_key = str(100 - biling_key) + '_pct'
            for lang, key in zip(['L1', 'L2'], [L1_key, L2_key]):
                self.lang_stats[lang]['t'] = self.model.lang_ICs[key]['t']
                self.lang_stats[lang]['S'] = self.model.lang_ICs[key]['S']
                self.lang_stats[lang]['R'] = np.exp(- self.k *
                                                              self.lang_stats[lang]['t'] /
                                                              self.lang_stats[lang]['S']
                                                            ).astype(np.float32)
                self.lang_stats[lang]['wc'] = np.zeros(self.model.vocab_red, dtype=np.int32)
                self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)


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
            other = with_agent

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
            self.lang_stats['l']['LT']['freqs'][l1] += 1
            self.lang_stats['l']['ST']['freqs'][l1][-1] += 1
            self.lang_stats['l']['LT']['freqs'][l2] += 1
            self.lang_stats['l']['ST']['freqs'][l2][-1] += 1

    def read(self):
        pass


    def update_lang_counter(self, ag_1, ag_2, l1, l2):
        """ Update status of lang arrays for two agents involved in conversation
            Args:
                * ag_1, ag_2 : agents objects
                * l1, l2 : integers in [0,1]. Languages spoken by ag_1 and ag_2

        """
        ag_1.lang_stats[l1]



        ag_1.lang_stats['s']['LT']['freqs'][l1] += 1
        ag_1.lang_stats['l']['LT']['freqs'][l2] += 1

        ag_2.lang_stats['s']['LT']['freqs'][l2] += 1
        ag_2.lang_stats['l']['LT']['freqs'][l1] += 1


    def get_conversation_lang(self, ag_1, ag_2, return_values=False):

        if (ag_1.language, ag_2.language) in [(0, 0), (0, 1), (1, 0)]:# spa-bilingual
            l1 = l2 = 0

            self.get_lang_stats()



            #OLD self.update_lang_counter(ag_1, ag_2, 0, 0)

        elif (ag_1.language, ag_2.language) in [(2, 1), (1, 2), (2, 2)]:# bilingual-cat
            l1 = l2 = 1
            self.update_lang_counter(ag_1, ag_2, 1, 1)

        elif (ag_1.language, ag_2.language) == (1, 1): # bilingual-bilingual
            p11 = ((2 / 3) * (ag_1.lang_stats['s']['LT']['L2_pct']) +
                   (1 / 3) * (ag_1.lang_stats['l']['LT']['L2_pct']))
            # find out lang spoken by self ( self STARTS CONVERSATION !!)
            l1 = np.random.binomial(1, p11)
            l2 = l1
            self.update_lang_counter(ag_1, ag_2, l1, l2)

        else: # mono L1 vs mono L2
            p11 = ((2 / 3) * (ag_1.lang_stats['s']['LT']['L2_pct']) +
                   (1 / 3) * (ag_1.lang_stats['l']['LT']['L2_pct']))
            p21 = ((2 / 3) * (ag_2.lang_stats['s']['LT']['L2_pct']) +
                   (1 / 3) * (ag_2.lang_stats['l']['LT']['L2_pct']))
            if ag_1.language == 0:
                l1 = 0
                if (1 - ag_2.lang_stats['s']['LT']['L2_pct']) or (1 - ag_2.lang_stats['l']['LT']['L2_pct']):
                    l2 = np.random.binomial(1, p21)
                    if l2 == 0:
                        self.update_lang_counter(ag_1, ag_2, l1, l2)
                    elif l2 == 1:
                        l1 = np.random.binomial(1, p11)
                        self.update_lang_counter(ag_1, ag_2, l1, l2)
                else:
                    l1 = np.random.binomial(1, p11)
                    l2 = 1
                    self.update_lang_counter(ag_1, ag_2, l1, l2)

            elif ag_1.language == 2:
                l1 = 1
                if (ag_2.lang_stats['s']['LT']['L2_pct']) or (ag_2.lang_stats['l']['LT']['L2_pct']):
                    l2 = np.random.binomial(1, p21)
                    if l2 == 1:
                        self.update_lang_counter(ag_1, ag_2, l1, l2)
                    elif l2 == 0:
                        l1 = np.random.binomial(1, p11)
                        self.update_lang_counter(ag_1, ag_2, l1, l2)
                else:
                    l1 = np.random.binomial(1, p11)
                    l2 = 0
                    self.update_lang_counter(ag_1, ag_2, l1, l2)

        if return_values:
            return l1, l2

    def study_lang(self, lang):
        if np.random.binomial(1, p=0.25):
            self.lang_stats['s']['LT']['freqs'][lang] += 1
            self.lang_stats['s']['ST']['freqs'][lang] += 1
        self.lang_stats['l']['LT']['freqs'][lang] += 1
        self.lang_stats['l']['ST']['freqs'][lang] += 1


    def get_words_per_conv(self, conversation=True):
        """ Computes number of words spoken per conversation for a given age
            If conversation=False, computes average number of words per day,
            assuming 16000 tokens per adult per day as average """
        if self.age < 36 * 14:
            factor = 2.5 + 100 * np.exp(-0.014 * self.age)
        elif 36 * 14 <= self.age <= 36 * 65:
            factor = 2.5
        else:
            factor = 1.5 + np.exp(0.002 * (self.age - 36 * 65))

        if conversation:
            return self.model.num_words_conv / factor
        else:
            return self.model.vocab_red / factor



    def get_lang_stats(self, lang, pct_hours,
                       a=7.6, b=0.023, c=-0.031, d=-0.2,
                       min_mem_times=5, max_age=60, pct_threshold=0.9):
        """ Function to compute and update main arrays that define agent linguistic knowledge
            Args:
                * t: numpy array(shape=vocab_size) that counts elapsed steps from last activation of each word
                * S: numpy array(shape=vocab_size) that measures memory stability for each word
                * cdf_data: dict with keys 's','w'. It gives a CDF for each step
                * num_steps1, num_steps2: scalars. Initial and final steps for calculation
                * pct_hours: percentage of daily hours devoted to chosen language
                * lang_knowledge: array with shape = steps. For each step ,
                  it gives percentage knowledge of language as measured
                  by a given threshold of retrievability(pct_threshold)
                * a, b, c, d: parameters to definE memory functioN from SUPERMEMO by Piotr A. Wozniak

            MEMORY MODEL: https://www.supermemo.com/articles/stability.htm

            Assumptions ( see "HOW MANY WORDS DO WE KNOW ???" By Marc Brysbaert*,
            Michaël Stevens, Paweł Mandera and Emmanuel Keuleers):
                * ~16000 spoken tokens per day + 16000 heard tokens per day + TV, RADIO
                * 1min reading -> 220-300 tokens with large individual differences, thus
                  in 1 h we get ~ 16000 words"""


        # if not self.age:
        #     self.lang_stats[lang]['R'] = np.zeros(self.model.vocab_red, dtype=np.float32)
        # else:
        #     self.lang_stats[lang]['R'] = np.exp(- self.k * self.lang_stats[lang]['t'] / self.lang_stats[lang]['S'])

        # get real number of words per conversation for a given age
        num_words = self.get_words_per_conv()
        # get conv samples
        zipf_samples = randZipf(self.model.cdf_data['s'][self.age], int(pct_hours * num_words * 10))

        # assess which words and how many of each were encountered in current step
        act, act_c = np.unique(zipf_samples, return_counts=True)
        # update word counter according to previous line
        self.lang_stats[lang]['wc'][act] += act_c
        # check which words are available for memorization ( need minimum number of times)
        mem_availab_words = np.where(self.lang_stats[lang]['wc'] > min_mem_times)[0]

        # compute indices of active words that are available for memory
        idxs_act = np.nonzero(np.in1d(act, mem_availab_words, assume_unique=True))
        # get really activated words
        act = act[idxs_act]
        # Apply indices to counts ( retrieve only counts of really active words)
        act_c = act_c[idxs_act]

        if act.any():  # check that there are any real active words
            # compute increase in memory stability S due to (re)activation
            delta_S = a * (self.lang_stats[lang]['S'][act] ** (-b)) * np.exp(c * 100 * self.lang_stats[lang]['R'][act]) + d
            # update memory stability value
            # np.add.at(S, act, delta_S) #
            self.lang_stats[lang]['S'][act] += delta_S

            # define mask to update elapsed-steps array t
            mask = np.zeros(self.model.vocab_red, dtype=np.bool)
            mask[act] = True
            self.lang_stats[lang]['t'][~mask] += 1  # add ones to last activation time counter if word not act
            self.lang_stats[lang]['t'][mask] = 0  # set last activation time counter to one if word act

            # discount one to counts
            act_c -= 1
            # Simplification with good approx : we apply delta_S without iteration !!
            delta_S = act_c * (a * (self.lang_stats[lang]['S'][act] ** (-b)) * np.exp(c * 100 * self.lang_stats[lang]['R'][act]) + d)
            # np.add.at(S, act, delta_S) #
            self.lang_stats[lang]['S'][act] += delta_S

        else:
            # define mask to update elapsed-steps array t
            mask = np.zeros(self.model.vocab_red, dtype=np.bool)
            mask[act] = True
            self.lang_stats[lang]['t'][~mask] += 1  # add ones to last activation time counter if word not act
            self.lang_stats[lang]['t'][mask] = 0  # set last activation time counter to one if word act

        # memory becomes ever shakier after turning 65...
        if self.age > 65 * 36:
            self.lang_stats[lang]['S'] = np.where(self.lang_stats[lang]['S'] >= 0.01,
                                                  self.lang_stats[lang]['S'] - 0.01,
                                                  0.000001)
        # compute memory retrievability R from t, S
        self.lang_stats[lang]['R'] = np.exp(-self.k * self.lang_stats[lang]['t'] / self.lang_stats[lang]['S'])
        self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > pct_threshold)[0].shape[0] /
                                                  self.model.vocab_red)



    def speak_model(self, lang):

        # sample must come from AVAILABLE words in R!!!! This can be done in two steps

        # 1. First sample from lang CDF ( that encapsulates all to-be-known concepts at a given age-step)
        word_samples = randZipf(self.model.cdf_data['s'][self.age], int(self.get_words_per_conv() * 10))
        act, act_c = np.unique(word_samples, return_counts=True)
        # 2. Then assess which words can be succesfully retrieved from memory
        # get mask for words successfully retrieved from memory
        mask_R = np.random.rand(self.lang_stats[lang]['R'][act].shape[0]) <= self.lang_stats[lang]['R'][act]

        spoken_words = act[mask_R]

        return spoken_words

        # TODO: NOW NEED MODEL of how to deal with missed words = > L3, emergent lang with mixed vocab ???




    def update_lang_pcts(self):

        freqs_s = self.lang_stats['s']['LT']['freqs']
        freqs_l = self.lang_stats['l']['LT']['freqs']

        self.lang_stats['s']['LT']['L2_pct'] = round(freqs_s[1] / sum(freqs_s), 2)
        self.lang_stats['l']['LT']['L2_pct'] = round(freqs_l[1] / sum(freqs_l), 2)

        a = np.array(self.lang_stats['s']['ST']['freqs'])
        np.average(np.around(a[0] / (a[0] + a[1]), 2), weights=a[0] + a[1])

    def update_lang_switch(self): #TODO : use ST info and model correct threshold
        """Between 600 and 1500 hours to learn a second similar language at decent level"""
        days_per_year = 365
        max_lang_h_day = 16
        max_words_per_day = 50 # saturation value
        lang_hours_per_day = (max_lang_h_day *
                             (sum(self.lang_stats['l']['LT']['freqs']) + sum(self.lang_stats['s']['LT']['freqs'])) /
                             (self.model.schedule.steps + max_words_per_day))
        steps_per_year = 36.5
        pct = (self.lang_stats['l']['LT']['L2_pct'] + self.lang_stats['s']['LT']['L2_pct'])
        if self.model.schedule.steps > self.lang_stats['maxmem']:
            if self.language == 0:
                num_hours_L2 = (pct * days_per_year *
                                lang_hours_per_day * self.model.schedule.steps) / steps_per_year
                if num_hours_L2 >= self.lang_stats['learning_hours']:
                    self.language = 1
            elif self.language == 2:
                pct = 1 - pct
                num_hours_L1 = (pct * days_per_year *
                                lang_hours_per_day * self.model.schedule.steps) / steps_per_year
                if num_hours_L1 >= self.lang_stats['learning_hours']:
                    self.language = 1
            else: # LANGUAGE ATTRITION

                if pct:
                    pass



                if self.lang_stats['l']['LT']['L2_pct'] >= 0.8:
                    if 0 not in self.lang_stats['s']['ST']['freqs']:
                        self.language = 2
                elif self.lang_stats['l']['LT']['L2_pct'] <= 0.2:
                    if 1 not in self.lang_stats['s']['ST']['freqs']:
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
        self.move_random()
        self.listen()

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
        # update at the end of each step
        self.update_lang_status()
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