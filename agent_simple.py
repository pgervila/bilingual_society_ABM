# IMPORT LIBS
import random
import numpy as np
import networkx as nx
import bisect
from scipy.spatial.distance import pdist
from collections import defaultdict

#import private library to model lang zipf CDF
from zipf_generator import Zipf_Mand_CDF_compressed, randZipf

class Simple_Language_Agent:

    #define memory retrievability constant
    k = np.log(10 / 9)

    def __init__(self, model, unique_id, language, lang_act_thresh=0.1, lang_passive_thresh=0.025, age=0,
                 num_children=0, ag_home=None, ag_school=None, ag_job=None, city_idx=None, import_IC=False):
        self.model = model
        self.unique_id = unique_id
        self.language = language # 0, 1, 2 => spa, bil, cat
        self.lang_thresholds = {'speak':lang_act_thresh, 'understand':lang_passive_thresh}
        self.age = age
        self.num_children = num_children # TODO : group marital/parental info in dict ??

        self.loc_info = {'home':ag_home, 'school':ag_school, 'job':ag_job, 'city_idx':city_idx}

        # define container for languages' tracking and statistics
        # Need three key levels to entirely define lang branch ->
        # Language {'L1', 'L2'}, mode{'a','p'}, attribute{'R','t','S','pct'}
        self.lang_stats = defaultdict(lambda:defaultdict(dict))
        self.day_mask = {'L1':np.zeros(self.model.vocab_red, dtype=np.bool),
                         'L2':np.zeros(self.model.vocab_red, dtype=np.bool)}
        if import_IC:
            self.set_lang_ics()
        else:
            for lang in ['L1', 'L2']:
                self._set_null_lang_attrs(lang)

    def _set_lang_attrs(self, lang, pct_key):
        """ Private method that sets agent linguistic status for a given age
            Args:
                * lang: string. It can take two different values: 'L1' or 'L2'
                * pct_key: string. It must be of the form '%_pct' with % an integer
                  from following list [10,25,50,75,90,100]. ICs are not available for every single level
        """
        # numpy array(shape=vocab_size) that counts elapsed steps from last activation of each word
        self.lang_stats[lang]['t'] = np.copy(self.model.lang_ICs[pct_key]['t'][self.age])
        # S: numpy array(shape=vocab_size) that measures memory stability for each word
        self.lang_stats[lang]['S'] = np.copy(self.model.lang_ICs[pct_key]['S'][self.age])
        # compute R from t, S (R defines retrievability of each word)
        self.lang_stats[lang]['R'] = np.exp(- self.k *
                                                  self.lang_stats[lang]['t'] /
                                                  self.lang_stats[lang]['S']
                                                  ).astype(np.float64)
        # word counter
        self.lang_stats[lang]['wc'] = np.copy(self.model.lang_ICs[pct_key]['wc'][self.age])
        # vocab pct
        self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
        self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                  self.model.vocab_red)

    def _set_null_lang_attrs(self, lang, S_0=0.01, t_0=1000):
        """Private method that sets null agent linguistic status, i.e. without knowledge
           of the specified language
           Args:
               * lang: string. It can take two different values: 'L1' or 'L2'
               * S_0: float. Initial value of memory stability
               * t_0: integer. Initial value of time-elapsed ( in days) from last time words were encountered
        """
        self.lang_stats[lang]['S'] = np.full(self.model.vocab_red, S_0)
        self.lang_stats[lang]['t'] = np.full(self.model.vocab_red, t_0)
        self.lang_stats[lang]['R'] = np.exp(- self.k *
                                            self.lang_stats[lang]['t'] /
                                            self.lang_stats[lang]['S']
                                            ).astype(np.float32)
        self.lang_stats[lang]['wc'] = np.zeros(self.model.vocab_red)
        self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
        self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                  self.model.vocab_red)

    def set_lang_ics(self, S_0=0.01, t_0=1000, biling_key=None):
        """ set agent's linguistic Initial Conditions
        Args:
            * S_0: float <= 1. Initial memory intensity
            * t_0: last-activation days counter
            * biling_key: integer from [10,25,50,75,90,100]. Specify only if
              specific bilingual level is needed as input
        """
        if self.language == 0:
            self._set_lang_attrs('L1', '100_pct')
            self._set_null_lang_attrs('L2', S_0, t_0)
        elif self.language == 2:
            self._set_null_lang_attrs('L1', S_0, t_0)
            self._set_lang_attrs('L2', '100_pct')
        else: # BILINGUAL
            if not biling_key:
                biling_key = np.random.choice(self.model.ic_pct_keys)
            L1_key = str(biling_key) + '_pct'
            L2_key = str(100 - biling_key) + '_pct'
            for lang, key in zip(['L1', 'L2'], [L1_key, L2_key]):
                self._set_lang_attrs(lang, key)

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
        age_1, age_2 = age_1 * 36, age_2 * 36
        if (age_1 <= self.age <= age_2) and (self.num_children < 1) and (random.random() < 5/(age_2 - age_1)):
            id_ = self.model.set_available_ids.pop()
            lang = self.language
            # find closest school to parent home
            city_idx = self.loc_info['city_idx']
            clust_schools_coords = [sc.pos for sc in self.model.clusters_info[city_idx]['schools']]
            closest_school_idx = np.argmin([pdist([self.loc_info['home'].pos, sc_coord])
                                            for sc_coord in clust_schools_coords])
            # instantiate agent
            a = Simple_Language_Agent(self.model, id_, lang, ag_home=self.loc_info['home'],
                                      ag_school=self.model.clusters_info[city_idx]['schools'][closest_school_idx],
                                      ag_job=None,
                                      city_idx=self.loc_info['city_idx'])
            # Add agent to model
            self.model.add_agent_to_grid_sched_networks(a)
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

    def look_for_job(self):
        # loop through shuffled job centers list until a job is found
        np.random.shuffle(self.model.clusters_info[self.loc_info['city_idx']]['jobs'])
        for job_c in self.model.clusters_info[self.loc_info['city_idx']]['jobs']:
            if job_c.num_places:
                job_c.num_places -= 1
                self.loc_info['job'] = job_c
                break

    def speak(self, with_agent=None, lang=None):
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
                self.get_conv_lang(self, other)
        else:
            if not lang:
                self.get_conv_lang(self, with_agent)
            else:
                self.vocab_choice_model(lang, with_agent)
                with_agent.vocab_choice_model(lang, self)

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
                self.get_conv_lang(initiator, rest_of_group)
            else:
                rest_of_group = list(group_set.difference({self}))
                self.get_conv_lang(self, rest_of_group)

    def listen(self):
        """Listen to random agents placed on the same cell as calling agent"""
        pos = [self.pos]
        # get all agents currently placed on chosen cell
        others = self.model.grid.get_cell_list_contents(pos)
        others.remove(self)
        ## if two or more agents in cell, conversation is possible
        if len(others) >= 2:
            ag_1, ag_2 = np.random.choice(others, size=2, replace=False)
            l1, l2 = self.get_conv_lang(ag_1, [ag_2], return_values=True)
            k1, k2 = 'L' + str(l1), 'L' + str(l2)

            self.update_lang_arrays(l1, spoken_words, speak=False)
            self.update_lang_arrays(l2, spoken_words, speak=False)

    def read(self):
        pass

    def get_conv_lang(self, ag_init, others):
        """
        Method to model conversation between 2 or more agents
        It defines speakers, lang for each speakers and makes all of them speak and the rest listen
        It implements MAXIMIN language rule from Van Parijs
        Args:
            * ag_init : object instance .Agent that starts conversation
            * others : list of agent object instances. Rest of agents that take part in conversation
        Returns:
            * it calls 'vocab_choice_model' method for each agent
        """

        ags = [ag_init]
        ags.extend(others)
        num_ags = len(ags)

        def model_monolang_conv(lang_group, others): # unique group lang only
            ag_init.vocab_choice_model(lang_group, others)
            for ix, ag in enumerate(others):
                if ix < num_ags - 2:
                    ag.vocab_choice_model(lang_group, [ag_init] + others[:ix] + others[ix + 1:])
                else:
                    ag.vocab_choice_model(lang_group, ags[:-1])

        def model_multilang_conv(max_pcts, multilang=True, lang=None, excluded_type=None):
            if multilang:
                ag_init.vocab_choice_model(max_pcts[0], others, long=False)
                for ix, ag in enumerate(others):
                    if ix < num_ags - 2:
                        ag.vocab_choice_model(max_pcts[ix], [ag_init] + others[:ix] + others[ix + 1:], long=False)
                    else:
                        ag.vocab_choice_model(max_pcts[ix], ags[:-1], long=False)
            else: # excluded-type agents are unable to understand lang
                if ag_init.language == excluded_type:
                    pass # no conversation possible if originated by ag_init
                else:
                    ag_init.vocab_choice_model(lang, others, long=False)
                    for ix, ag in enumerate(others):
                        if ix < num_ags - 2:
                            if ags[ix + 1].language == excluded_type:
                                pass # agent unable to speak in only lang everybody partially understands
                            else:
                                ag.vocab_choice_model(lang, [ag_init] + others[:ix] + others[ix + 1:], long=False)
                        else:
                            if ags[ix + 1].language == excluded_type:
                                pass
                            else:
                                ag.vocab_choice_model(lang, ags[:-1], long=False)

        l1_pcts = [ag.lang_stats['L1']['pct'][ag.age] for ag in ags]
        l2_pcts = [ag.lang_stats['L2']['pct'][ag.age] for ag in ags]
        max_pcts = np.argmax([l1_pcts, l2_pcts], axis=0)
        ag_langs = [ag.language for ag in ags]
        # define current case
        if set(ag_langs) in [{0}, {0, 1}]: # TODO: need to save info of how init wanted to talk-> Feedback for AI learning
            lang_group = 0
            model_monolang_conv(lang_group, others)
        elif set(ag_langs) == {1}:
            # simplified PRELIMINARY NEUTRAL assumption: ag_init will start speaking the language they speak best
            # ( TODO : at this stage no modeling of place bias !!!!)
            # who starts conversation matters, but also average lang spoken with already known agents
            lang_init = np.argmax([l1_pcts[0], l2_pcts[0]])
            langs_with_known_agents = [self.model.known_people_network[ag_init][ag]['lang']
                                       for ag in others
                                       if ag in self.model.known_people_network[ag_init]]
            if langs_with_known_agents:
                av_k_lang = round(sum(langs_with_known_agents) / len(langs_with_known_agents))
                lang_group = av_k_lang
            else:
                lang_group = lang_init
            model_monolang_conv(lang_group, others)
        elif set(ag_langs) in [{1, 2}, {2}]:
            lang_group = 1
            model_monolang_conv(lang_group, others)
        else: # monolinguals on both linguistic sides => SHORT CONVERSATION
            # get agents on both lang sides unable to speak in other lang
            idxs_real_monolings_l2 = [idx for idx, pct in enumerate(l1_pcts) if pct < 0.025]
            idxs_real_monolings_l1 = [idx for idx, pct in enumerate(l2_pcts) if pct < 0.025]
            # => All agents partially understand each other langs, but some can't speak l1 and some can't speak l2
            if not idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # each agent picks favorite lang
                model_multilang_conv(max_pcts, multilang=True)
            # some agents only understand and speak l1, while some partially understand but can't speak l1
            elif idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # slight bias towards l1 => conversation in l1 but some speakers unable to speak = > short conversation
                model_multilang_conv(max_pcts, multilang=False, lang=0, excluded_type=2)
            # some agents only understand and speak l2, while some partially understand but can't speak l2
            elif not idxs_real_monolings_l1 and idxs_real_monolings_l2: # some agents don't understand l1
                # slight bias towards l2 => all speak in l2 but some speakers unable to speak
                model_multilang_conv(max_pcts, multilang=False, lang=1, excluded_type=0)
            else: # conversation impossible because there are agents on both lang sides unable to follow other's lang
                pass # DO NOT CALL update

    def get_group_conversation_lang_OLD(self, initiator, rest_of_group):
        """ Method that allows to
            Initiator of conversation is separated from rest_of_group
        """
        ags_lang_profile = [(ag.language, ag.lang_freq['cat_pct_s'], ag) for ag in rest_of_group]

        ags_lang_profile = sorted(ags_lang_profile, key=lambda elem: (elem[0], elem[1]))
        rest_of_group = [tup[2] for tup in ags_lang_profile]

        # map init lang profile
        def fun_map_init(x):
            if x <= 0.2:
                return 0
            elif 0.2 < x <= 0.5:
                return 1
            elif 0.5 < x <= 0.8:
                return 2
            elif x > 0.8:
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
        pass

    def get_words_per_conv(self, long=True, age_1=14, age_2=65):
        """ Computes number of words spoken per conversation for a given age
            If conversation=False, computes average number of words per day,
            assuming 16000 tokens per adult per day as average """
        # TODO : define 3 types of conv: short, average and long ( with close friends??)
        age_1, age_2 = 36 * age_1, 36 * age_2
        real_vocab_size = 40 * self.model.vocab_red
        real_spoken_words_per_day = 16000
        f = real_vocab_size / real_spoken_words_per_day

        if self.age < age_1:
            delta = 0.0001
            decay = -np.log(delta / 100) / (age_1)
            factor =  f + 400 * np.exp(-decay * self.age)
        elif age_1 <= self.age <= age_2:
            factor = f
        else:
            factor = f - 1 + np.exp(0.0005 * (self.age - age_2 ) )
        if long:
            return self.model.num_words_conv[1] / factor
        else:
            return self.model.num_words_conv[0] / factor

    def update_lang_arrays(self, lang, sample_words, speak=True, a=7.6, b=0.023, c=-0.031, d=-0.2,
                           delta_s_factor=0.25, min_mem_times=5, pct_threshold=0.9):
        """ Function to compute and update main arrays that define agent linguistic knowledge
            Args:
                * lang : integer in [0,1] {0:'spa', 1:'cat'}
                * sample_words: tuple of 2 numpy arrays of integers.
                                First array is of word indices, second of word counts
                * speak: boolean. Defines whether agent is speaking or listening
                * a, b, c, d: float parameters to define memory function from SUPERMEMO by Piotr A. Wozniak
                * delta_s_factor: positive float < 1.
                                  Defines increase of mem stability due to passive rehearsal
                                  as a fraction of that due to active rehearsal
                * min_mem_times: positive integer. Minimum number of times
                * pct_threshold: positive float < 1. Value to define percentage lang knowledge.
                                 If retrievability R for a given word is higher than R, the word is considered
                                 as well known. Otherwise, it is not

            MEMORY MODEL: https://www.supermemo.com/articles/stability.htm

            Assumptions ( see "HOW MANY WORDS DO WE KNOW ???" By Marc Brysbaert*,
            Michaël Stevens, Paweł Mandera and Emmanuel Keuleers):
                * ~16000 spoken tokens per day + 16000 heard tokens per day + TV, RADIO
                * 1min reading -> 220-300 tokens with large individual differences, thus
                  in 1 h we get ~ 16000 words """

        act, act_c = sample_words
        # update word counter with newly active words
        self.lang_stats[lang]['wc'][act] += act_c
        # if words are from listening, they might be new to agent
        if not speak:
            ds_factor = delta_s_factor
            # check which words are available for memorization (need minimum number of times)
            mem_availab_words = np.where(self.lang_stats[lang]['wc'] > min_mem_times)[0]
            # compute indices of active words that are available for memory
            idxs_act = np.nonzero(np.in1d(act, mem_availab_words, assume_unique=True))
            # get really activated words and apply indices to counts ( retrieve only counts of really active words)
            act, act_c = act[idxs_act], act_c[idxs_act]
        else:
            ds_factor = 1

        if len(act):  # check that there are any real active words
            # compute increase in memory stability S due to (re)activation
            # TODO : I think it should be dS[reading]  < dS[media_listening]  < dS[listen_in_conv] < dS[speaking]
            delta_S = ds_factor * (a * (self.lang_stats[lang]['S'][act] ** (-b)) * np.exp(c * 100 * self.lang_stats[lang]['R'][act]) + d)
            # update memory stability value
            self.lang_stats[lang]['S'][act] += delta_S
            # discount one to counts
            act_c -= 1
            # Simplification with good approx : we apply delta_S without iteration !!
            delta_S = ds_factor * (act_c * (a * (self.lang_stats[lang]['S'][act] ** (-b)) * np.exp(c * 100 * self.lang_stats[lang]['R'][act]) + d))
            # update
            self.lang_stats[lang]['S'][act] += delta_S

        # define mask to update elapsed-steps array t
        self.day_mask[lang][act] = True
        self.lang_stats[lang]['t'][~self.day_mask[lang]] += 1  # add ones to last activation time counter if word not act
        self.lang_stats[lang]['t'][self.day_mask[lang]] = 0  # set last activation time counter to zero if word act

        # compute memory retrievability R from t, S
        self.lang_stats[lang]['R'] = np.exp(-self.k * self.lang_stats[lang]['t'] / self.lang_stats[lang]['S'])
        self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > pct_threshold)[0].shape[0] /
                                                  self.model.vocab_red)

    def vocab_choice_model(self, lang, listeners, long=True, return_values=False):
        """ Method that models word choice by self agent in a conversation
            Word choice is governed by vocabulary knowledge constraints
            Both self and listeners lang arrays are updated according to the sampled words

            Args:
                * lang: integer in [0, 1] {0:'spa', 1:'cat'}
                * listeners: list of agent instances self is talking to
                * long: boolean that defines conversation length
                * return_values : boolean. If true, chosen words are returned
        """

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
        for ag in listeners:
            ag.update_lang_arrays(lang, spoken_words, speak=False)
        # TODO: NOW NEED MODEL of how to deal with missed words = > L3, emergent lang with mixed vocab ???
        if return_values:
            return spoken_words

    def update_lang_switch(self, switch_threshold=0.1):
        """Switch to a new linguistic regime when threshold is reached
           If language knowledge falls below switch_threshold value, agent
           becomes monolingual"""

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

    def stage_1(self):
        self.speak()

    def stage_2(self):
        self.loc_info['home'].agents_in.remove(self)
        self.move_random()
        self.speak()
        self.move_random()
        #self.listen()

    def stage_3(self):
        if self.age < 720:
            self.model.grid.move_agent(self, self.loc_info['school'].pos)
            self.loc_info['school'].agents_in.add(self)
            self.study_lang(0)
            self.study_lang(1)
            self.speak()
        else:
            if self.loc_info['job']:
                self.model.grid.move_agent(self, self.loc_info['job'].pos)
                self.loc_info['job'].agents_in.add(self)
                # TODO speak to people in job !!!
                self.speak()
                self.speak()
            else:
                self.look_for_job()
                self.speak()

    def stage_4(self):
        self.move_random()
        self.speak()
        self.model.grid.move_agent(self, self.loc_info['home'].pos)
        self.loc_info['home'].agents_in.add(self)
        try:
            for key in self.model.family_network[self]:
                if key.pos == self.loc_info['home'].pos:
                    lang = self.model.family_network[self][key]['lang']
                    self.speak(with_agent=key, lang=lang)
        except:
            pass
        # memory becomes ever shakier after turning 65...
        if self.age > 65 * 36:
            for lang in ['L1', 'L2']:
                self.lang_stats[lang]['S'] = np.where(self.lang_stats[lang]['S'] >= 0.01,
                                                      self.lang_stats[lang]['S'] - 0.01,
                                                      0.000001)
        # Update at the end of each step
        # for lang in ['L1', 'L2']:
        #     self.lang_stats[lang]['pct'][self.age] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
        #                                               self.model.vocab_red)

    def __repr__(self):
        return 'Lang_Agent_{0.unique_id!r}'.format(self)



class Home:
    def __init__(self, pos):
        self.pos=pos
        self.agents_in = set()
    def __repr__(self):
        return 'Home_{0.pos!r}'.format(self)

class School:
    def __init__(self, pos, num_places):
        self.pos=pos
        self.num_free=num_places
        self.agents_in = set()
    def __repr__(self):
        return 'School_{0.pos!r}'.format(self)

class Job:
    def __init__(self, pos, num_places, skill_level=0):
        self.pos=pos
        self.num_places=num_places
        self.skill_level = skill_level
        self.agents_in = set()
    def __repr__(self):
        return 'Job{0.pos!r}'.format(self)