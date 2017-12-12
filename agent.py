# IMPORT LIBS
import random
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from collections import defaultdict
from itertools import groupby
from math import ceil
import string

#import private library to model lang zipf CDF
from zipf_generator import Zipf_Mand_CDF_compressed, randZipf


class BaseAgent:
    """ Basic agent class that contains attributes and methods common to all lang agents subclasses"""

    # define memory retrievability constant
    k = np.log(10 / 9)

    def __init__(self, model, unique_id, language, sex, age=0, home=None, school=None, city_idx=None,
                 lang_act_thresh=0.1, lang_passive_thresh=0.025, import_ic=False):
        """ language => 0, 1, 2 => spa, bil, cat  """
        self.model = model
        self.unique_id = unique_id
        self.info = {'age': age, 'language': language, 'sex': sex}
        self.loc_info = {'home': home, 'city_idx': city_idx, 'school': school}
        self.lang_thresholds = {'speak': lang_act_thresh, 'understand': lang_passive_thresh}

        # define container for languages' tracking and statistics
        self.lang_stats = defaultdict(lambda: defaultdict(dict))
        self.day_mask = {l: np.zeros(self.model.vocab_red, dtype=np.bool)
                         for l in ['L1', 'L12', 'L21', 'L2']}
        if import_ic:
            self.set_lang_ics()
        else:
            # set null knowledge in all possible langs
            for lang in ['L1', 'L12', 'L21', 'L2']:
                self._set_null_lang_attrs(lang)

    def _set_lang_attrs(self, lang, pct_key):
        """ Private method that sets agent linguistic status for a GIVEN AGE
            Args:
                * lang: string. It can take two different values: 'L1' or 'L2'
                * pct_key: string. It must be of the form '%_pct' with % an integer
                  from following list [10,25,50,75,90,100]. ICs are not available for every single level
        """
        # numpy array(shape=vocab_size) that counts elapsed steps from last activation of each word
        self.lang_stats[lang]['t'] = np.copy(self.model.lang_ICs[pct_key]['t'][self.info['age']])
        # S: numpy array(shape=vocab_size) that measures memory stability for each word
        self.lang_stats[lang]['S'] = np.copy(self.model.lang_ICs[pct_key]['S'][self.info['age']])
        # compute R from t, S (R defines retrievability of each word)
        self.lang_stats[lang]['R'] = np.exp(- self.k *
                                                  self.lang_stats[lang]['t'] /
                                                  self.lang_stats[lang]['S']
                                                  ).astype(np.float64)
        # word counter
        self.lang_stats[lang]['wc'] = np.copy(self.model.lang_ICs[pct_key]['wc'][self.info['age']])
        # vocab pct
        self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
        self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                  self.model.vocab_red)

    def _set_null_lang_attrs(self, lang, S_0=0.01, t_0=1000):
        """Private method that sets null linguistic knowledge in specified language, i.e. no knowledge
           at all of it
           Args:
               * lang: string. It can take two different values: 'L1' or 'L2'
               * S_0: float. Initial value of memory stability
               * t_0: integer. Initial value of time-elapsed ( in days) from last time words were encountered
        """
        self.lang_stats[lang]['S'] = np.full(self.model.vocab_red, S_0, dtype=np.float)
        self.lang_stats[lang]['t'] = np.full(self.model.vocab_red, t_0, dtype=np.float)
        self.lang_stats[lang]['R'] = np.exp(- self.k *
                                            self.lang_stats[lang]['t'] /
                                            self.lang_stats[lang]['S']
                                            ).astype(np.float32)
        self.lang_stats[lang]['wc'] = np.zeros(self.model.vocab_red)
        self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
        self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                          self.model.vocab_red)

    def set_lang_ics(self, S_0=0.01, t_0=1000, biling_key=None):
        """ set agent's linguistic Initial Conditions by calling set up methods
        Args:
            * S_0: float <= 1. Initial memory intensity
            * t_0: last-activation days counter
            * biling_key: integer from [10, 25, 50, 75, 90, 100]. Specify only if
              specific bilingual level is needed as input
        """
        if self.info['language'] == 0:
            self._set_lang_attrs('L1', '100_pct')
            self._set_null_lang_attrs('L2', S_0, t_0)
        elif self.info['language'] == 2:
            self._set_null_lang_attrs('L1', S_0, t_0)
            self._set_lang_attrs('L2', '100_pct')
        else: # BILINGUAL
            if not biling_key:
                biling_key = np.random.choice(self.model.ic_pct_keys)
            L1_key = str(biling_key) + '_pct'
            L2_key = str(100 - biling_key) + '_pct'
            for lang, key in zip(['L1', 'L2'], [L1_key, L2_key]):
                self._set_lang_attrs(lang, key)
        # always null conditions for transition languages
        self._set_null_lang_attrs('L12', S_0, t_0)
        self._set_null_lang_attrs('L21', S_0, t_0)

    def listen(self, to_agent=None, min_age_interlocs=None, num_days=10):
        """
            Method to listen to conversations, media, etc... and update corresponding vocabulary
            Args:
                * to_agent: class instance (optional). It can be either a language agent or a media agent.
                    If not specified, self agent will listen to a random conversation taking place on his cell
                * min_age_interlocs: integer. Allows to adapt vocabulary pick to the youngest
                    participant in the conversation
                * num_days:
        """
        if not to_agent:
            # get all agents currently placed on chosen cell
            others = self.model.grid.get_cell_list_contents(self.pos)
            others.remove(self)
            # if two or more agents in cell, conversation is possible
            if len(others) >= 2:
                ag_1, ag_2 = np.random.choice(others, size=2, replace=False)
                # call run conversation with bystander
                self.model.run_conversation(ag_1, ag_2, bystander=self)
        else:
            # make other agent speak and 'self' agent get the listened vocab
            if to_agent in self.model.known_people_network[self]:
                lang = self.model.known_people_network[self][to_agent]['lang']
            else:
                lang = self.model.define_lang_interaction(self, to_agent)
            words = to_agent.pick_vocab(lang, long=False, min_age_interlocs=min_age_interlocs,
                                        num_days=num_days)

            for lang_key, lang_words in words:
                words = np.intersect1d(self.model.cdf_data['s'][self.info['age']],
                                                  lang_words, assume_unique=True)
                self.update_lang_arrays(words, speak=False)


        # TODO : implement 'listen to media' option

    def update_lang_arrays(self, sample_words, speak=True, a=7.6, b=0.023, c=-0.031, d=-0.2,
                           delta_s_factor=0.25, min_mem_times=5, pct_threshold=0.9, pct_threshold_und=0.1):
        """ Function to compute and update main arrays that define agent linguistic knowledge
            Args:
                * sample_words: dict where keys are lang labels and values are tuples of
                    2 NumPy integer arrays. First array is of conversation-active unique word indices,
                    second is of corresponding counts of those words
                * speak: boolean. Defines whether agent is speaking or listening
                * a, b, c, d: float parameters to define memory function from SUPERMEMO by Piotr A. Wozniak
                * delta_s_factor: positive float < 1.
                    Defines increase of mem stability due to passive rehearsal
                    as a fraction of that due to active rehearsal
                * min_mem_times: positive integer. Minimum number of times to start remembering a word.
                    It should be understood as average time it takes for a word meaning to be incorporated
                    in memory
                * pct_threshold: positive float < 1. Value to define lang knowledge in percentage.
                    If retrievability R for a given word is higher than pct_threshold,
                    the word is considered as well known. Otherwise, it is not
                * pct_threshold_und : positive float < 1. If retrievability R for a given word
                    is higher than pct_threshold, the word can be correctly understood.
                    Otherwise, it cannot


            MEMORY MODEL: https://www.supermemo.com/articles/stability.htm

            Assumptions ( see "HOW MANY WORDS DO WE KNOW ???" By Marc Brysbaert*,
            Michaël Stevens, Paweł Mandera and Emmanuel Keuleers):
                * ~16000 spoken tokens per day + 16000 heard tokens per day + TV, RADIO
                * 1min reading -> 220-300 tokens with large individual differences, thus
                  in 1 h we get ~ 16000 words
        """

        # TODO: need to define similarity matrix btw language vocabularies?
        # TODO: cat-spa have dist mean ~ 2 and std ~ 1.6 ( better to normalize distance ???)
        # TODO for cat-spa np.random.choice(range(7), p=[0.05, 0.2, 0.45, 0.15, 0.1, 0.025,0.025], size=500)
        # TODO 50% of unknown words with edit distance == 1 can be understood, guessed

        for lang, (act, act_c) in sample_words.items():
            # UPDATE WORD COUNTING +  preprocessing for S, t, R UPDATE
            # If words are from listening, they might be new to agent
            # ds_factor value will depend on action type (speaking or listening)
            if not speak:
                known_words = np.nonzero(self.lang_stats[lang]['R'] > pct_threshold_und)
                # boolean mask of known active words
                known_act_bool = np.in1d(act, known_words, assume_unique=True)
                if np.all(known_act_bool):
                    # all active words in conversation are known
                    # update all active words
                    self.lang_stats[lang]['wc'][act] += act_c
                else:
                    # some heard words are unknown. Find them in 'act' words vector base
                    unknown_act_bool = np.invert(known_act_bool)
                    unknown_act , unknown_act_c = act[unknown_act_bool], act_c[unknown_act_bool]
                    # Select most frequent unknown word
                    ix_most_freq_unk = np.argmax(unknown_act_c)
                    most_freq_unknown = unknown_act[ix_most_freq_unk]
                    # update most active unknown word's count (the only one actually grasped)
                    self.lang_stats[lang]['wc'][most_freq_unknown] += unknown_act_c[ix_most_freq_unk]
                    # update known active words count
                    self.lang_stats[lang]['wc'][act[known_act_bool]] += act_c[known_act_bool]
                    # get words available for S, t, R update
                    if self.lang_stats[lang]['wc'][most_freq_unknown] > min_mem_times:
                        known_act_ixs = np.concatenate((np.nonzero(known_act_bool)[0], [ix_most_freq_unk]))
                        act, act_c = act[known_act_ixs ], act_c[known_act_ixs ]
                    else:
                        act, act_c = act[known_act_bool], act_c[known_act_bool]
                ds_factor = delta_s_factor
            else:
                # update word counter with newly active words
                self.lang_stats[lang]['wc'][act] += act_c
                ds_factor = 1
            # check if there are any words for S, t, R update
            if act.size:
                # compute increase in memory stability S due to (re)activation
                # TODO : I think it should be dS[reading]  < dS[media_listening]  < dS[listen_in_conv] < dS[speaking]
                S_act_b = self.lang_stats[lang]['S'][act] ** (-b)
                R_act = self.lang_stats[lang]['R'][act]
                delta_S = ds_factor * (a * S_act_b * np.exp(c * 100 * R_act) + d)
                # update memory stability value
                self.lang_stats[lang]['S'][act] += delta_S
                # discount one to counts
                act_c -= 1
                # Simplification with good approx : we apply delta_S without iteration !!
                S_act_b = self.lang_stats[lang]['S'][act] ** (-b)
                R_act = self.lang_stats[lang]['R'][act]
                delta_S = ds_factor * (act_c * (a * S_act_b * np.exp(c * 100 * R_act) + d))
                # update
                self.lang_stats[lang]['S'][act] += delta_S
                # update daily boolean mask to update elapsed-steps array t
                self.day_mask[lang][act] = True
                # set last activation time counter to zero if word act
                self.lang_stats[lang]['t'][self.day_mask[lang]] = 0
                # compute new memory retrievability R and current lang_knowledge from t, S values
                self.lang_stats[lang]['R'] = np.exp(-self.k * self.lang_stats[lang]['t'] / self.lang_stats[lang]['S'])
                self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > pct_threshold)[0].shape[0] /
                                                          self.model.vocab_red)

    def grow(self, new_class):
        """ It replaces current agent with a new agent subclass instance.
            It removes current instance from all networks, lists, sets in model and adds new instance
            to them instead.
            Args:
                * new_class: class. Agent class that will replace the current one
        """
        grown_agent = new_class(self.model, self.unique_id, self.info['language'], self.info['age'])
        # copy all current instance attributes to new agent instance
        for key, val in self.__dict__.items():
            setattr(grown_agent, key, val)
        # relabel network nodes
        relabel_key = {self: grown_agent}
        for network in [self.model.family_network,
                        self.model.known_people_network,
                        self.model.friendship_network]:
            try:
                nx.relabel_nodes(network, relabel_key, copy=False)
            except nx.NetworkXError:
                continue
        # remove agent from all locations where it belongs to
        for loc, attr in zip([self.loc_info['home'], self.loc_info['job'], self.loc_info['school']],
                             ['occupants', 'employees', 'students']):
            try:
                getattr(loc, attr).remove(self)
                getattr(loc, attr).add(grown_agent)
                loc.agents_in.remove(self)
                loc.agents_in.add(grown_agent)
            except:
                continue
        # remove old instance and add new one to city
        self.model.clusters_info[self.loc_info['city_idx']]['agents'].remove(self)
        self.model.clusters_info[self.loc_info['city_idx']]['agents'].add(grown_agent)
        # remove agent from grid and schedule
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)
        # add new_agent to grid and schedule
        self.model.grid.place_agent(grown_agent, self.pos)
        self.model.schedule.add(grown_agent)

    def random_death(self, a=1.23368173e-05, b=2.99120806e-03, c=3.19126705e+01):
        """ Method to randomly determine agent death or survival at each step
            The fitted function provides the death likelihood for a given rounded age
            In order to get the death-probability per step we divide by number of steps in a year (36)
            Fitting parameters are from https://www.demographic-research.org/volumes/vol27/20/27-20.pdf
            'Smoothing and projecting age-specific probabilities of death by TOPALS' by Joop de Beer
            Resulting life expectancy is 77 years and std is ~ 15 years
        """
        if random.random() < a * (np.exp(b * self.info['age']) + c) / 36:
            self.model.remove_after_death(self)


class Baby(BaseAgent): # from 0 to 2

    def __init__(self, model, unique_id, language, sex, age=0, home=None, school=None, city_idx=None,
                 lang_act_thresh=0.1, lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, school, city_idx, lang_act_thresh,
                         lang_passive_thresh, import_ic)

        self.info['close_family'] = {}
        self.info['close_family']['mother'] = [ag for ag in self.model.family_network[self]
                                               if self.model.family_network[self][ag]['fam_link'] == 'mother'][0]
        self.info['close_family']['father'] = [ag for ag in self.model.family_network[self]
                                               if self.model.family_network[self][ag]['fam_link'] == 'father'][0]
        self.info['close_family']['siblings'] = [ag for ag in self.model.family_network[self]
                                                 if self.model.family_network[self][ag]['fam_link'] == 'sibling']
        #TODO modify self.info['teacher'] = random.choice(self.loc_info['school'].info['teachers'])

    def get_conversation_lang(self):
        """ Adapt to baby exceptions"""
        pass

    def set_family_links(self, father, mother, lang_with_father, lang_with_mother):
        """ Method to define family links and interaction language of newborn baby """
        self.model.family_network.add_edge(father, self, fam_link='child', lang=lang_with_father)
        self.model.family_network.add_edge(self, father, fam_link='father', lang=lang_with_father)
        self.model.family_network.add_edge(mother, self, fam_link='child', lang=lang_with_mother)
        self.model.family_network.add_edge(self, father, fam_link='mother', lang=lang_with_mother)
        # rest of family will if possible speak same language with baby as their link agent to the baby
        for elder, lang in zip([father, mother], [lang_with_father, lang_with_mother]):
            for agent, labels in self.model.family_network[elder].items():
                lang_labels = ['L1', 'L12'] if lang == 0 else ['L2', 'L21']
                if True in [agent.lang_stats[l]['pct'][agent.info['age']] > agent.lang_act_thresh
                            for l in lang_labels]:
                    com_lang = lang
                else:
                    com_lang = 1 if lang == 0 else 0
                if labels["fam_link"] == 'father':
                    self.model.family_network.add_edge(self, agent, fam_link='grandfather',
                                                  lang=com_lang )
                    self.model.family_network.add_edge(agent, self, fam_link='grandchild',
                                                  lang=com_lang)
                elif labels["fam_link"] == 'mother':
                    self.model.family_network.add_edge(self, agent, fam_link='grandmother', lang=com_lang)
                    self.model.family_network.add_edge(agent, self, fam_link='grandchild', lang=com_lang)
                elif labels["fam_link"] == 'sibling':
                    self.model.family_network.add_edge(self, agent,
                                                       fam_link='uncle' if agent.info['sex'] == 'M' else 'aunt',
                                                       lang=com_lang)
                    self.model.family_network.add_edge(agent, self, fam_link='nephew', lang=com_lang)
                    if agent.info['married']:
                        consort = [key for key, value
                                   in self.model.family_network[agent].items()
                                   if value['fam_link'] == 'consort'][0]
                        self.model.family_network.add_edge(self, consort,
                                                           fam_link ='uncle' if consort.info['sex'] == 'M' else 'aunt')
                        self.model.family_network.add_edge(consort, self, fam_link='nephew', lang=com_lang)
                elif labels["fam_link"] == 'nephew':
                    self.model.family_network.add_edge(self, agent, fam_link='cousin', lang=com_lang)
                    self.model.family_network.add_edge(agent, self, fam_link='cousin', lang=com_lang)

    def stage_1(self):
        # listen to close family at home
        if self.age > 36:
            for ag in [ag for ag in self.loc_info['home'].agents_in.difference({self})
                       if ag.info['age'] > 72]:
                self.listen(to_agent=ag, min_age_interlocs=self.age)

    def stage_2(self):
        # go to daycare with mom or dad
        if self.age > 36:
            school = self.loc_info['school']
            self.model.grid.move_agent(self, school.pos)
            mother = self.info['close_family']['mother']
            self.listen(to_agent=mother, min_age_interlocs=self.age)
            # TODO : talk to children teacher
            school.agents_in.add(self)
            school_ags = np.random.choice(self.loc_info['school'].agents_in, size=3)
            for ag in school_ags:
                self.listen(to_agent=ag)
            teacher = random.choice(school.info['teachers'])
            self.listen(to_agent=teacher)

    def stage_3(self):
        if self.age > 36:
            mother = self.info['close_family']['mother']
            school = self.loc_info['school']
            parents = [p.info['close_family']['mother']
                       if random.random() > 0.4 else p.info['close_family']['father']
                       for p in school.info['students'] if p['mother'].pos == school.pos]
            if parents:
                self.model.grid.move_agent(mother, school.pos)
                num_peop = random.randint(1, min(len(parents), 4))
                mother.start_conversation(with_agents = random.sample(parents, num_peop))
            for stud in school.info['students']:
                if stud.info['close_family']['mother']:
                    pass

            self.listen(to_agent=mother, min_age_interlocs=self.age)

    def stage_4(self):
        if self.age > 36:
            for ag in [ag for ag in self.loc_info['home'].agents_in.difference({self})
                       if ag.info['age'] > 72]:
                self.listen(to_agent=ag, min_age_interlocs=self.age)
        if self.info['age'] == 36 * 2:
            self.grow(Child)

class Child(BaseAgent): #from 2 to 10

    def __init__(self, model, unique_id, language, sex, age=0, home=None, school=None, city_idx=None,
                 lang_act_thresh=0.1, lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, school, city_idx, lang_act_thresh,
                         lang_passive_thresh, import_ic)

    def pick_vocab(self, lang, long=True, min_age_interlocs=None,
                   biling_interloc=False, num_days=10):
        """ Method that models word choice by self agent in a conversation
            Word choice is governed by vocabulary knowledge constraints
            Args:
                * lang: integer in [0, 1] {0:'spa', 1:'cat'}
                * long: boolean that defines conversation length
                * min_age_interlocs: integer. The youngest age among all interlocutors, EXPRESSED IN STEPS.
                    It is used to modulate conversation vocabulary to younger agents
                * biling_interloc : boolean. If True, speaker word choice might be mixed, since
                    he/she is certain interlocutor will understand
                * num_days : integer [1, 10]. Number of days in one 10day-step this kind of speech is done
            Output:
                * spoken words: dict where keys are lang labels and values are lists with words spoken
                    in lang key and corresponding counts
        """

        # TODO: VERY IMPORTANT -> Model language switch btw bilinguals, reflecting easiness of retrieval

        # TODO : model 'Grammatical foreigner talk' =>
        # TODO : how word choice is adapted by native speakers when speaking to adult learners
        # TODO: NEED MODEL of how to deal with missed words = > L12 and L21, emergent langs with mixed vocab ???

        # sample must come from AVAILABLE words in R ( retrievability) !!!! This can be modeled in following STEPS

        # 1. First sample from lang CDF ( that encapsulates all to-be-known concepts at a given age-step)
        # These are the thoughts that speaker tries to convey
        # TODO : VI BETTER IDEA. Known thoughts are determined by UNION of all words known in L1 + L12 + L21 + L2
        num_words = int(self.get_num_words_per_conv(long) * num_days)
        if min_age_interlocs:
            word_samples = randZipf(self.model.cdf_data['s'][min_age_interlocs], num_words)
        else:
            word_samples = randZipf(self.model.cdf_data['s'][self.info['age']], num_words)

        # get unique words and counts
        bcs = np.bincount(word_samples)
        act, act_c = np.where(bcs > 0)[0], bcs[bcs > 0]
        # SLOWER BUT CLEANER APPROACH =>  act, act_c = np.unique(word_samples, return_counts=True)

        # check if conversation is btw bilinguals and therefore lang switch is possible
        # TODO : key is that most biling agents will not express surprise to lang mixing or switch
        # TODO : whereas monolinguals will. Feedback for correction
        # if biling_interloc:
        #     if lang == 'L1':
        #         # find all words among 'act' words where L2 retrievability is higher than L1 retrievability
        #         # Then fill L12 container with a % of those words ( their knowledge is known at first time of course)
        #
        #         self.lang_stats['L1']['R'][act] <= self.lang_stats['L2']['R'][act]
        #         L2_strongest = self.lang_stats['L2']['R'][act] == 1.
                #act[L2_strongest]

                # rand_info_access = np.random.rand(4, len(act))
                # mask_L1 = rand_info_access[0] <= self.lang_stats['L1']['R'][act]
                # mask_L12 = rand_info_access[1] <= self.lang_stats['L12']['R'][act]
                # mask_L21 = rand_info_access[2] <= self.lang_stats['L21']['R'][act]
                # mask_L2 = rand_info_access[3] <= self.lang_stats['L2']['R'][act]

        # 2. Given a lang, pick the variant that is most familiar to agent
        lang = 'L1' if lang == 0 else 'L2'
        if lang == 'L1':
            pct1, pct2 = self.lang_stats['L1']['pct'][self.info['age']] , self.lang_stats['L12']['pct'][self.info['age']]
            lang = 'L1' if pct1 >= pct2 else 'L12'
        elif lang == 'L2':
            pct1, pct2 = self.lang_stats['L2']['pct'][self.info['age']], self.lang_stats['L21']['pct'][self.info['age']]
            lang = 'L2' if pct1 >= pct2 else 'L21'

        # 3. Then assess which sampled words-concepts can be successfully retrieved from memory
        # get mask for words successfully retrieved from memory
        mask_R = np.random.rand(len(act)) <= self.lang_stats[lang]['R'][act]
        spoken_words = {lang:[act[mask_R], act_c[mask_R]]}
        # if there are missing words-concepts, they might be found in the other known language(s)
        if np.count_nonzero(mask_R) < len(act):
            if lang in ['L1', 'L12']:
                lang2 = 'L12' if lang == 'L1' else 'L1'
            elif lang in ['L2', 'L21']:
                lang2 = 'L21' if lang == 'L2' else 'L2'
            mask_R2 = np.random.rand(len(act[~mask_R])) <= self.lang_stats[lang2]['R'][act[~mask_R]]
            if act[~mask_R][mask_R2].size:
                spoken_words.update({lang2:[act[~mask_R][mask_R2], act_c[~mask_R][mask_R2]]})
            # if still missing words, check in last lang available
            if (act[mask_R].size + act[~mask_R][mask_R2].size) < len(act):
                lang3 = 'L2' if lang2 in ['L12', 'L1'] else 'L1'
                rem_words = act[~mask_R][~mask_R2]
                mask_R3 = np.random.rand(len(rem_words)) <= self.lang_stats[lang3]['R'][rem_words]
                if rem_words[mask_R3].size:
                    # VERY IMP: add to transition language instead of 'pure' one.
                    # This is the process of creation/adaption/translation
                    tr_lang = max([lang, lang2], key=len)
                    spoken_words.update({lang2: [rem_words[mask_R3], act_c[~mask_R][~mask_R2][mask_R3]]})

        return spoken_words


    def stage_1(self):
        pass

    def stage_2(self):
        pass

    def stage_3(self):
        pass

    def stage_4(self):
        if self.info['age'] == 36 * 10:
            self.grow(Adolescent)

class Adolescent(Child): # from 10 to 18

    def __init__(self, model, unique_id, language, sex, age=0, home=None, school=None, city_idx=None,
                 lang_act_thresh=0.1, lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, school, city_idx, lang_act_thresh,
                         lang_passive_thresh, import_ic)

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

    def stage_1(self):
        pass

    def stage_2(self):
        pass

    def stage_3(self):
        pass

    def stage_4(self):
        if self.info['age'] == 36 * 18:
            self.grow(Young)


class Young(Adolescent): # from 18 to 30

    def __init__(self, model, unique_id, language, sex, age=0, home=None, school=None, city_idx=None,
                 lang_act_thresh=0.1, lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, school, city_idx, lang_act_thresh,
                         lang_passive_thresh, import_ic)
        self.info['married'] = married
        self.info['num_children'] = num_children
        self.loc_info['job'] = job
        self.loc_info['university'] = None
        self.info['close_family']['consort'] = [ag for ag in self.model.family_network[self]
                                                if self.model.family_network[self][ag]['fam_link'] == 'consort'][0]
        self.info['close_family']['children'] = [ag for ag in self.model.family_network[self]
                                                 if self.model.family_network[self][ag]['fam_link'] == 'child']


    def get_partner(self):
        """ Check lang distance is not > 1"""
        pass

    def reproduce(self, day_prob=0.001):
        # TODO : check appropriate day_prob . Shouldn'it be 0.0015 ? to have > 1 child/per parent
        if (self.info['num_children'] < 4) and (self.info['married']) and (random.random() < day_prob):
            id_ = self.model.set_available_ids.pop()

            consort = [agent for agent, labels in self.model.family_network[self].items()
                       if labels["fam_link"] == 'consort'][0]
            # find out baby-agent language
            lang_consorts, pcts = self.model.define_lang_interaction(self, consort, ret_pcts=True)
            l1, l2 = np.random.choice([0, 1], p=pcts[:2]), np.random.choice([0, 1], p=pcts[2:])
            lang_with_father = l1 if self.info['sex'] == 'M' else l2
            lang_with_mother = l2 if self.info['sex'] == 'M' else l1

            if [lang_with_father, lang_with_mother] in [0, 0]:
                lang = 0
            elif [lang_with_father, lang_with_mother] in [[0, 1], [1, 0]]:
                lang = 1
            elif [lang_with_father, lang_with_mother] in [1, 1]:
                lang = 2

            # find closest school to parent home
            city_idx = self.loc_info['city_idx']
            clust_schools_coords = [sc.pos for sc in self.model.clusters_info[city_idx]['schools']]
            closest_school_idx = np.argmin([pdist([self.loc_info['home'].pos, sc_coord])
                                            for sc_coord in clust_schools_coords])
            # instantiate agent
            sex = 'M' if random.random() > 0.5 else 'F'
            a = Baby(self.model, id_, lang, sex, home=self.loc_info['home'],
                     school=closest_school_idx, city_idx=self.loc_info['city_idx'])
            # Add agent to model
            self.model.add_agent_to_grid_sched_networks(a)
            # Update num of children for both self and consort
            self.info['num_children'] += 1
            consort.info['num_children'] += 1

            # set up family links
            if self.info['sex'] == 'M':
                a.set_family_links(self, consort, lang_with_father, lang_with_mother)
            else:
                a.set_family_links(consort, self, lang_with_father, lang_with_mother)

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


    def simulate_random_death(self, age_1=20, age_2=75, age_3=90, prob_1=0.25, prob_2=0.7):  ##
        # transform ages to steps
        age_1, age_2, age_3 = age_1 * 36, age_2 * 36, age_3 * 36
        # define stochastic probability of agent death as function of age
        if (self.info['age'] > age_1) and (self.info['age'] <= age_2):
            if random.random() < prob_1 / (age_2 - age_1):  # 25% pop will die through this period
                self.remove_after_death()
        elif (self.info['age'] > age_2) and (self.info['age'] < age_3):
            if random.random() < prob_2 / (age_3 - age_2):  # 70% will die
                self.remove_after_death()
        elif self.info['age'] >= age_3:
            self.remove_after_death()

    def look_for_job(self):
        # loop through shuffled job centers list until a job is found
        np.random.shuffle(self.model.clusters_info[self.loc_info['city_idx']]['jobs'])
        for job_c in self.model.clusters_info[self.loc_info['city_idx']]['jobs']:
            if job_c.num_places:
                job_c.num_places -= 1
                self.loc_info['job'] = job_c
                break


    def stage_1(self):
        pass

    def stage_2(self):
        pass

    def stage_3(self):
        pass

    def stage_4(self):
        if self.info['age'] == 36 * 30:
            self.grow(Adult)

class Adult(Young): # from 30 to 65

    def __init__(self, model, unique_id, language, sex, age=0, home=None, school=None, city_idx=None,
                 lang_act_thresh=0.1, lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, school, city_idx, lang_act_thresh,
                         lang_passive_thresh, import_ic)


    def reproduce(self, day_prob=0.005):
        if self.info['age'] < 40 * 36:
            pass  # try chance

    def stage_1(self):
        pass

    def stage_2(self):
        pass

    def stage_3(self):
        pass

    def stage_4(self):
        if self.info['age'] == 36 * 65:
            self.grow(Pensioner)

class Pensioner(Adult): # from 65 to death

    def __init__(self, model, unique_id, language, sex, age=0, home=None, school=None, city_idx=None,
                 lang_act_thresh=0.1, lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, school, city_idx, lang_act_thresh,
                         lang_passive_thresh, import_ic)

    def stage_1(self):
        pass

    def stage_2(self):
        pass

    def stage_3(self):
        pass

    def stage_4(self):
        pass

class LanguageAgent:

    #define memory retrievability constant
    k = np.log(10 / 9)

    def __init__(self, model, unique_id, language, lang_act_thresh=0.1, lang_passive_thresh=0.025, age=0,
                 num_children=0, ag_home=None, ag_school=None, ag_job=None, city_idx=None, import_IC=False):
        self.model = model
        self.unique_id = unique_id
        # language values: 0, 1, 2 => spa, bil, cat
        self.info = {'language': language, 'age': age, 'num_children': num_children} # TODO : group marital/parental info in dict ??
        self.lang_thresholds = {'speak': lang_act_thresh, 'understand': lang_passive_thresh}
        self.loc_info = {'home': ag_home, 'school': ag_school, 'job': ag_job, 'city_idx': city_idx}

        # define container for languages' tracking and statistics
        self.lang_stats = defaultdict(lambda:defaultdict(dict))
        self.day_mask = {l: np.zeros(self.model.vocab_red, dtype=np.bool)
                         for l in ['L1', 'L12', 'L21', 'L2']}
        if import_IC:
            self.set_lang_ics()
        else:
            # set null knowledge in all possible langs
            for lang in ['L1', 'L12', 'L21', 'L2']:
                self._set_null_lang_attrs(lang)

    def _set_lang_attrs(self, lang, pct_key):
        """ Private method that sets agent linguistic status for a GIVEN AGE
            Args:
                * lang: string. It can take two different values: 'L1' or 'L2'
                * pct_key: string. It must be of the form '%_pct' with % an integer
                  from following list [10,25,50,75,90,100]. ICs are not available for every single level
        """
        # numpy array(shape=vocab_size) that counts elapsed steps from last activation of each word
        self.lang_stats[lang]['t'] = np.copy(self.model.lang_ICs[pct_key]['t'][self.info['age']])
        # S: numpy array(shape=vocab_size) that measures memory stability for each word
        self.lang_stats[lang]['S'] = np.copy(self.model.lang_ICs[pct_key]['S'][self.info['age']])
        # compute R from t, S (R defines retrievability of each word)
        self.lang_stats[lang]['R'] = np.exp(- self.k *
                                                  self.lang_stats[lang]['t'] /
                                                  self.lang_stats[lang]['S']
                                                  ).astype(np.float64)
        # word counter
        self.lang_stats[lang]['wc'] = np.copy(self.model.lang_ICs[pct_key]['wc'][self.info['age']])
        # vocab pct
        self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
        self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                  self.model.vocab_red)

    def _set_null_lang_attrs(self, lang, S_0=0.01, t_0=1000):
        """Private method that sets null linguistic knowledge in specified language, i.e. no knowledge
           at all of it
           Args:
               * lang: string. It can take two different values: 'L1' or 'L2'
               * S_0: float. Initial value of memory stability
               * t_0: integer. Initial value of time-elapsed ( in days) from last time words were encountered
        """
        self.lang_stats[lang]['S'] = np.full(self.model.vocab_red, S_0, dtype=np.float)
        self.lang_stats[lang]['t'] = np.full(self.model.vocab_red, t_0, dtype=np.float)
        self.lang_stats[lang]['R'] = np.exp(- self.k *
                                            self.lang_stats[lang]['t'] /
                                            self.lang_stats[lang]['S']
                                            ).astype(np.float32)
        self.lang_stats[lang]['wc'] = np.zeros(self.model.vocab_red)
        self.lang_stats[lang]['pct'] = np.zeros(3600, dtype=np.float64)
        self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                  self.model.vocab_red)

    def set_lang_ics(self, S_0=0.01, t_0=1000, biling_key=None):
        """ set agent's linguistic Initial Conditions by calling set up methods
        Args:
            * S_0: float <= 1. Initial memory intensity
            * t_0: last-activation days counter
            * biling_key: integer from [10, 25, 50, 75, 90, 100]. Specify only if
              specific bilingual level is needed as input
        """
        if self.info['language'] == 0:
            self._set_lang_attrs('L1', '100_pct')
            self._set_null_lang_attrs('L2', S_0, t_0)
        elif self.info['language'] == 2:
            self._set_null_lang_attrs('L1', S_0, t_0)
            self._set_lang_attrs('L2', '100_pct')
        else: # BILINGUAL
            if not biling_key:
                biling_key = np.random.choice(self.model.ic_pct_keys)
            L1_key = str(biling_key) + '_pct'
            L2_key = str(100 - biling_key) + '_pct'
            for lang, key in zip(['L1', 'L2'], [L1_key, L2_key]):
                self._set_lang_attrs(lang, key)
        # always null conditions for transition languages
        self._set_null_lang_attrs('L12', S_0, t_0)
        self._set_null_lang_attrs('L21', S_0, t_0)

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

    def reproduce(self, age_1=20, age_2=40, repr_prob_per_step=0.005):
        """ Method to generate a new agent out of self agent under certain age conditions
            Args:
                * age_1: integer. Lower age bound of reproduction period
                * age_2: integer. Higher age bound of reproduction period
        """
        age_1, age_2 = age_1 * 36, age_2 * 36
        # check reproduction conditions
        if (age_1 <= self.info['age'] <= age_2) and (self.info['num_children'] < 1) and (random.random() < repr_prob_per_step):
            id_ = self.model.set_available_ids.pop()
            lang = self.info['language']
            # find closest school to parent home
            city_idx = self.loc_info['city_idx']
            clust_schools_coords = [sc.pos for sc in self.model.clusters_info[city_idx]['schools']]
            closest_school_idx = np.argmin([pdist([self.loc_info['home'].pos, sc_coord])
                                            for sc_coord in clust_schools_coords])
            # instantiate new agent
            a = LanguageAgent(self.model, id_, lang, ag_home=self.loc_info['home'],
                                      ag_school=self.model.clusters_info[city_idx]['schools'][closest_school_idx],
                                      ag_job=None,
                                      city_idx=self.loc_info['city_idx'])
            # Add agent to model
            self.model.add_agent_to_grid_sched_networks(a)
            # add newborn agent to home presence list
            a.loc_info['home'].agents_in.add(a)
            # Update num of children
            self.info['num_children'] += 1

    def simulate_random_death(self, age_1=20, age_2=75, age_3=90, prob_1=0.25, prob_2=0.7):
        """ Method that may randomly kill self agent at each step. Agent death likelihood varies with age
            Args:
                * age_1: integer. Minimum age for death to be possible
                * age_2: integer.
                * age_3: integer
                * prob_1: death probability in first period
                * prob_2: death probability in second period
        """
        # transform ages to steps
        age_1, age_2, age_3 = age_1 * 36, age_2 * 36, age_3 * 36
        # define stochastic probability of agent death as function of age
        if (self.info['age'] > age_1) and (self.info['age'] <= age_2):
            if random.random() < prob_1 / (age_2 - age_1):  # 25% pop will die through this period
                self.remove_after_death()
        elif (self.info['age'] > age_2) and (self.info['age'] < age_3):
            if random.random() < prob_2 / (age_3 - age_2):  # 70% pop will die through this period
                self.remove_after_death()
        elif self.info['age'] >= age_3:
            self.remove_after_death()

    # def remove_after_death(self):
    #     """ Removes agent object from all places where it belongs.
    #         It makes sure no references to agent object are left aftr removal,
    #         so that garbage collector can free memory
    #         Call this function if death conditions for agent are verified
    #     """
    #     # Remove agent from all networks
    #     for network in [self.model.family_network,
    #                     self.model.known_people_network,
    #                     self.model.friendship_network]:
    #         try:
    #             network.remove_node(self)
    #         except nx.NetworkXError:
    #             pass
    #     # remove agent from all locations where it might be
    #     for loc, attr in zip([self.loc_info['home'], self.loc_info['job'], self.loc_info['school']],
    #                          ['occupants','employees','students']) :
    #         try:
    #             getattr(loc, attr).remove(self)
    #             loc.agents_in.remove(self)
    #         except:
    #             continue
    #     # remove agent from city
    #     self.model.clusters_info[self.loc_info['city_idx']]['agents'].remove(self)
    #
    #     # remove agent from grid and schedule
    #     self.model.grid._remove_agent(self.pos, self)
    #     self.model.schedule.remove(self)
    #
    #     # make id from deceased agent available
    #     self.model.set_available_ids.add(self.unique_id)

    def look_for_job(self):
        """ Method for agent to look for a job """
        # loop through shuffled job centers list until a job is found
        np.random.shuffle(self.model.clusters_info[self.loc_info['city_idx']]['jobs'])
        for job_c in self.model.clusters_info[self.loc_info['city_idx']]['jobs']:
            if job_c.num_places:
                job_c.num_places -= 1
                self.loc_info['job'] = job_c
                job_c.info['employees'].add(self)
                job_c.agents_in.add(self)
                break

    def update_acquaintances(self, other, lang):
        """ Add edges to known_people network when meeting for first time """
        if other not in self.model.known_people_network[self]:
            self.model.known_people_network.add_edge(self, other)
            self.model.known_people_network[self][other].update({'num_meet': 1, 'lang': lang})
        elif (other not in self.model.family_network[self]) and (other not in self.model.friendship_network[self]):
            self.model.known_people_network[self][other]['num_meet'] += 1

    def make_friendship(self):
        """ Check num_meet in known people network to filter candidates """
        pass

    def start_conversation(self, with_agents=None, num_other_agents=1):
        """ Method that starts a conversation. It picks either a list of known agents or
            a list of random agents from current cell and starts a conversation with them.
            This method can also simulate distance contact e.g.
            phone, messaging, etc ... by specifying an agent through 'with_agents' variable

            Arguments:
                * with_agents : specify a specific agent or list of agents
                                with which conversation will take place
                                If None, by default the agent will be picked randomly
                                from all lang agents in current cell
            Returns:
                * Runs conversation and determines language(s) in which it takes place.
                  Updates heard/used stats
        """
        if not with_agents:
            # get all agents currently placed on chosen cell
            others = self.model.grid.get_cell_list_contents(self.pos)
            others.remove(self)
            # linguistic model of encounter with another random agent
            if len(others) >= num_other_agents:
                others = random.sample(others, num_other_agents)
                self.model.run_conversation(self, others, False)
        else:
            self.model.run_conversation(self, with_agents, False)

    def get_num_words_per_conv(self, long=True, age_1=14, age_2=65, scale_f=40,
                               real_spoken_tokens_per_day=16000):
        """ Computes number of words spoken per conversation for a given age
            drawing from a 'num_words vs age' curve
            It assumes a compression scale of 40 in SIMULATED vocabulary and
            16000 tokens per adult per day as REAL number of spoken words on average
            Args:
                * long: boolean. Describes conversation length
                * age_1, age_2: integers. Describe key ages for slope change in num_words vs age curve
                * scale_f : integer. Compression factor from real to simulated number of words in vocabulary
                * real_spoken_tokens_per_day: integer. Average REAL number of tokens used by an adult per day"""
        # TODO : define 3 types of conv: short, average and long ( with close friends??)
        age_1, age_2 = 36 * age_1, 36 * age_2
        real_vocab_size = scale_f * self.model.vocab_red
        # define factor total_words/avg_words_per_day
        f = real_vocab_size / real_spoken_tokens_per_day
        if self.info['age'] < age_1:
            delta = 0.0001
            decay = -np.log(delta / 100) / (age_1)
            factor =  f + 400 * np.exp(-decay * self.info['age'])
        elif age_1 <= self.info['age'] <= age_2:
            factor = f
        else:
            factor = f - 1 + np.exp(0.0005 * (self.info['age'] - age_2 ) )
        if long:
            return self.model.num_words_conv[1] / factor
        else:
            return self.model.num_words_conv[0] / factor

    def pick_vocab(self, lang, long=True, biling_interloc=False):
        """ Method that models word choice by self agent in a conversation
            Word choice is governed by vocabulary knowledge constraints
            Args:
                * lang: integer in [0, 1] {0:'spa', 1:'cat'}
                * long: boolean that defines conversation length
                * biling_interloc : boolean. If True, speaker word choice might be mixed, since
                    he/she is certain interlocutor will understand
            Output:
                * spoken words: dict where keys are lang labels and values are lists with words spoken
                    in lang key and corresponding counts
        """

        # TODO: VERY IMPORTANT -> Model language switch btw bilinguals, reflecting easiness of retrieval

        #TODO : model 'Grammatical foreigner talk' =>
        #TODO : how word choice is adapted by native speakers when speaking to learners
        #TODO : or more importantly, by adults to children

        #TODO: NEED MODEL of how to deal with missed words = > L12 and L21, emergent langs with mixed vocab ???

        # sample must come from AVAILABLE words in R ( retrievability) !!!! This can be modeled in following STEPS

        # 1. First sample from lang CDF ( that encapsulates all to-be-known concepts at a given age-step)
        # These are the thoughts that speaker tries to convey
        # TODO : VI BETTER IDEA. Known thoughts are determined by UNION of all words known in L1 + L12 + L21 + L2
        word_samples = randZipf(self.model.cdf_data['s'][self.info['age']],
                                int(self.get_num_words_per_conv(long) * 10)) # 10 is num_days per step
        # get unique words and counts
        bcs = np.bincount(word_samples)
        act, act_c = np.where(bcs > 0)[0], bcs[bcs > 0]
        # SLOWER BUT CLEANER APPROACH =>  act, act_c = np.unique(word_samples, return_counts=True)

        # check if conversation is btw bilinguals and therefore lang switch is possible
        # TODO : key is that most biling agents will not express surprise to lang mixing or switch
        # TODO : whereas monolinguals will. Feedback for correction
        # if biling_interloc:
        #     if lang == 'L1':
        #         # find all words among 'act' words where L2 retrievability is higher than L1 retrievability
        #         # Then fill L12 container with a % of those words ( their knowledge is known at first time of course)
        #
        #         self.lang_stats['L1']['R'][act] <= self.lang_stats['L2']['R'][act]
        #         L2_strongest = self.lang_stats['L2']['R'][act] == 1.
                #act[L2_strongest]

                # rand_info_access = np.random.rand(4, len(act))
                # mask_L1 = rand_info_access[0] <= self.lang_stats['L1']['R'][act]
                # mask_L12 = rand_info_access[1] <= self.lang_stats['L12']['R'][act]
                # mask_L21 = rand_info_access[2] <= self.lang_stats['L21']['R'][act]
                # mask_L2 = rand_info_access[3] <= self.lang_stats['L2']['R'][act]

        # 2. Given a lang, pick the variant that is most familiar to agent
        lang = 'L1' if lang == 0 else 'L2'
        if lang == 'L1':
            pct1, pct2 = self.lang_stats['L1']['pct'][self.info['age']] , self.lang_stats['L12']['pct'][self.info['age']]
            lang = 'L1' if pct1 >= pct2 else 'L12'
        elif lang == 'L2':
            pct1, pct2 = self.lang_stats['L2']['pct'][self.info['age']], self.lang_stats['L21']['pct'][self.info['age']]
            lang = 'L2' if pct1 >= pct2 else 'L21'

        # 3. Then assess which sampled words-concepts can be successfully retrieved from memory
        # get mask for words successfully retrieved from memory
        mask_R = np.random.rand(len(act)) <= self.lang_stats[lang]['R'][act]
        spoken_words = {lang: [act[mask_R], act_c[mask_R]]}
        # if there are missing words-concepts, they might be found in the other known language(s)
        if np.count_nonzero(mask_R) < len(act):
            if lang in ['L1', 'L12']:
                lang2 = 'L12' if lang == 'L1' else 'L1'
            elif lang in ['L2', 'L21']:
                lang2 = 'L21' if lang == 'L2' else 'L2'
            mask_R2 = np.random.rand(len(act[~mask_R])) <= self.lang_stats[lang2]['R'][act[~mask_R]]
            if act[~mask_R][mask_R2].size:
                spoken_words.update({lang2: [act[~mask_R][mask_R2], act_c[~mask_R][mask_R2]]})
            # if still missing words, check in last lang available
            if (act[mask_R].size + act[~mask_R][mask_R2].size) < len(act):
                lang3 = 'L2' if lang2 in ['L12', 'L1'] else 'L1'
                rem_words = act[~mask_R][~mask_R2]
                mask_R3 = np.random.rand(len(rem_words)) <= self.lang_stats[lang3]['R'][rem_words]
                if rem_words[mask_R3].size:
                    # VERY IMP: add to transition language instead of 'pure' one.
                    # This is the process of creation/adaption/translation
                    tr_lang = max([lang, lang2], key=len)
                    spoken_words.update({lang2: [rem_words[mask_R3], act_c[~mask_R][~mask_R2][mask_R3]]})
        # TODO : add further effect of 'pressure' from very clear-in-mind words from other lang on spoken lang
        return spoken_words

    def update_lang_arrays(self, sample_words, speak=True, a=7.6, b=0.023, c=-0.031, d=-0.2,
                           delta_s_factor=0.25, min_mem_times=5, pct_threshold=0.9, pct_threshold_und=0.1):
        """ Function to compute and update main arrays that define agent linguistic knowledge
            Args:
                * sample_words: dict where keys are lang labels and values are tuples of
                    2 NumPy integer arrays. First array is of conversation-active unique word indices,
                    second is of corresponding counts of those words
                * speak: boolean. Defines whether agent is speaking or listening
                * a, b, c, d: float parameters to define memory function from SUPERMEMO by Piotr A. Wozniak
                * delta_s_factor: positive float < 1.
                    Defines increase of mem stability due to passive rehearsal
                    as a fraction of that due to active rehearsal
                * min_mem_times: positive integer. Minimum number of times to start remembering a word.
                    It should be understood as average time it takes for a word meaning to be incorporated
                    in memory
                * pct_threshold: positive float < 1. Value to define lang knowledge in percentage.
                    If retrievability R for a given word is higher than pct_threshold,
                    the word is considered as well known. Otherwise, it is not
                * pct_threshold_und : positive float < 1. If retrievability R for a given word
                    is higher than pct_threshold, the word can be correctly understood.
                    Otherwise, it cannot


            MEMORY MODEL: https://www.supermemo.com/articles/stability.htm

            Assumptions ( see "HOW MANY WORDS DO WE KNOW ???" By Marc Brysbaert*,
            Michaël Stevens, Paweł Mandera and Emmanuel Keuleers):
                * ~16000 spoken tokens per day + 16000 heard tokens per day + TV, RADIO
                * 1min reading -> 220-300 tokens with large individual differences, thus
                  in 1 h we get ~ 16000 words
        """

        # TODO: need to define similarity matrix btw language vocabularies?
        # TODO: cat-spa have dist mean ~ 2 and std ~ 1.6 ( better to normalize distance ???)
        # TODO for cat-spa np.random.choice(range(7), p=[0.05, 0.2, 0.45, 0.15, 0.1, 0.025,0.025], size=500)
        # TODO 50% of unknown words with edit distance == 1 can be understood, guessed

        for lang, (act, act_c) in sample_words.items():
            # UPDATE WORD COUNTING +  preprocessing for S, t, R UPDATE
            # If words are from listening, they might be new to agent
            # ds_factor value will depend on action type (speaking or listening)
            if not speak:
                known_words = np.nonzero(self.lang_stats[lang]['R'] > pct_threshold_und)
                # boolean mask of known active words
                known_act_bool = np.in1d(act, known_words, assume_unique=True)
                if np.all(known_act_bool):
                    # all active words in conversation are known
                    # update all active words
                    self.lang_stats[lang]['wc'][act] += act_c
                else:
                    # some heard words are unknown. Find them in 'act' words vector base
                    unknown_act_bool = np.invert(known_act_bool)
                    unknown_act , unknown_act_c = act[unknown_act_bool], act_c[unknown_act_bool]
                    # Select most frequent unknown word
                    ix_most_freq_unk = np.argmax(unknown_act_c)
                    most_freq_unknown = unknown_act[ix_most_freq_unk]
                    # update most active unknown word's count (the only one actually grasped)
                    self.lang_stats[lang]['wc'][most_freq_unknown] += unknown_act_c[ix_most_freq_unk]
                    # update known active words count
                    self.lang_stats[lang]['wc'][act[known_act_bool]] += act_c[known_act_bool]
                    # get words available for S, t, R update
                    if self.lang_stats[lang]['wc'][most_freq_unknown] > min_mem_times:
                        known_act_ixs = np.concatenate((np.nonzero(known_act_bool)[0], [ix_most_freq_unk]))
                        act, act_c = act[known_act_ixs ], act_c[known_act_ixs ]
                    else:
                        act, act_c = act[known_act_bool], act_c[known_act_bool]
                ds_factor = delta_s_factor
            else:
                # update word counter with newly active words
                self.lang_stats[lang]['wc'][act] += act_c
                ds_factor = 1
            # check if there are any words for S, t, R update
            if act.size:
                # compute increase in memory stability S due to (re)activation
                # TODO : I think it should be dS[reading]  < dS[media_listening]  < dS[listen_in_conv] < dS[speaking]
                S_act_b = self.lang_stats[lang]['S'][act] ** (-b)
                R_act = self.lang_stats[lang]['R'][act]
                delta_S = ds_factor * (a * S_act_b * np.exp(c * 100 * R_act) + d)
                # update memory stability value
                self.lang_stats[lang]['S'][act] += delta_S
                # discount one to counts
                act_c -= 1
                # Simplification with good approx : we apply delta_S without iteration !!
                S_act_b = self.lang_stats[lang]['S'][act] ** (-b)
                R_act = self.lang_stats[lang]['R'][act]
                delta_S = ds_factor * (act_c * (a * S_act_b * np.exp(c * 100 * R_act) + d))
                # update
                self.lang_stats[lang]['S'][act] += delta_S
                # update daily boolean mask to update elapsed-steps array t
                self.day_mask[lang][act] = True
                # set last activation time counter to zero if word act
                self.lang_stats[lang]['t'][self.day_mask[lang]] = 0
                # compute new memory retrievability R and current lang_knowledge from t, S values
                self.lang_stats[lang]['R'] = np.exp(-self.k * self.lang_stats[lang]['t'] / self.lang_stats[lang]['S'])
                self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > pct_threshold)[0].shape[0] /
                                                          self.model.vocab_red)

    def listen(self, conversation=True):
        """ Method to listen to conversations, media, etc... and update corresponding vocabulary
            Args:
                * conversation: Boolean. If True, agent will listen to potential conversation taking place
                on its current cell.If False, agent will listen to media"""
        # get all agents currently placed on chosen cell
        if conversation:
            others = self.model.grid.get_cell_list_contents(self.pos)
            others.remove(self)
            # if two or more agents in cell, conversation is possible
            if len(others) >= 2:
                ag_1, ag_2 = np.random.choice(others, size=2, replace=False)
                # call run conversation with bystander
                self.model.run_conversation(ag_1, ag_2, bystander=self)

        # TODO : implement 'listen to media' option


    def read(self):
        pass

    def study_lang(self, lang):
        pass

    def update_lang_switch(self, switch_threshold=0.1):
        """Switch to a new linguistic regime when threshold is reached
           If language knowledge falls below switch_threshold value, agent
           becomes monolingual"""

        if self.info['language'] == 0:
            if self.lang_stats['L2']['pct'][self.info['age']] >= switch_threshold:
                self.info['language'] = 1
        elif self.info['language'] == 2:
            if self.lang_stats['L1']['pct'][self.info['age']] >= switch_threshold:
                self.info['language'] = 1
        elif self.info['language'] == 1:
            if self.lang_stats['L1']['pct'][self.info['age']] < switch_threshold:
                self.info['language'] == 2
            elif self.lang_stats['L2']['pct'][self.info['age']] < switch_threshold:
                self.info['language'] == 0

    def stage_1(self):
        self.start_conversation()

    def stage_2(self):
        self.loc_info['home'].agents_in.remove(self)
        self.move_random()
        self.start_conversation()
        self.move_random()
        #self.listen()

    def stage_3(self):
        if self.info['age'] < 720:
            self.model.grid.move_agent(self, self.loc_info['school'].pos)
            self.loc_info['school'].agents_in.add(self)
            # TODO : DEFINE GROUP CONVERSATIONS !
            self.study_lang(0)
            self.study_lang(1)
            self.start_conversation() # WITH FRIENDS, IN GROUPS
        else:
            if self.loc_info['job']:
                self.model.grid.move_agent(self, self.loc_info['job'].pos)
                self.loc_info['job'].agents_in.add(self)
                # TODO speak to people in job !!! DEFINE GROUP CONVERSATIONS !
                self.start_conversation()
                self.start_conversation()
            else:
                self.look_for_job()
                self.start_conversation()

    def stage_4(self):
        if self.info['age'] < 720:
            self.loc_info['school'].agents_in.remove(self)
        elif self.loc_info['job']:
            self.loc_info['job'].agents_in.remove(self)
        self.move_random()
        self.start_conversation()
        if random.random() > 0.5 and self.model.friendship_network[self]:
            picked_friend = np.random.choice(self.model.friendship_network.neighbors(self))
            self.start_conversation(with_agents=picked_friend)
        self.model.grid.move_agent(self, self.loc_info['home'].pos)
        self.loc_info['home'].agents_in.add(self)
        try:
            for key in self.model.family_network[self]:
                if key.pos == self.loc_info['home'].pos:
                    lang = self.model.family_network[self][key]['lang']
                    self.start_conversation(with_agents=key, lang=lang)
        except:
            pass
        # memory becomes ever shakier after turning 65...
        if self.info['age'] > 65 * 36:
            for lang in ['L1', 'L2']:
                self.lang_stats[lang]['S'] = np.where(self.lang_stats[lang]['S'] >= 0.01,
                                                      self.lang_stats[lang]['S'] - 0.01,
                                                      0.000001)
        # Update at the end of each step
        # for lang in ['L1', 'L2']:
        #     self.lang_stats[lang]['pct'][self.info['age']] = (np.where(self.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
        #                                               self.model.vocab_red)

    def __repr__(self):
        return 'Lang_Agent_{0.unique_id!r}'.format(self)


class Home:
    def __init__(self, pos):
        self.occupants = set()
        self.pos=pos
        self.agents_in = set()

    def __repr__(self):
        return 'Home_{0.pos!r}'.format(self)


class EducationCenter:
    """ Common methods to all educational centers such a Schools, Faculties """

    def group_students_per_year(self):
        """ group school students by age in order to create different courses """
        if not self.grouped_studs:
            list_studs = list(self.info['students'])
            if list_studs:
                studs_sorted = sorted(list_studs, key=lambda x: int(x.info['age'] / 36))
                self.grouped_studs = dict([(c_key, {'students': set(gr_studs)})
                                           for c_key, gr_studs
                                           in groupby(studs_sorted, lambda x: int(x.info['age'] / 36))
                                           if c_key <= self.info['age_range'][1]])
                # assign course key to students
                for c_key, ags in self.grouped_studs.items():
                    for st in ags['students']:
                        st.loc_info['course_key'] = c_key

    def find_teachers(self, courses_keys, ret_output=False):
        """ Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age
        """
        # TODO : when teacher dies or retires =>
        # TODO a signal needs to be sent to school so that it can update => Call 'look_for_teachers'
        num_needed_teachers = len(courses_keys)
        school_clust_ix = self.info['clust']
        hired_teachers = []
        # loop over clusters from closest to farthest from school
        for ix in self.model.clusters_info[school_clust_ix]['closest_clusters']:
            # list cluster teacher candidates. Shuffle them to add randomness
            # TODO : hire teachers based on fact they have UNIV education !!!
            clust_cands = [ag for ag in self.model.clusters_info[ix]['agents']
                           if ag.info['language'] in self.info['lang_policy'] and
                           ag.info['age'] > (self.info['min_age_teacher'] * self.model.steps_per_year) and
                           not isinstance(ag.loc_info['job'], (School, University))]
            random.shuffle(clust_cands)
            hired_teachers.extend(clust_cands)
            if len(hired_teachers) >= num_needed_teachers:
                # hire all teachers and assign school to them before stopping loop
                hired_teachers = set(hired_teachers[:num_needed_teachers])
                break
            else:
                # continue looking for missing teachers in next cluster
                continue
        if ret_output:
            return hired_teachers

    def hire_teachers(self, keys):
        """ This method will be fully implemented in subclasses"""
        self.find_teachers(keys)
        pass

    def set_up_courses(self):
        """ Method to hire all necessary suitable teacher agents for a school instance.
            It calls method to group students by age to form courses, then it assigns
            a teacher to each course
        """
        if not self.grouped_studs:
            self.group_students_per_year()
        self.hire_teachers(self.grouped_studs.keys())

    def assign_stud(self, student):
        course_key = int(student.info['age'] / 36)
        if course_key in self.grouped_studs:
            # if course already exists
            self.grouped_studs[course_key]['students'].add(student)
        else:
            self.grouped_studs[course_key] = {'students': {student}}
            self.hire_teachers([course_key])
        self.info['students'].add(student)
        student.loc_info['course_key'] = course_key

    def update_courses(self, max_course_dist=3):
        """ Method to update courses at the end of the year. It moves all students forward to
            next course and checks each new course has a corresponding teacher. It sends a percentage of
            last year students to university, rest to job market
            * max_course_dist : integer. Maximum distance in years between courses to allow
                teacher exchange"""
        if not self.grouped_studs:
            self.group_students_per_year()
        else:
            updated_groups = {}
            # define set with keys for next year courses with students but without teacher
            missing_teachers_keys = set()
            for (c_id, course) in self.grouped_studs.items():
                # If course_id is smaller than maximum allowed age in school, rearrange
                if c_id < self.info['age_range'][1]:
                    # move students forward to next class and delete students from current class
                    updated_groups[c_id + 1] = {'students': course['students']} # TODO make set instead of list ??
                    self.grouped_studs[c_id]['students'] = None
                    # check if c_id + 1 was already part of last year courses
                    if c_id + 1 not in self.grouped_studs:
                        # if it's a new course, add course id to missing teacher keys
                        missing_teachers_keys.add(c_id + 1)
                    else:
                        # keep teacher
                        updated_groups[c_id + 1].update({'teacher': self.grouped_studs[c_id + 1]['teacher']})
                else:
                    self.exit_studs(course['students'])

            # define set of teachers that are left with an empty class
            jobless_teachers_keys = {key for key in self.grouped_studs if key not in updated_groups}
            # define reallocating pairs dict {jobless_key: missing_key}
            # to reassign school jobless teachers to courses with students but without teacher
            pairs = {}
            for jlt_key in jobless_teachers_keys:
                for mt_key in missing_teachers_keys:
                    if abs(jlt_key - mt_key) < max_course_dist and mt_key not in pairs.values():
                        pairs.update({jlt_key: mt_key})
            for (jlt_key, mt_key) in pairs.items():
                updated_groups[mt_key]['teacher'] = self.grouped_studs[jlt_key]['teacher']
                # update sets after reallocation
                jobless_teachers_keys.remove(jlt_key)
                missing_teachers_keys.remove(mt_key)

            # check if there are still jobless teachers left in school
            if jobless_teachers_keys:
                # TODO : for jobless teachers, find another occupation. Use them when needed
                for jlt_key in jobless_teachers_keys:
                    self.grouped_studs[jlt_key]['teacher'].loc_info['course_key'] = None

            self.grouped_studs = updated_groups

            # check if there are still missing teachers for some courses and hire them
            if missing_teachers_keys:
                self.hire_teachers(missing_teachers_keys)

            # assign course id to each student
            for (c_id, course) in self.grouped_studs.items():
                for st in course['students']:
                    st.loc_info['course_key'] = c_id


class School(EducationCenter):
    """ Class that defines a School object
        Args:
            * pos: 2-D tuple of integers. School coordinates
            * clust: integer. Cluster to which school belongs
            * num_places: integer. Total number of school places available
            * age_range: 2-D tuple of integers.
            * lang_policy: requested languages in order to work at this site
                [0, 1] -> both 0, 1 agents may work here
                [1] -> only 1 agents may work here
                [1, 2] -> both 1, 2 agents may work here
    """
    def __init__(self, model, pos, clust, num_places, age_range=(1, 18),
                 lang_policy=None, min_age_teacher=30):
        self.model = model
        self.pos = pos
        self.agents_in = set()
        self.info = {'employees': set(), 'students': set(),
                     'lang_policy': lang_policy, 'clust': clust,
                     'age_range': age_range, 'num_places': num_places,
                     'min_age_teacher': min_age_teacher}
        self.grouped_studs = dict()

    def hire_teachers(self, courses_keys):
        """ Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age
        """
        hired_teachers = self.find_teachers(courses_keys, ret_output=True)
        # assign class key to teachers and add teachers to grouped studs
        # TODO : sort employees by lang competence from lowest to highest
        for (k, t) in zip(courses_keys, hired_teachers):
            self.grouped_studs[k]['teacher'] = t
            t.loc_info['job'] = self
            t.loc_info['course_key'] = k
            self.info['employees'].add(t)

    def assign_stud(self, student):
        super().assign_stud(student)
        student.loc_info['school'] = self

    def exit_studs(self, studs):
        # sent studs from last year to univ or job market
        universities = [clust_info['university']
                        for clust_info in self.model.clusters_info.values()
                        if 'university' in clust_info]
        # get closest univ
        idx_univ = np.argmin([pdist([self.pos, univ.pos])
                              for univ in universities])
        univ = universities[idx_univ]
        # approx half of last year studs to univ
        univ_stds = np.random.choice(list(studs),
                                     size=ceil(0.5 * len(studs)),
                                     replace=False)
        for st in univ_stds:
            self.info['students'].remove(st)
            st.loc_info['school'] = None
            fac_key = random.choice(string.ascii_letters[:5])
            fac = univ.faculties[fac_key]
            fac.assign_stud(st)

        # rest of last year students to job market
        job_stds = set(studs).difference(set(univ_stds))
        for st in job_stds:
            self.info['students'].remove(st)
            # st.grow() # TODO : activate time evolving agents before calling this method
            st.loc_info['school'] = None
            del st.loc_info['course_key']
            st.look_for_job()

    def teachers_course_swap(self):
        """ Every n years on average teachers are swapped between classes """
        # sort dict keys before iteration
        sorted_keys = sorted(list(self.grouped_studs.keys()))
        # swap teacher keys, and teachers for each pair of classes
        for (k1, k2) in zip(*[iter(sorted_keys)] * 2):
            self.grouped_studs[k1]['teacher'].loc_info['course_key'] = k2
            self.grouped_studs[k2]['teacher'].loc_info['course_key'] = k1
            (self.grouped_studs[k1]['teacher'],
             self.grouped_studs[k2]['teacher']) = (self.grouped_studs[k2]['teacher'],
                                                   self.grouped_studs[k1]['teacher'])

    def __repr__(self):
        return 'School_{0.pos!r}'.format(self)


class Faculty(EducationCenter):

    def __init__(self, fac, univ, model):
        self.model = model
        self.univ = univ
        self.pos = self.univ.pos
        self.info = {'students': set(), 'employees': set(),
                     'lang_policy': univ.info['lang_policy'],
                     'age_range': univ.info['age_range'],
                     'min_age_teacher': univ.info['min_age_teacher'], 'type': fac,
                     'clust': univ.info['clust']}

        self.agents_in = set()
        #self.grouped_studs = {k: defaultdict(set) for k in range(*self.info['age_range'])}
        self.grouped_studs = dict()

    def hire_teachers(self, courses_keys):
        """ Hire teachers for the specified courses
            Args:
                * courses_keys: list of integer(s). Identifies courses for which teachers are missing
                    through years of age
        """
        hired_teachers = self.find_teachers(courses_keys, ret_output=True)
        # assign class key to teachers and add teachers to grouped studs
        # TODO : sort employees by lang competence from lowest to highest
        for (k, t) in zip(courses_keys, hired_teachers):
            self.grouped_studs[k]['teacher'] = t
            t.loc_info['job'] = [self.univ, self.info['type']]
            t.loc_info['course_key'] = k
            self.info['employees'].add(t)
            self.univ.info['employees'].add(t)

    def exit_studs(self, studs):
        for st in studs:
            self.info['students'].remove(st)
            self.univ.info['students'].remove(st)
            st.loc_info['university'] = None
            del st.loc_info['course_key']
            st.look_for_job()
            # TODO : activate time evolving agents before calling this method
            #st.grow()

    def assign_stud(self, student):
        super().assign_stud(student)
        self.univ.info['students'].add(student)
        student.loc_info['university'] = [self.univ, self.info['type']]
        student.loc_info['course_key'] = int(student.info['age'] / 36)

    def __repr__(self):
        return 'Faculty_{0.pos!r}'.format(self)


class University:
    """ class that defines a University object for tertiary studies """

    def __init__(self, model, pos, clust, age_range=(19, 23),
                 lang_policy=[0, 1], min_age_teacher=35):
        self.model = model
        self.pos = pos
        self.info = {'clust': clust, 'lang_policy': lang_policy, 'students': set(),
                     'employees': set(), 'age_range': age_range, 'min_age_teacher': min_age_teacher}
        self.faculties = {key: Faculty(key, self, model) for key in string.ascii_letters[:5]}

    def __repr__(self):
        return 'University_{0.pos!r}'.format(self)


class Job:
    """ class that defines a Job object.
        Args:
            *
            * lang_policy: requested languages in order to work at this site
                0 -> only L1 (so both 0, 1 agents may work here)
                1 -> both L1 and L2 ( only 1 agents may work here )
                2 -> only L2 (so both 1, 2 agents may work here)
    """
    def __init__(self, pos, num_places, skill_level=0, lang_policy = 1):
        self.pos=pos
        self.num_places=num_places
        self.info = {'employees': set(), 'lang_policy': lang_policy,
                     'skill_level': skill_level }
        self.agents_in = set()

    def look_for_employees(self):
        pass

    # TODO : update workers by department and send them to retirement when age reached

    def __repr__(self):
        return 'Job{0.pos!r}'.format(self)