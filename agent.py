# IMPORT LIBS
import random
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from collections import defaultdict

#import private library to model lang zipf CDF
from zipf_generator import Zipf_Mand_CDF_compressed, randZipf


class BaseAgent:
    """ Basic agent class that contains attributes and methods common to all lang agents subclasses"""

    # define memory retrievability constant
    k = np.log(10 / 9)

    def __init__(self, model, unique_id, language, sex, age=0, home=None, lang_act_thresh=0.1,
                 lang_passive_thresh=0.025, import_ic=False):
        """ language => 0, 1, 2 => spa, bil, cat  """
        self.model = model
        self.unique_id = unique_id
        self.info = {'age': age, 'language': language, 'sex': sex}
        self.loc_info = {'home': home}
        self.lang_thresholds = {'speak': lang_act_thresh, 'understand': lang_passive_thresh}

        # define container for languages' tracking and statistics
        self.lang_stats = defaultdict(lambda: defaultdict(dict))
        # define mask for each step
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

    def get_langs_pcts(self):
        pct_lang1 = self.lang_stats['L1']['pct'][self.info['age']]
        pct_lang2 = self.lang_stats['L2']['pct'][self.info['age']]
        return pct_lang1, pct_lang2

    def get_dominant_lang(self, ret_pcts=False):
        pcts = self.get_langs_pcts()
        dominant_lang = np.argmax(pcts)
        if ret_pcts:
            return dominant_lang, pcts
        else:
            return dominant_lang

    def update_lang_switch(self, switch_threshold=0.2):
        """Switch to a new linguistic regime when threshold is reached
           If language knowledge falls below switch_threshold value, agent
           becomes monolingual
           This method is called from schedule at each step"""

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

    def grow(self, new_class):
        """ It replaces current agent with a new agent subclass instance.
            It removes current instance from all networks, lists, sets in model and adds new instance
            to them instead.
            Args:
                * new_class: class. Agent subclass that will replace the current one
        """
        grown_agent = new_class(self.model, self.unique_id, self.info['language'], self.info['sex'])
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
        for loc, attr in zip([self.loc_info['home'], self.loc_info['job'],
                              self.loc_info['school'], self.loc_info['university']],
                             ['occupants', 'employees', 'students', 'students']):
            try:
                getattr(loc, attr).remove(self)
                getattr(loc, attr).add(grown_agent)
                loc.agents_in.remove(self)
                loc.agents_in.add(grown_agent)
            except AttributeError:
                continue
        # remove old instance and add new one to city
        self.model.clusters_info[self.loc_info['home'].clust]['agents'].remove(self)
        self.model.clusters_info[self.loc_info['home'].clust]['agents'].add(grown_agent)
        # remove agent from grid and schedule
        self.model.grid._remove_agent(self.pos, self)
        self.model.schedule.remove(self)
        # add new_agent to grid and schedule
        self.model.grid.place_agent(grown_agent, self.pos)
        self.model.schedule.add(grown_agent)

    def random_death(self, a=1.23368173e-05, b=2.99120806e-03, c=3.19126705e+01):
        """ Method to randomly determine agent death or survival at each step
            The fitted function provides the death likelihood for a given rounded age
            In order to get the death-probability per step we divide
            by number of steps in a year (36)
            Fitting parameters are from
            https://www.demographic-research.org/volumes/vol27/20/27-20.pdf
            " Smoothing and projecting age-specific probabilities of death by TOPALS "
            by Joop de Beer
            Resulting life expectancy is 77 years and std is ~ 15 years
        """
        if random.random() < a * (np.exp(b * self.info['age']) + c) / self.model.steps_per_year:
            self.model.remove_after_death(self)

    def __repr__(self):
        return 'Lang_Agent_{0.unique_id!r}_clust_{1!r}'.format(self, self.loc_info['home'].clust )


class ListenerAgent(BaseAgent):
    """ BaseAgent class augmented with listening-related methods """

    def listen(self, to_agent=None, min_age_interlocs=None, num_days=10):
        """
            Method to listen to conversations, media, etc... and update corresponding vocabulary
            Args:
                * to_agent: class instance (optional). It can be either a language agent or a media agent.
                    If not specified, self agent will listen to a random conversation taking place on his cell
                * min_age_interlocs: integer. Allows to adapt vocabulary choice to the youngest
                    participant in the conversation
                * num_days: integer. Number of days per step listening action takes place on average.
                    A step equals 10 days.
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
            if self in self.model.known_people_network[to_agent]:
                lang = self.model.known_people_network[to_agent][self]['lang']
            else:
                conv_params = self.model.get_conv_params([to_agent, self])
                if conv_params['bilingual']:
                    lang = conv_params['lang_group'][0]
                else:
                    lang = conv_params['lang_group']
            words = to_agent.pick_vocab(lang, long=False, min_age_interlocs=min_age_interlocs,
                                        num_days=num_days)
            for lang_key, lang_words in words:
                words = np.intersect1d(self.model.cdf_data['s'][self.info['age']],
                                       lang_words, assume_unique=True)
                self.update_lang_arrays(words, speak=False, delta_s_factor=0.75)

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
                # compute language knowledge in percentage
                if lang in ['L1', 'L12']:
                    real_lang_knowledge = np.maximum(self.lang_stats['L1']['R'], self.lang_stats['L12']['R'])
                    self.lang_stats['L1']['pct'][self.info['age']] = (np.where(real_lang_knowledge > pct_threshold)[0].shape[0] /
                                                                      self.model.vocab_red)
                else:
                    real_lang_knowledge = np.maximum(self.lang_stats['L2']['R'], self.lang_stats['L21']['R'])
                    self.lang_stats['L2']['pct'][self.info['age']] = (np.where(real_lang_knowledge > pct_threshold)[0].shape[0] /
                                                                      self.model.vocab_red)


class SpeakerAgent(ListenerAgent):

    def get_num_words_per_conv(self, long=True, age_1=14, age_2=65, scale_f=40,
                               real_spoken_tokens_per_day=16000):
        """ Computes number of words spoken per conversation for a given age
            based on a 'num_words vs age' curve
            It assumes a compression scale of 40 in SIMULATED vocabulary and
            16000 tokens per adult per day as REAL number of spoken words on average
            Args:
                * long: boolean. Describes conversation length
                * age_1: integer. Defines lower key age for slope change in num_words vs age curve
                * age_2: integer. Defines higher key age for slope change in num_words vs age curve
                * scale_f : integer. Compression factor from real to simulated number of words in vocabulary
                * real_spoken_tokens_per_day: integer. Average REAL number of tokens used by an adult per day"""
        # TODO : define 3 types of conv: short, average and long ( with close friends??)
        age_1, age_2 = [age * self.model.steps_per_year for age in [age_1, age_2]]
        real_vocab_size = scale_f * self.model.vocab_red
        # define factor total_words/avg_words_per_day
        f = real_vocab_size / real_spoken_tokens_per_day
        if self.info['age'] < age_1:
            delta = 0.0001
            decay = -np.log(delta / 100) / age_1
            factor =  f + 400 * np.exp(-decay * self.info['age'])
        elif age_1 <= self.info['age'] <= age_2:
            factor = f
        else:
            factor = f - 1 + np.exp(0.0005 * (self.info['age'] - age_2 ) )
        if long:
            return self.model.num_words_conv[1] / factor
        else:
            return self.model.num_words_conv[0] / factor

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

        # 1. First sample from lang CDF (that encapsulates all to-be-known concepts at a given age-step)
        # These are the thoughts or concepts a  speaker tries to convey
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
            # if still missing words, look for them in last lang available
            if (act[mask_R].size + act[~mask_R][mask_R2].size) < len(act):
                lang3 = 'L2' if lang2 in ['L12', 'L1'] else 'L1'
                rem_words = act[~mask_R][~mask_R2]
                mask_R3 = np.random.rand(len(rem_words)) <= self.lang_stats[lang3]['R'][rem_words]
                if rem_words[mask_R3].size:
                    # VERY IMP: add to transition language instead of 'pure' one.
                    # This is the process of creation/adaption/translation
                    tr_lang = max([lang, lang2], key=len)
                    spoken_words.update({tr_lang: [rem_words[mask_R3], act_c[~mask_R][~mask_R2][mask_R3]]})
        return spoken_words

    def update_acquaintances(self, other, lang):
        """ Add edges to known_people network when meeting (speaking) for first time """
        if other not in self.model.nws.known_people_network[self]:
            self.model.nws.known_people_network.add_edge(self, other)
            self.model.nws.known_people_network[self][other].update({'num_meet': 1, 'lang': lang})
        elif (other not in self.model.nws.family_network[self]) and (other not in self.model.nws.friendship_network[self]):
            self.model.nws.known_people_network[self][other]['num_meet'] += 1


class SchoolAgent(SpeakerAgent):

    def speak_at_school(self, school, course_key, num_days):
        # talk to school mates
        mates = school.grouped_studs[course_key]['students'].difference({self})
        if mates:
            num_mates = random.randint(1, len(mates))
            mates = random.sample(mates, num_mates)
            self.model.run_conversation(self, mates, num_days=num_days)
        # talk to friends
        for friend in self.model.nws.friendship_network:
            self.model.run_conversation(self, friend, num_days=num_days)

    def study_vocab(self, lang, delta_s_factor=0.25, num_words=100):
        """ Method to update vocabulary without conversations
        Args:
            * lang: string. Language in which agent studies ['L1', 'L2']
            * delta_s_factor: positive float < 1. Defines increase of mem stability
                due to passive rehearsal as a fraction of that due to active rehearsal
            * num_words : integer. Number of words studied
        """
        word_samples = randZipf(self.model.cdf_data['s'][self.info['age']], num_words)
        # get unique words and counts
        bcs = np.bincount(word_samples)
        act, act_c = np.where(bcs > 0)[0], bcs[bcs > 0]
        studied_words = {lang: [act, act_c]}
        self.update_lang_arrays(studied_words, delta_s_factor=delta_s_factor, speak=False)


class IndepAgent(ListenerAgent):

    def move_random(self):
        """ Take a random step into any surrounding cell
            All eight surrounding cells are available as choices
            Current cell is not an output choice

            Returns:
                * modifies self.pos attribute
        """
        x, y = self.pos  # agent attr 'pos' is defined when adding agent to GRID
        possible_steps = self.model.grid.get_neighborhood((x, y),
                                                          moore=True,
                                                          include_center=False)
        chosen_cell = random.choice(possible_steps)
        self.model.grid.move_agent(self, chosen_cell)

    def pick_random_friend(self, ix_agent):
        # get current agent neighboring nodes ids
        adj_mat = self.model.nws.adj_mat_friend_nw[ix_agent]
        # get agents ids and probs
        ags_ids = np.nonzero(adj_mat)[0]
        probs = adj_mat[ags_ids]
        picked_agent = self.model.schedule.agents[np.random.choice(ags_ids, p=probs)]
        return picked_agent

    def speak_to_random_friend(self, ix_agent, num_days):
        random_friend = self.pick_random_friend(ix_agent)
        self.model.run_conversation(self, random_friend, num_days=num_days)

    def meet_agent(self):
        pass
    # TODO : method to meet new agents


class Baby(ListenerAgent):

    """ Agent from 0 to 2 years old.
        It can only be initialized from 'reproduce' method in Young and Adult agents
    """

    age_low, age_high = 0, 2

    def __init__(self, model, unique_id, language, sex,
                 father, mother, lang_with_father, lang_with_mother,
                 age=0, home=None, lang_act_thresh=0.1, lang_passive_thresh=0.025,
                 import_ic=False):

        super().__init__(model, unique_id, language, sex, age, home, lang_act_thresh,
                         lang_passive_thresh, import_ic)

        self.set_family_links(father, mother, lang_with_father, lang_with_mother)

        self.school_parent = np.random.choice([self.info['close_family']['mother'],
                                               self.info['close_family']['father']],
                                              p=[0.7, 0.3])

    def set_family_links(self, father, mother, lang_with_father, lang_with_mother):
        """
            Method to define family links and interaction language of a nw agent.
            Corresponding edges are created in family network
            Args:
                * father: agent instance. The baby's father
                * mother: agent instance. The baby's mother
                * lang_with_father: integer [0, 1]. Defines lang with father
                * lang_with_mother: integer [0, 1]. Defines lang with mother
        """
        #TODO : should it be a model method ??
        fam_nw = self.model.nws.family_network
        k_people_nw = self.model.nws.known_people_network
        # Define family links with parents
        for (i, j, fam_link) in [(self, father, 'father'), (father, self, 'child')]:
            fam_nw.add_edge(i, j, fam_link=fam_link, lang=lang_with_father)
            k_people_nw.add_edge(i, j, family=True, lang=lang_with_father)
        for (i, j, fam_link) in [(self, mother, 'mother'), (mother, self, 'child')]:
            fam_nw.add_edge(i, j, fam_link=fam_link, lang=lang_with_mother)
            k_people_nw.add_edge(i, j, family=True, lang=lang_with_mother)
        # Define links with agent siblings if any
        siblings = [agent for agent, labels in fam_nw[mother].items()
                    if labels['fam_link'] == 'child' and agent != self]
        for sibling in siblings:
            lang_with_sibling = sibling.get_dominant_lang()
            for (i, j, fam_link) in [(self, sibling, 'sibling'), (sibling, self, 'sibling')]:
                fam_nw.add_edge(i, j, fam_link=fam_link, lang=lang_with_sibling)
                k_people_nw.add_edge(i, j, family=True, lang=lang_with_sibling)
        # rest of family will if possible speak same language with baby as their link agent to the baby
        for elder, lang in zip([father, mother], [lang_with_father, lang_with_mother]):
            for agent, labels in fam_nw[elder].items():
                lang_label = 'L1' if lang == 0 else 'L2'
                if agent.lang_stats[lang_label]['pct'][agent.info['age']] > agent.lang_act_thresh:
                    com_lang = lang
                else:
                    com_lang = 1 if lang == 0 else 0
                if labels["fam_link"] == 'father':
                    for (i, j, fam_link) in [(self, agent, 'grandfather'), (agent, self, 'grandchild')]:
                        fam_nw.add_edge(i, j, fam_link=fam_link, lang=com_lang)
                        k_people_nw.add_edge(i, j, family=True, lang=com_lang)
                elif labels["fam_link"] == 'mother':
                    for (i, j, fam_link) in [(self, agent, 'grandmother'), (agent, self, 'grandchild')]:
                        fam_nw.add_edge(i, j, fam_link=fam_link, lang=com_lang)
                        k_people_nw.add_edge(i, j, family=True, lang=com_lang)
                elif labels["fam_link"] == 'sibling':
                    fam_nw.add_edge(self, agent,
                                    fam_link='uncle' if agent.info['sex'] == 'M' else 'aunt',
                                    lang=com_lang)
                    fam_nw.add_edge(agent, self, fam_link='nephew', lang=com_lang)
                    for (i, j) in [(self, agent), (agent, self)]:
                        k_people_nw.add_edge(i, j, family=True, lang=com_lang)
                    if agent.info['married']:
                        consort = [key for key, value in fam_nw[agent].items()
                                   if value['fam_link'] == 'consort'][0]
                        fam_nw.add_edge(self, consort,
                                        fam_link ='uncle' if consort.info['sex'] == 'M' else 'aunt')
                        fam_nw.add_edge(consort, self, fam_link='nephew', lang=com_lang)
                        for (i, j) in [(self, consort), (consort, self)]:
                            k_people_nw.add_edge(i, j, family=True, lang=com_lang)
                elif labels["fam_link"] == 'nephew':
                    fam_nw.add_edge(self, agent, fam_link='cousin', lang=com_lang)
                    fam_nw.add_edge(agent, self, fam_link='cousin', lang=com_lang)
                    for (i, j) in [(self, agent), (agent, self)]:
                        k_people_nw.add_edge(i, j, family=True, lang=com_lang)
        self.info['close_family'] = dict()
        self.info['close_family']['mother'] = mother
        self.info['close_family']['father'] = father
        self.info['close_family']['siblings'] = siblings
        # TODO modify self.info['teacher'] = random.choice(self.loc_info['school'].info['teachers'])

    def register_to_school(self):
        # find closest school in cluster
        clust_info = self.model.geo.clusters_info[self.loc_info['home'].clust]
        idx_school = np.argmin([pdist([self.loc_info['home'].pos, school.pos])
                                for school in clust_info['schools']])
        school = clust_info['schools'][idx_school]
        # register
        school.assign_stud(self)

    def stage_1(self, num_days=10):
        # listen to close family at home
        # all activities are DRIVEN by current agent
        if self.info['age'] > 36:
            ags_at_home = self.loc_info['home'].agents_in.difference({self})
            if ags_at_home:
                for ag in ags_at_home:
                    if isinstance(ag, Child):
                        self.listen(to_agent=ag, min_age_interlocs=self.info['age'],
                                    num_days=num_days)
            if 'school' not in self.loc_info:
                self.register_to_school()

    def stage_2(self, num_days = 7):
        # go to daycare with mom or dad - 7 days out of 10
        # ONLY 7 out of 10 days are driven by current agent (rest by parents or caretakers)
        if self.info['age'] > 36:
            school = self.loc_info['school']
            self.model.grid.move_agent(self, school.pos)
            self.listen(to_agent=self.school_parent, min_age_interlocs=self.info['age'], num_days=num_days)
            # TODO : a part of speech from teacher to all course(driven from teacher stage method),
            school.agents_in.add(self)
            course_key = self.loc_info['course_key']
            teacher = school.grouped_studs[course_key]['teacher']
            self.model.run_conversation(teacher, self.school_parent, num_days=num_days)
            self.listen(to_agent=teacher, min_age_interlocs=self.info['age'], num_days=num_days)
        # model week-ends time are modeled in PARENTS stages

    def stage_3(self, num_days=7):
        # parent comes to pick up and speaks with other parents. Then baby listens to parent on way back
        if self.info['age'] > 36:
            school = self.loc_info['school']
            self.model.grid.move_agent(self.school_parent, school.pos)
            course_key = self.loc_info['course_key']
            parents = [stud.school_parent for stud in school.grouped_studs[course_key]['students']
                       if stud.school_parent.pos == school.pos ]
            if parents:
                num_peop = random.randint(1, min(len(parents), 4))
                self.school_parent.start_conversation(with_agents=random.sample(parents, num_peop))
            school.agents_in.remove(self)
            self.listen(to_agent=self.school_parent, min_age_interlocs=self.info['age'], num_days=num_days)
        # TODO : pick random friends from parents. Set up meeting witht them

    def stage_4(self):
        # baby goes to bed early during week
        self.stage_1(num_days=3)
        if self.info['age'] == self.age_high * self.model.steps_per_year:
            self.grow(Child)


class Child(SchoolAgent): #from 2 to 10

    age_low, age_high = 2, 10

    def __init__(self, model, unique_id, language, sex, age=0, home=None, lang_act_thresh=0.1,
                 lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, lang_act_thresh, lang_passive_thresh, import_ic)

        # set up close family
        self.school_parent = np.random.choice([self.info['close_family']['mother'],
                                               self.info['close_family']['father']],
                                              p=[0.7, 0.3])

    def stage_1(self, *args, num_days=10):
        ags_at_home = self.loc_info['home'].agents_in.difference({self})
        ags_at_home = [ag for ag in ags_at_home if isinstance(ag, Child)]
        others = random.randint(1, min(len(ags_at_home), 4))
        self.model.run_conversation(self, others)

    def stage_2(self, num_days=7):
        # go to daycare with mom or dad - 7 days out of 10
        # ONLY 7 out of 10 days are driven by current agent (rest by parents or caretakers)
        school = self.loc_info['school']
        self.model.grid.move_agent(self, school.pos)
        self.model.run_conversation(self, self.school_parent, num_days=num_days)
        # TODO : a part of speech from teacher to all course(driven from teacher stage method),
        school.agents_in.add(self)
        # talk to teacher and mates
        course_key = self.loc_info['course_key']
        # talk with teacher
        teacher = school.grouped_studs[course_key]['teacher']
        self.model.run_conversation(teacher, self, num_days=num_days)
        self.model.run_conversation(teacher, self.school_parent, num_days=2)
        # talk with school mates
        self.speak_at_school(self, school, course_key, num_days)
        # week-ends time are modeled in PARENTS stages

    def stage_3(self, num_days=7):
        school = self.loc_info['school']
        self.model.grid.move_agent(self.school_parent, school.pos)
        course_key = self.loc_info['course_key']
        parents = [stud.school_parent for stud in school.grouped_studs[course_key]['students']
                   if stud.school_parent.pos == school.pos ]
        if parents:
            num_peop = random.randint(1, min(len(parents), 4))
            self.school_parent.start_conversation(with_agents=random.sample(parents, num_peop))
        school.agents_in.remove(self)
        self.model.run_conversation(self, self.school_parent, num_days=num_days)

    def stage_4(self, num_days=10):
        self.stage_1(num_days=num_days)
        if self.info['age'] == self.age_high * self.model.steps_per_year:
            self.grow(Adolescent)


class Adolescent(IndepAgent, SchoolAgent): # from 10 to 18

    age_low, age_high = 10, 18

    def __init__(self, model, unique_id, language, sex, age=0, home=None, lang_act_thresh=0.1,
                 lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, lang_act_thresh, lang_passive_thresh, import_ic)

    def move_random(self):
        """ Take a random step into any surrounding cell
            All eight surrounding cells are available as choices
            Current cell is not an output choice

            Returns:
                * modifies self.pos attribute
        """
        x, y = self.pos  # attr pos is defined when adding agent to schedule
        possible_steps = self.model.grid.get_neighborhood((x, y),
                                                          moore=True,
                                                          include_center=False)
        chosen_cell = random.choice(possible_steps)
        self.model.grid.move_agent(self, chosen_cell)

    def study_vocab(self, lang, delta_s_factor=0.25, num_words=100):
        """ Method to update vocabulary without conversations
        Args:
            * lang: string. Language in which agent studies ['L1', 'L2']
            * delta_s_factor: positive float < 1. Defines increase of mem stability
                due to passive rehearsal as a fraction of that due to active rehearsal
            * num_words : integer. Number of words studied
        """
        word_samples = randZipf(self.model.cdf_data['s'][self.info['age']], num_words)
        # get unique words and counts
        bcs = np.bincount(word_samples)
        act, act_c = np.where(bcs > 0)[0], bcs[bcs > 0]
        studied_words = {lang: [act, act_c]}
        self.update_lang_arrays(studied_words, delta_s_factor=delta_s_factor, speak=False)

    def pick_random_friend(self, ix_agent):
        # get current agent neighboring nodes ids
        adj_mat = self.model.nws.adj_mat_friend_nw[ix_agent]
        # get agents ids and probs
        ags_ids = np.nonzero(adj_mat)[0]
        probs = adj_mat[ags_ids]
        picked_agent = self.model.schedule.agents[np.random.choice(ags_ids, p=probs)]
        return picked_agent

    def speak_to_random_friend(self, ix_agent, num_days):
        random_friend = self.pick_random_friend(ix_agent)
        self.model.run_conversation(self, random_friend, num_days=num_days)

    def speak_at_school(self, school, ix_agent, num_days):
        # talk to teacher and mates
        course_key = self.loc_info['course_key']
        # talk with teacher
        teacher = school.grouped_studs[course_key]['teacher']
        self.model.run_conversation(teacher, self, num_days=2)
        # talk with school mates
        mates = school.grouped_studs[course_key]['students'].difference({self})
        if mates:
            num_mates = random.randint(1, len(mates))
            mates = random.sample(mates, num_mates)
            self.model.run_conversation(self, mates, num_days=num_days)
        # talk to close friends
        self.speak_to_random_friend(ix_agent, num_days)
        # week-ends time are modeled in PARENTS stages

    def stage_1(self, *args, num_days=7):
        super().stage_1()

    def stage_2(self, ix_agent, num_days=7):
        school = self.loc_info['school']
        self.model.grid.move_agent(self, school.pos)
        # TODO : a part of speech from teacher to all course(driven from teacher stage method)
        school.agents_in.add(self)
        self.speak_at_school(school, num_days)

    def stage_3(self, ix_agent, num_days=7):
        school = self.loc_info['school']
        self.speak_at_school(school, num_days)
        school.agents_in.remove(self)
        if school.info['lang_policy'] == [0, 1]:
            self.study_vocab('L1', num_words=100)
            self.study_vocab('L2', num_words=50)
        elif school.info['lang_policy'] == [1]:
            self.study_vocab('L1')
            self.study_vocab('L2')
        elif school.info['lang_policy'] == [1, 2]:
            self.study_vocab('L1', num_words=50)
            self.study_vocab('L2', num_words=100)

    def stage_4(self, ix_agent, num_days=7):
        self.stage_1(ix_agent, num_days=num_days)
        # go out with friends at least once
        num_out = random.randint(1, 5)
        self.speak_to_random_friend(ix_agent, num_days=num_out)
        if self.info['age'] == self.age_high:
            self.grow(Young)


class Young(IndepAgent): # from 18 to 30

    age_low, age_high = 18, 30

    def __init__(self, model, unique_id, language, sex, age=0, home=None, lang_act_thresh=0.1,
                 lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age)
        self.info['married'] = married
        self.info['num_children'] = num_children
        self.loc_info['job'] = job
        self.loc_info['university'] = university
        self.info['close_family']['consort'] = [ag for ag in self.model.family_network[self]
                                                if self.model.family_network[self][ag]['fam_link'] == 'consort'][0]
        self.info['close_family']['children'] = [ag for ag in self.model.family_network[self]
                                                 if self.model.family_network[self][ag]['fam_link'] == 'child']

    def move_to_new_home(self, *ags, marriage=True):
        """ Method to move self agent and other optional agents to a new home
            Args:
            * ags: optional. Other agents instances
            * marriage: boolean. Specifies if moving is because of marriage or not. If not,
                it is assumed moving is because of university reasons
        """

        job1 = self.loc_info['job']
        home1 = self.loc_info['home']
        clust1_ix = self.loc_info['home'].clust
        free_homes_clust1 = [home for home in self.model.geo.clusters_info[clust1_ix]['homes']
                             if not home.occupants]
        if ags:
            if marriage:
                job2 = ags.loc_info['job']
                clust2_ix = ags.loc_info['home'].clust
                if job2:
                    job_dist_fun = lambda home: (pdist([job1.pos, home.pos]) +
                                                 pdist([job2.pos, home.pos]))[0]
                    if clust1_ix == clust2_ix :
                        sorted_homes = sorted(free_homes_clust1, key=job_dist_fun )
                        new_home = sorted_homes[0]
                    else:
                        # each agent lives and works in different clusters
                        # move to town with more job offers
                        num_jobs_1 = len(self.model.geo.clusters_info[clust1_ix]['jobs'])
                        num_jobs_2 = len(ags.model.geo.clusters_info[clust1_ix]['jobs'])
                        if num_jobs_1 >= num_jobs_2:
                            sorted_homes = sorted(free_homes_clust1, key=job_dist_fun)
                        else:
                            free_homes_clust2 = [home for home
                                                 in self.model.geo.clusters_info[clust2_ix]['homes']
                                                 if not home.occupants]
                            sorted_homes = sorted(free_homes_clust2, key=job_dist_fun)
                        new_home = sorted_homes[0]
                else:
                    job_dist_fun = lambda home: pdist([job1.pos, home.pos])[0]
                    sorted_homes = sorted(free_homes_clust1, key=job_dist_fun)
                    home_ix = random.randint(1, int(len(sorted_homes) / 2))
                    new_home = sorted_homes[home_ix]
            else:
                # find place close to university
                univ = self.loc_info['university']
                clust_univ = univ.info['clust']
                free_homes_univ_clust = [home for home in self.model.geo.clusters_info[clust_univ]['homes']
                                         if not home.occupants]
                job_dist_fun = lambda home: pdist([univ.pos, home.pos])[0]
                sorted_homes = sorted(free_homes_univ_clust, key=job_dist_fun)
                new_home = sorted_homes[0]
            new_home.assign_to_agent(self, ags)
        else:
            # assign empty home relatively close to current job
            sorted_homes = sorted(free_homes_clust1, key=lambda home: pdist([job1.pos, home.pos])[0])
            home_ix = random.randint(1, int(len(sorted_homes) / 2))
            new_home = sorted_homes[home_ix]
            new_home.assign_to_agent(self)

    def look_for_partner(self, avg_years=6, age_diff=10, thresh_comm_lang=0.3):
        """ Find partner every avg_years if agent is not married yet. Marrying agents
            must have a sufficiently high knowledge of common language

             Args:
                 * avg_years: integer. Real years on average to get a partner
                 * age_diff: integer. Max difference in real years between partners to get married
                 * thresh_comm_lang: float. Minimum required knowledge of common lang
        """

        # first check luck ( once every 10 years on average )
        if random.random() < 1 / (avg_years * self.model.steps_per_year):
            # find suitable partner amongst known people
            for ag in self.model.nws.known_people_network[self]:
                link_description = self.model.nws.known_people_network[self][ag]
                if ('num_meet' in link_description and link_description['num_meet'] > 10 and
                ag.info['sex'] != self.info['sex'] and
                abs(self.info['age'] - ag.info['age']) <= self.model.steps_per_year * age_diff):
                    if abs(ag.info['language'] - self.info['language']) <= 1:
                        # check both agents have sufficiently high level in common language
                        common_lang = link_description['lang']
                        pct_1 = self.lang_stats[common_lang]['pct'][self.info['age']]
                        pct_2 = ag.lang_stats[common_lang]['pct'][ag.info['age']]
                        if pct_1 > thresh_comm_lang and pct_2 > thresh_comm_lang:
                            # set marriage flags and links between partners
                            self.info['married'] = True
                            ag.info['married'] = True
                            fam_nw = self.model.nws.family_network
                            lang = self.model.nws.known_people_network[self][ag]['lang']
                            fam_nw.add_edge(self, ag, lang=lang, fam_link='consort')
                            # find appartment to move in together
                            self.move_to_new_home(ag)
                            break

    def reproduce(self, day_prob=0.001, max_num_children=4):
        """ give birth to a new agent if conditions and likelihoods are met """

        # TODO : check appropriate day_prob . Shouldn'it be 0.0015 ? to have > 1 child/per parent

        # TODO: check method integration with creation of 'Baby' class
        if (self.info['num_children'] < max_num_children and self.info['married'] and
            random.random() < day_prob):
            id_baby = self.model.set_available_ids.pop()
            # find consort
            consort = [agent for agent, labels in self.model.family_network[self].items()
                       if labels["fam_link"] == 'consort'][0]
            # find out baby language atribute and langs with parents
            newborn_lang, lang_with_father, lang_with_mother = self.model.get_newborn_lang(self, consort)
            # Determine baby's sex
            sex = 'M' if random.random() > 0.5 else 'F'
            father, mother = (self, consort) if self.info['sex'] == 'M' else (consort, self)
            # Make baby instance
            baby = Baby(self.model, id_baby, newborn_lang, sex,
                        father, mother, lang_with_father, lang_with_mother,
                        home=self.loc_info['home'])
            # Add agent to model
            self.model.geo.add_agents_to_grid_and_schedule(baby)
            self.model.nws.add_ags_to_networks(baby)
            # Update num of children for both self and consort
            self.info['num_children'] += 1
            consort.info['num_children'] += 1

    def look_for_job(self):
        # loop through shuffled job centers list until a job is found
        np.random.shuffle(self.model.clusters_info[self.loc_info['home'].clust]['jobs'])
        for job_c in self.model.clusters_info[self.loc_info['home'].clust]['jobs']:
            if job_c.num_places and self.info['language'] in job_c.info['lang_policy']:
                job_c.num_places -= 1
                self.loc_info['job'] = job_c
                self.grow(Worker)
                break

    def stage_1(self, ix_agent, num_days=7):
        pass

    def stage_2(self, ix_agent):
        pass

    def stage_3(self, ix_agent):
        pass

    def stage_4(self, ix_agent):
        self.speak_to_random_friend(ix_agent)
        if not self.info['married'] and self.info['job']:
            self.look_for_partner()
        if self.info['age'] == self.age_high * self.model.steps_per_year:
            self.grow(Adult)


class YoungUniv(IndepAgent, SchoolAgent):
    pass


class Adult(Young): # from 30 to 65

    age_low, age_high = 30, 65

    def __init__(self, model, unique_id, language, sex, age=0, home=None, lang_act_thresh=0.1,
                 lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, lang_act_thresh, lang_passive_thresh, import_ic)

    def reproduce(self, day_prob=0.005, limit_age=40):
        if self.info['age'] <= limit_age * self.model.steps_per_year:
            super().reproduce()

    def look_for_partner(self, avg_years=4, age_diff=10, thresh_comm_lang=0.3):
        super().look_for_partner(avg_years=avg_years)

    def stage_1(self):
        pass

    def stage_2(self):
        pass

    def stage_3(self):
        pass

    def stage_4(self, ix_agent, num_days=7):
        self.speak_to_random_friend(ix_agent)
        if not self.info['married'] and self.info['job']:
            self.look_for_partner()
        if self.info['age'] == self.model.steps_per_year * self.age_high:
            self.grow(Pensioner)


class Worker(Adult):
    pass


class Teacher(Adult):
    pass


class Pensioner(Adult): # from 65 to death

    def __init__(self, model, unique_id, language, sex, age=0, home=None, lang_act_thresh=0.1,
                 lang_passive_thresh=0.025, import_ic=False):
        super().__init__(model, unique_id, language, sex, age, home, lang_act_thresh, lang_passive_thresh, import_ic)

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
                 num_children=0, ag_home=None, ag_school=None, ag_job=None, import_IC=False):
        self.model = model
        self.unique_id = unique_id
        # language values: 0, 1, 2 => spa, bil, cat
        self.info = {'language': language, 'age': age, 'num_children': num_children} # TODO : group marital/parental info in dict ??
        self.lang_thresholds = {'speak': lang_act_thresh, 'understand': lang_passive_thresh}
        self.loc_info = {'home': ag_home, 'school': ag_school, 'job': ag_job}

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
            clust_ix = self.loc_info['home'].clust
            clust_schools_coords = [sc.pos for sc in self.model.geo.clusters_info[clust_ix]['schools']]
            closest_school_idx = np.argmin([pdist([self.loc_info['home'].pos, sc_coord])
                                            for sc_coord in clust_schools_coords])
            # instantiate new agent
            a = LanguageAgent(self.model, id_, lang, ag_home=self.loc_info['home'],
                              ag_school=self.model.geo.clusters_info[clust_ix]['schools'][closest_school_idx],
                              ag_job=None)
            # Add agent to model
            self.model.geo.add_agents_to_grid_and_schedule(a)
            self.model.nws.add_ags_to_networks(a)
            # add newborn agent to home presence list
            a.loc_info['home'].agents_in.add(a)
            # Update num of children
            self.info['num_children'] += 1

    def random_death(self, a=1.23368173e-05, b=2.99120806e-03, c=3.19126705e+01):
        """ Method to randomly determine agent death or survival at each step
            The fitted function provides the death likelihood for a given rounded age
            In order to get the death-probability per step we divide
            by number of steps in a year (36)
            Fitting parameters are from
            https://www.demographic-research.org/volumes/vol27/20/27-20.pdf
            " Smoothing and projecting age-specific probabilities of death by TOPALS "
            by Joop de Beer
            Resulting life expectancy is 77 years and std is ~ 15 years
        """
        if random.random() < a * (np.exp(b * self.info['age']) + c) / self.model.steps_per_year:
            self.model.remove_after_death(self)

    def look_for_job(self):
        """ Method for agent to look for a job """
        # loop through shuffled job centers list until a job is found
        np.random.shuffle(self.model.geo.clusters_info[self.loc_info['home'].clust]['jobs'])
        for job_c in self.model.geo.clusters_info[self.loc_info['home'].clust]['jobs']:
            if job_c.num_places:
                job_c.num_places -= 1
                self.loc_info['job'] = job_c
                job_c.info['employees'].add(self)
                job_c.agents_in.add(self)
                break

    def update_acquaintances(self, other, lang):
        """ Add edges to known_people network when meeting (speaking) for first time """
        if other not in self.model.nws.known_people_network[self]:
            self.model.nws.known_people_network.add_edge(self, other)
            self.model.nws.known_people_network[self][other].update({'num_meet': 1, 'lang': lang})
        elif (other not in self.model.nws.family_network[self]) and (other not in self.model.nws.friendship_network[self]):
            self.model.nws.known_people_network[self][other]['num_meet'] += 1

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

        # sample must come from AVAILABLE words in R (retrievability) !!!! This can be modeled in following STEPS

        # 1. First sample from lang CDF ( that encapsulates all to-be-known concepts at a given age-step)
        # These are the thoughts or concepts a speaker tries to convey
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
                if not isinstance(self.loc_info['job'], list):
                    job = self.loc_info['job']
                else:
                    job = self.loc_info['job'][0]
                self.model.grid.move_agent(self, job.pos)
                job.agents_in.add(self)
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
        if random.random() > 0.5 and self.model.nws.friendship_network[self]:
            picked_friend = np.random.choice(self.model.nws.friendship_network.neighbors(self))
            self.start_conversation(with_agents=picked_friend)
        self.model.grid.move_agent(self, self.loc_info['home'].pos)
        self.loc_info['home'].agents_in.add(self)
        try:
            for key in self.model.nws.family_network[self]:
                if key.pos == self.loc_info['home'].pos:
                    lang = self.model.nws.family_network[self][key]['lang']
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

