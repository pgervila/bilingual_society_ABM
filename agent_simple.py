# IMPORT LIBS
import random
import numpy as np
import networkx as nx
from collections import deque, Counter, defaultdict



class Simple_Language_Agent:

    def __init__(self, model, unique_id, language, age=0, avg_learning_hours=1000,
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

        # MEMORY MODEL
        # keep container, 's' and 'l' keys
        # subkeys retrievability, stability, last activat time: 'R', 'S', 't'


        num_init_occur = 100

        # Add randomness to number of hours needed to learn second language
        self.lang_stats['maxmem'] = np.random.poisson(self.model.avg_max_mem)
        # define hours needed for agent to be able to converse in other language
        self.lang_stats['learning_hours'] = np.random.normal(avg_learning_hours, 100)
        # define container to track last activation steps for each language
        self.lang_stats['last_activ_step'] = [None, None]
        if self.language == 0:
            for action in ['s', 'l']:
                self.lang_stats[action]['freqs'] = [np.random.poisson(num_init_occur), 0]

                self.lang_stats[action]['freqs'] = np.zeros((2, self.model.max_run_steps), dtype=np.uint8)

                self.lang_stats[action]['L2_pct'] = 0.
                self.lang_stats[action]['ST']['freqs'] = [deque([], maxlen=self.lang_stats['maxmem']),
                                                          deque([], maxlen=self.lang_stats['maxmem'])]
                self.lang_stats[action]['ST']['L2_pct'] = 0.
                self.lang_stats[action]['ST']['step_counts'] = [0, 0]
        elif self.language == 2:
            for action in ['s', 'l']:
                self.lang_stats[action]['LT']['freqs'] = [0, np.random.poisson(num_init_occur)]
                self.lang_stats[action]['LT']['L2_pct'] = 1.
                self.lang_stats[action]['ST']['freqs'] = [deque([], maxlen=self.lang_stats['maxmem']),
                                                          deque([], maxlen=self.lang_stats['maxmem'])]
                self.lang_stats[action]['ST']['L2_pct'] = 1.
                self.lang_stats[action]['ST']['step_counts'] = [0, 0]
        else:
            for action in ['s', 'l']:
                f1 = np.random.poisson(num_init_occur/2)
                f2 = np.random.poisson(num_init_occur/2)
                self.lang_stats[action]['LT']['freqs'] = [f1, f2]
                self.lang_stats[action]['LT']['L2_pct'] = f2 / (f1 + f2)
                self.lang_stats[action]['ST']['freqs'] = [deque([], maxlen=self.lang_stats['maxmem']),
                                                          deque([], maxlen=self.lang_stats['maxmem'])]
                self.lang_stats[action]['ST']['L2_pct'] = f2 / (f1 + f2)
                self.lang_stats[action]['ST']['step_counts'] = [0, 0]

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


    def update_lang_counter(self, ag_1, ag_2, l1, l2):
        """ Update counts of LT and ST lang arrays for two agents
            Args:
                * ag_1, ag_2 : agents objects
                * l1, l2 : integers in [0,1]. Languages spoken by ag_1 and ag_2

        """
        ag_1.lang_stats['s']['LT']['freqs'][l1] += 1
        ag_1.lang_stats['s']['ST']['freqs'][l1][-1] += 1
        ag_1.lang_stats['l']['LT']['freqs'][l2] += 1
        ag_1.lang_stats['l']['ST']['freqs'][l2][-1] += 1

        ag_2.lang_stats['s']['LT']['freqs'][l2] += 1
        ag_2.lang_stats['s']['ST']['freqs'][l2][-1] += 1
        ag_2.lang_stats['l']['LT']['freqs'][l1] += 1
        ag_2.lang_stats['l']['ST']['freqs'][l1][-1] += 1


    def get_conversation_lang(self, ag_1, ag_2, return_values=False):

        if (ag_1.language, ag_2.language) in [(0, 0), (0, 1), (1, 0)]:# spa-bilingual
            l1 = l2 = 0
            self.update_lang_counter(ag_1, ag_2, 0, 0)

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




    def words_day_factor(self):
        """ Define coeff that determines num hours spoken per day
            as pct of vocabulary size, assuming 16000 tokens per adult per day
            as average """
        if self.age < 36 * 14:
            return 2 + 100 * np.exp(-0.014 * self.age)
        elif 36 * 14 <= self.age <= 36 * 65:
            return 2
        elif self.age > 36 * 65:
            return 2 + np.exp(0.002 * (self.age - 36 * 65))



    def get_lang_knowledge(self, t, S, cdf_data, word_counter, num_steps1, num_steps2,
                           pct_hours, lang_knowledge,
                           vocab_size=40000, a=7.6, b=0.023, c=-0.031, d=-0.2, n_red=1000):

        for age in range(num_steps1, num_steps2):

            if not self.age:
                self.R = np.zeros(n_red)
            else:
                self.R = np.exp(- k * t / S)

            days = n_red / self.words_day_factor()
            if self.age > 7 * 36 and random.random() > 0.1:
                zipf_samples = randZipf(cdf_data['speech'][age], int(0.9 * pct_hours * days * 10))
                zipf_samples_written = randZipf(cdf_data['written'][age], int(0.1 * pct_hours * days * 10))
                zipf_samples = np.concatenate((zipf_samples, zipf_samples_written))
            else:
                zipf_samples = randZipf(cdf_data['speech'][age], int(pct_hours * days * 10))

            activated, activ_counts = np.unique(zipf_samples, return_counts=True)

            np.add.at(word_counter, activated, activ_counts)
            availab_for_activ = np.where(word_counter > 3)[0]
            activated = np.intersect1d(availab_for_activ, activated, assume_unique=True)

            if activated.any():
                delta_S = a * (S[activated] ** (-b)) * np.exp(c * 100 * R[activated]) + d
                np.add.at(S, activated, delta_S)

                int_bool = np.nonzero(np.in1d(activated, availab_for_activ, assume_unique=True))

                activated = activated[int_bool]  # call it act

                activ_counts = activ_counts[int_bool]  # call it act_c

                mask = np.zeros(n_red, dtype=np.bool)
                mask[activated] = True
                t[~mask] += 1  # add ones to last activation time counter if word not activated
                t[mask] = 0  # set last activation time counter to one if word activated

                activ_counts -= 1
                delta_S = activ_counts * (a * (S[activated] ** (-b)) * np.exp(c * 100 * R[activated]) + d)
                np.add.at(S, activated, delta_S)

            else:
                mask = np.zeros(n_red, dtype=np.bool)
                mask[activated] = True
                t[~mask] += 1  # add ones to last activation time counter if word not activated
                t[mask] = 0  # set last activation time counter to one if word activated

            if age > 60 * 36:
                S = np.where(S >= 0.01, S - 0.01, 0.000001)

            lang_knowledge[age] = np.where(R > 0.9)[0].shape[0] / n_red

        return t, S, lang_knowledge



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
        # add new step to ST deques
        for action in ['s', 'l']:
            self.lang_stats[action]['ST']['freqs'][0].append(0)
            self.lang_stats[action]['ST']['freqs'][1].append(0)
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