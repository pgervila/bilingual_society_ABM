# IMPORT LIBS
import random
import numpy as np
import networkx as nx
from collections import deque

class Simple_Language_Agent:

    def __init__(self, model, unique_id, language, S, lang_pride=0.5, financial_greed=0.5):
        self.model = model
        self.unique_id = unique_id
        self.language = language  # 0, 1, 2 => spa, bil, cat
        self.S = S
        self.lang_pride = lang_pride
        self.financial_greed = financial_greed

        self.job = None
        # model agent relative financial wealth : relative to average financial wealth
        self.wealth = 100
        self.relat_wealth = 1 # endowment
        # model agent relative lang wealth : how often favorite lang can be used relative to competitor lang
        if self.language in [0,2]:
            self.lang_wealth = 1

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
        if with_agent is None:
            pos = [self.pos]
            # get all agents currently placed on chosen cell
            others = self.model.grid.get_cell_list_contents(pos)
            ## linguistic model of encounter with another random agent
            if len(others) > 1:
                other = random.choice(others)
                self.get_conversation_lang(other)
                # update lang status
                self.update_lang_status()
                other.update_lang_status()
        else:
            self.get_conversation_lang(with_agent)
            other = with_agent
            # update lang status
            self.update_lang_status()
            other.update_lang_status()

    def get_conversation_lang(self, other):
        # spa-bilingual
        if (self.language, other.language) in [(0,0),(0,1),(1,0)]:
            for key in ['heard','spoken']:
                self.lang_freq[key][0] += 1
                other.lang_freq[key][0] += 1
            self.lang_freq['maxmem_list'].append(0)
            other.lang_freq['maxmem_list'].append(0)
        # bilingual-cat
        elif (self.language, other.language) in [(2,1),(1,2),(2,2)]:
            for key in ['heard', 'spoken']:
                self.lang_freq[key][1] += 1
                other.lang_freq[key][1] += 1
            self.lang_freq['maxmem_list'].append(1)
            other.lang_freq['maxmem_list'].append(1)
        # bilingual-bilingual
        elif (self.language, other.language) == (1, 1):
            # find out lang spoken by self
            if sum(self.lang_freq['spoken']) != 0:
                p10 = (2/3 * self.lang_freq['spoken'][0]/sum(self.lang_freq['spoken']) +
                       1/3 * self.lang_freq['heard'][0]/sum(self.lang_freq['heard'])
                       )
                p11 = 1 - p10
                l1 = np.random.binomial(1,p11)
            else:
                l1 = random.choice([0,1])
            self.lang_freq['spoken'][l1] += 1
            other.lang_freq['heard'][l1] += 1
            # find out language spoken by other
            self.lang_freq['heard'][l1] += 1
            other.lang_freq['spoken'][l1] += 1
            self.lang_freq['maxmem_list'].append(l1)
            other.lang_freq['maxmem_list'].append(l1)
        # spa-cat
        else:
            if sum(self.lang_freq['spoken']) != 0:
                p10 = self.lang_freq['spoken'][0]/sum(self.lang_freq['spoken'])
                p11 = 1 - p10
                l1 = np.random.binomial(1, p11)
            else:
                l1 = random.choice([0, 1])
            self.lang_freq['spoken'][l1] += 1
            other.lang_freq['heard'][l1] += 1
            # find out language spoken by other
            if sum(other.lang_freq['spoken']) != 0:
                p20 = other.lang_freq['heard'][0]/sum(other.lang_freq['heard'])
                p21 = 1 - p20
                l2 = np.random.binomial(1,p21)
            else:
                l2 = random.choice([0, 1])
            self.lang_freq['heard'][l2] += 1
            other.lang_freq['spoken'][l2] += 1
            self.lang_freq['maxmem_list'].append(l1)
            other.lang_freq['maxmem_list'].append(l2)

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
        if self.model.schedule.steps > self.lang_freq['maxmem']:
            if self.language == 0:
                if self.lang_freq['cat_pct_h'] >= 0.25:
                    self.language = 1
            elif self.language == 2:
                if self.lang_freq['cat_pct_h'] <= 0.75:
                    self.language = 1
            else:
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

    def look_for_job(self):
        """ Compute probabilities based on language knowledge """
        if self.language in [0,1]:
            if np.random.binomial(1, (1 - self.lang_freq['cat_pct_s'])/10):
                self.job = True
        elif self.language == 2:
            if np.random.binomial(1, (1 - self.lang_freq['cat_pct_s'])/100):
                self.job = True

    def work(self):
        self.speak()
        self.wealth += 1

    def get_job_review(self):
        if self.lang_freq['cat_pct_s'] > 0.75:
            self.job = False

    def study_lang(self, lang):
        self.lang_freq['spoken'][lang] += 1
        self.lang_freq['heard'][lang] += 1

    def assess_wellness(self):
        """Assess how well or bad an agent feels as a
        result of combining the two parameters that define wellness:
        lang pride and financial wealth. The assessment will be
        used to define agent further action"""

        # wealth will be relative to other's wealth
        # lang pride as well
        if not self.job:
            self.study_lang(0)



    def step(self):
        #check if agent has a job
        if self.job:
            self.work()
            self.get_job_review()
        else:
            self.wealth -= 1
            self.look_for_job()
        #do usual stuff
        self.move_random()
        self.speak()
        #assess
        self.assess_wellness()
        #remove agent if wealth is over
        if self.wealth == 0:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)


    def __repr__(self):
        return 'Lang_Agent_{0.unique_id!r}'.format(self)