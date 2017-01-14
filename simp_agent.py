# IMPORT LIBS
import random
import numpy as np
import networkx as nx

class Simple_Language_Agent:

    def __init__(self, model, unique_id, language, S):
        self.model = model
        self.unique_id = unique_id
        self.language = language # 0, 1, 2 => spa, bil, cat
        self.S = S

        self.lang_freq = dict()
        self.lang_freq['spoken'] = [0, 0] # 0, 2 => spa, cat
        self.lang_freq['heard'] = [0, 0]
        self.lang_freq['cat_pct_s'] = 0
        self.lang_freq['cat_pct_h'] = 0


    def move_random(self):
        """ Take a random step into any surrounding cell
            All eight surrounding cells are available as choices
            Current cell is not an output choice

            Arguments:
                * model : ABM model

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
                * model : ABM model
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
        # bilingual-cat
        elif (self.language, other.language) in [(2,1),(1,2),(2,2)]:
            for key in ['heard', 'spoken']:
                self.lang_freq[key][1] += 1
                other.lang_freq[key][1] += 1
        # bilingual-bilingual
        elif (self.language, other.language) == (1, 1):
            # find out lang spoken by self
            if sum(self.lang_freq['spoken']) != 0:
                p10 = self.lang_freq['spoken'][0]/sum(self.lang_freq['spoken'])
                p11 = 1 - p10
                #l1 = np.random.choice([0,1], p=[p10,p11])
                l1 = np.random.binomial(1,p11,1)[0]
            else:
                l1 = random.choice([0,1])
            self.lang_freq['spoken'][l1] += 1
            other.lang_freq['heard'][l1] += 1
            # find out language spoken by other
            self.lang_freq['heard'][l1] += 1
            other.lang_freq['spoken'][l1] += 1
        # spa-cat
        else:
            if sum(self.lang_freq['spoken']) != 0:
                p10 = self.lang_freq['spoken'][0]/sum(self.lang_freq['spoken'])
                p11 = 1 - p10
                #l1 = np.random.choice([0,1], p=[p10,p11])
                l1 = np.random.binomial(1, p11, 1)[0]
            else:
                l1 = random.choice([0, 1])
            self.lang_freq['spoken'][l1] += 1
            other.lang_freq['heard'][l1] += 1
            # find out language spoken by other
            if sum(other.lang_freq['spoken']) != 0:
                p20 = other.lang_freq['heard'][0]/sum(other.lang_freq['heard'])
                p21 = 1 - p20
                #l2 = np.random.choice([0,1], p=[p20,p21])
                l2 = np.random.binomial(1,p21,1)[0]
            else:
                l2 = random.choice([0, 1])
            self.lang_freq['heard'][l2] += 1
            other.lang_freq['spoken'][l2] += 1

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
        if self.model.schedule.steps > 50:
            if self.language == 0:
                if self.lang_freq['cat_pct_h'] >= 0.25:
                    self.language = 1
            elif self.language == 2:
                if self.lang_freq['cat_pct_h'] <= 0.75:
                    self.language = 1
            else:
                if self.lang_freq['cat_pct_h'] >= 0.9:
                    self.language = 2
                elif self.lang_freq['cat_pct_h'] <= 0.1:
                    self.language = 0
    def update_lang_status(self):
        # update lang experience
        self.update_lang_pcts()
        # check lang switch
        self.update_lang_switch()


    def step(self):
        self.move_random()
        self.speak()

    def __repr__(self):
        return 'Lang_Agent_{0.unique_id!r}'.format(self)