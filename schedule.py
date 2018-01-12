from mesa.time import StagedActivation
#from agent import Young

import numpy as np
import random


class StagedActivationModif(StagedActivation):
    # TODO : add/separate agents by type ??? Is it a good idea ??

    def step(self):
        """ Executes all the stages for all agents. """
        # basic IDEA: network adj matrices will be fixed through all stages of one step
        # first
        for agent in self.agents[:]:
            agent.info['age'] += 1
            for lang in ['L1', 'L12', 'L21', 'L2']:
                # update last-time word use vector
                agent.lang_stats[lang]['t'][~agent.day_mask[lang]] += 1
                # set current lang knowledge
                agent.lang_stats[lang]['pct'][agent.info['age']] = (np.where(agent.lang_stats[lang]['R'] > 0.9)[0].shape[0] /
                                                                    agent.model.vocab_red)
                # reset day mask
                agent.day_mask[lang] = np.zeros(agent.model.vocab_red, dtype=np.bool)
            # Update lang switch
            agent.update_lang_switch()
        if self.shuffle:
            random.shuffle(self.agents)
        for stage in self.stage_list:
            for agent in self.agents[:]:
                getattr(agent, stage)()  # Run stage
            if self.shuffle_between_stages:
                random.shuffle(self.agents)
            self.time += self.stage_time
        # simulate reproduction and death chances
        for agent in self.agents[:]:
            # TODO: if isinstance(agent, Young)
            agent.reproduce()
            agent.random_death()

        # loop and update courses in schools and universities year after year
        if not self.steps % 36:
            for clust_idx, clust_info in self.model.clusters_info.items():
                if 'university' in clust_info:
                    for fac in clust_info['university'].faculties.values():
                        if fac.info['students']:
                            fac.update_courses()
                for school in clust_info['schools']:
                    school.update_courses()
                    if not self.steps % 72: # every 2 years only, teachers swap
                        school.teachers_course_swap()
        self.steps += 1
