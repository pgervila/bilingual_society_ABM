# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np

from mesa.time import StagedActivation
from agent import IndepAgent, Young


class StagedActivationModif(StagedActivation):
    # TODO : add/separate agents by type ??? Is it a good idea ??

    def step(self, pct_threshold=0.9):
        """ Executes all the stages for all agents """

        for ag in self.agents[:]:
            # set conversation counters to zero when step begins
            ag.call_cnts_final = ag.call_cnts_init = 0

            # new step -> older age
            ag.grow()
            # set exclusion counter to zero
            ag.lang_stats['L1' if ag.info['language'] == 2 else 'L2']['excl_c'][ag.info['age']] = 0

            for lang in ['L1', 'L12', 'L21', 'L2']:
                # save copy of wc for each agent
                ag.wc_init[lang] = ag.lang_stats[lang]['wc'].copy()

                # update last-time word use vector
                ag.lang_stats[lang]['t'][~ag.step_mask[lang]] += 1
                # compute new memory retrievability R using updated t values
                ag.lang_stats[lang]['R'] = np.exp(-ag.k * ag.lang_stats[lang]['t'] / ag.lang_stats[lang]['S'])

                # set current lang knowledge
                # compute current language knowledge in percentage after 't' update
                ag.update_lang_knowledge(lang, pct_threshold=pct_threshold)

                # reset day mask
                ag.step_mask[lang] = np.zeros(ag.model.vocab_red, dtype=np.bool)
            # Update lang switch
            ag.update_lang_switch()
        if self.shuffle:
            random.shuffle(self.agents)

        # Network adj matrices will be constant through all stages of one step
        self.model.nws.compute_adj_matrices()

        for stage in self.stage_list:
            for ix_ag, ag in enumerate(self.agents):
                if isinstance(ag, IndepAgent):
                    getattr(ag, stage)(ix_ag)
                else:
                    getattr(ag, stage)()
            if self.shuffle_between_stages:
                random.shuffle(self.agents)
            self.time += self.stage_time

        for ag in self.agents[:]:
            for lang in ['L1', 'L12', 'L21', 'L2']:
                ag.wc_final[lang] = ag.lang_stats[lang]['wc'].copy()

        # check reproduction, death : make shallow copy of agents list,
        # since we are potentially removing agents as we iterate
        for ag in self.agents[:]:
            if isinstance(ag, Young):
                ag.reproduce()
            ag.random_death()
        # loop and update courses in schools and universities year after year
        # update jobs lang policy
        if not self.steps % self.model.steps_per_year and self.steps:
            for clust_idx, clust_info in self.model.geo.clusters_info.items():
                if 'university' in clust_info:
                    for fac in clust_info['university'].faculties.values():
                        if fac.info['students']:
                            fac.update_courses_phase_1()
                for school in clust_info['schools']:
                    school.update_courses_phase_1()
                for job in clust_info['jobs']:
                    job.set_lang_policy()
            for clust_idx, clust_info in self.model.geo.clusters_info.items():
                if 'university' in clust_info:
                    for fac in clust_info['university'].faculties.values():
                        if fac.info['students']:
                            fac.update_courses_phase_2()
                for school in clust_info['schools']:
                    school.update_courses_phase_2()
                    if not self.steps % (4 * self.model.steps_per_year):  # every 4 years only, teachers swap
                        school.swap_teachers_courses()

        self.steps += 1

    def replace_agent(self, old_agent, new_agent):
        ix_in_schedule = self.model.schedule.agents.index(old_agent)
        self.model.schedule.remove(old_agent)
        self.model.schedule.agents.insert(ix_in_schedule, new_agent)

