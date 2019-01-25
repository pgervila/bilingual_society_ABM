import random

from mesa.time import StagedActivation
from .agent import IndepAgent, Young


class StagedActivationModif(StagedActivation):
    # TODO : add/separate agents by type ??? Is it a good idea ??

    def step(self, pct_threshold=0.9):
        """ Executes all the stages for all agents """
        #
        for ag in self.agents[:]:
            # set conversation counters to zero when step begins
            ag._conv_counts_per_step = 0
            # new step -> older age
            ag.grow()
            for lang in ['L1', 'L12', 'L21', 'L2']:
                # save copy of wc for each agent
                ag.wc_init[lang] = ag.lang_stats[lang]['wc'].copy()
                ag.update_word_activation_elapsed_time(lang)
                ag.update_memory_retention(lang, pct_threshold=pct_threshold)
                ag.reset_step_mask(lang)
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
            # An agent might have changed type if hired as Teacher to replace dead one
            # Old agent instance will be removed from model but it might still be in the copied list !
            try:
                if isinstance(ag, Young):
                    ag.reproduce()
                ag.random_death()
            except KeyError:
                pass

        # loop and update courses in schools and universities year after year
        # update jobs lang policy
        if self.steps and not self.steps % self.model.steps_per_year:
            self.model.update_centers()

        self.steps += 1

    def replace_agent(self, old_agent, new_agent):
        ix_in_schedule = self.model.schedule.agents.index(old_agent)
        self.model.schedule.remove(old_agent)
        self.model.schedule.agents.insert(ix_in_schedule, new_agent)

