# IMPORT RELEVANT LIBRARIES
import os, sys
from importlib import reload
import random
import numpy as np
import networkx as nx
# import matplotlib
# matplotlib.use("TKAgg")
import matplotlib.pylab as plt
import matplotlib.animation as animation
import pyprind
import deepdish as dd

# IMPORT MESA LIBRARIES ( Model, Grid, Schedule )
from mesa import Model
from mesa.space import MultiGrid
import schedule
reload(sys.modules['schedule'])
from schedule import StagedActivationModif

# IMPORT MODEL LIBRARIES
import geomapping, networks, dataprocess
reload(sys.modules['geomapping'])
reload(sys.modules['networks'])
reload(sys.modules['dataprocess'])
from geomapping import GeoMapper
from networks import Networks
from dataprocess import DataProcessor, DataViz



class LanguageModel(Model):

    ic_pct_keys = [10, 25, 50, 75, 90]
    family_size = 4
    school_lang_policy = [1]
    steps_per_year = 36

    def __init__(self, num_people, spoken_only=True, num_words_conv=(3, 25, 250),
                 width=100, height=100,
                 max_people_factor=5, init_lang_distrib=[0.25, 0.65, 0.1],
                 num_clusters=10, max_run_steps=1000,
                 lang_ags_sorted_by_dist=True, lang_ags_sorted_in_clust=True):
        self.num_people = num_people
        if spoken_only:
            self.vocab_red = 500
        else:
            self.vocab_red = 1000
        self.num_words_conv = num_words_conv
        self.grid_width = width
        self.grid_height = height
        self.max_people_factor = max_people_factor
        self.init_lang_distrib = init_lang_distrib
        self.num_clusters = num_clusters
        self.max_run_steps = max_run_steps
        self.lang_ags_sorted_by_dist = lang_ags_sorted_by_dist
        self.lang_ags_sorted_in_clust = lang_ags_sorted_in_clust
        # self.clust_centers = None
        # self.cluster_sizes = None
        self.random_seeds = np.random.randint(1, 10000, size=2)


        #define container for available ids
        self.set_available_ids = set(range(num_people, max_people_factor * num_people))

        # import lang ICs and lang CDFs data as function of steps. Use directory of executed file
        self.lang_ICs = dd.io.load(os.path.join(os.path.dirname(__file__), 'lang_spoken_ics_vs_step.h5'))
        self.cdf_data = dd.io.load(os.path.join(os.path.dirname(__file__), 'lang_cdfs_vs_step.h5'))

        # define model grid and schedule
        self.grid = MultiGrid(height, width, False)
        self.schedule = StagedActivationModif(self,
                                              stage_list=["stage_1", "stage_2",
                                                          "stage_3", "stage_4"],
                                              shuffle=True,
                                              shuffle_between_stages=False)

        # define model geomapping and model networks
        self.geo = GeoMapper(self, num_clusters)
        self.nws = Networks(self)
        # define datacollector and dataprocessor
        self.data_process = DataProcessor(self)
        # define dataviz
        self.data_viz = DataViz(self)

    @staticmethod
    def define_lang_interaction(ag1, ag2, ret_pcts=False):
        """ Method to find out lang of interaction between two given agents """
        # compute lang knowledge for each agent
        age1, age2 = ag1.info['age'], ag2.info['age']
        pct11 = max(ag1.lang_stats['L1']['pct'][age1],
                    ag1.lang_stats['L12']['pct'][age1])
        pct12 = max(ag1.lang_stats['L21']['pct'][age1],
                    ag1.lang_stats['L2']['pct'][age1])
        pct21 = max(ag2.lang_stats['L1']['pct'][age2],
                    ag2.lang_stats['L12']['pct'][age2])
        pct22 = max(ag2.lang_stats['L21']['pct'][age2],
                    ag2.lang_stats['L2']['pct'][age2])
        if (ag1.info['language'], ag2.info['language']) in [(0, 0), (0, 1), (1, 0)]:
            lang = 0
        elif (ag1.info['language'], ag2.info['language']) in [(2, 1), (1, 2), (2, 2)]:
            lang = 1
        elif (ag1.info['language'], ag2.info['language']) == (1, 1):
            # Find weakest combination lang-agent, pick other lang as common one
            idx_weakest = np.argmin([pct11, pct12, pct21, pct22])
            if idx_weakest in [0, 2]:
                lang = 1
            else:
                lang = 0
        if ret_pcts:
            return lang, [pct11, pct12, pct21, pct22]
        else:
            return lang

    def run_conversation(self, ag_init, others, bystander=None, num_days=10):
        """ Method that models conversation between ag_init and others
            Calls method to determine conversation parameters
            Then makes each speaker speak and the rest listen (loop through all involved agents)
            Args:
                * ag_init : agent object instance. Agent that starts conversation
                * others : list of agent class instances. Rest of agents that take part in conversation
                    It can be a single agent object that will be automatically converted into a list
                * bystander: extra agent that may listen to conversation words without actually being involved.
                    Agent vocabulary gets correspondingly updated if bystander agent is specified
                * num_days: integer [1, 10]. Number of days in one 10day-step this kind of speech is done
        """
        # define list of all agents involved
        ags = [ag_init]
        ags.extend(others) if (type(others) is list) else ags.append(others)
        # get all parameters of conversation
        conv_params = self.get_conv_params(ags)
        for ix, (ag, lang) in enumerate(zip(ags, conv_params['lang_group'])):
            if ag.info['language'] != conv_params['mute_type']:
                spoken_words = ag.pick_vocab(lang, long=conv_params['long'], num_days=num_days)
                # call 'self' agent update
                ag.update_lang_arrays(spoken_words)
                # call listeners' updates ( check if there is a bystander)
                listeners = ags[:ix] + ags[ix + 1:] + [bystander] if bystander else ags[:ix] + ags[ix + 1:]
                for listener in listeners:
                    listener.update_lang_arrays(spoken_words, speak=False)
        # update acquaintances
        if type(others) is list:
            for ix, ag in enumerate(others):
                if ag.info['language'] != conv_params['mute_type']:
                    ag_init.update_acquaintances(ag, conv_params['lang_group'][0])
                    ag.update_acquaintances(ag_init, conv_params['lang_group'][ix])
        else:
            if others.info['language'] != conv_params['mute_type']:
                ag_init.update_acquaintances(others, conv_params['lang_group'][0])
                others.update_acquaintances(ag_init, conv_params['lang_group'][1])

    def get_conv_params(self, ags):
        """
        Method to find out parameters of conversation between 2 or more agents:
            conversation lang or lang spoken by each involved speaker,
            conversation type(mono or bilingual),
            mute agents (agents that only listen),
            conversation length.
        It implements MAXIMIN language rule from Van Parijs
        Args:
            * ags : list of agent class instances. All agents that take part in conversation
        Returns:
            * conv_params dict with following keys:
                * lang: integer in [0, 1] if unique lang conv or list of integers in [0, 1] if multilang conversation
                * mute_type: integer. Agent lang type that is unable to speak in conversation
                * bilingual: boolean. True if conv is held in more than one language
                * long: boolean. True if conv is long
        """

        # redefine separate agents for readability
        ag_init = ags[0]
        others = ags[1:]

        # set output default parameters
        conv_params = dict(multilingual=False, mute_type=None, long=True)
        # define lists with agent competences and preferences in each language
        l1_pcts = [ag.lang_stats['L1']['pct'][ag.info['age']] for ag in ags]
        l2_pcts = [ag.lang_stats['L2']['pct'][ag.info['age']] for ag in ags]
        # get lists of favorite language per agent and set of language types involved
        fav_lang_per_agent = list(np.argmax([l1_pcts, l2_pcts], axis=0))
        ags_lang_types = set([ag.info['language'] for ag in ags])

        # define current case
        if ags_lang_types in [{0}, {0, 1}]: # TODO: need to save info of how init wanted to talk-> Feedback for AI learning
            lang_group = 0
            conv_params.update({'lang_group': lang_group})
        elif ags_lang_types == {1}:
            # simplified PRELIMINARY NEUTRAL assumption: ag_init will start speaking the language they speak best
            # ( TODO : at this stage no modeling of place bias !!!!)
            # who starts conversation matters, but also average lang spoken with already known agents
            if l1_pcts[0] == l2_pcts[0]:
                lang_init = 1 if random.random() > 0.5 else 0
            else:
                lang_init = np.argmax([l1_pcts[0], l2_pcts[0]])
            # TODO : why known agents only checked for this option ??????????????
            langs_with_known_agents = [self.nws.known_people_network[ag_init][ag]['lang']
                                       for ag in others
                                       if ag in self.nws.known_people_network[ag_init]]
            if langs_with_known_agents:
                lang_group = round(sum(langs_with_known_agents) / len(langs_with_known_agents))
            else:
                lang_group = lang_init
            conv_params.update({'lang_group': lang_group})
        elif ags_lang_types in [{1, 2}, {2}]:
            lang_group = 1
            conv_params.update({'lang_group': lang_group})
        else:
            # monolinguals on both linguistic sides => VERY SHORT CONVERSATION
            # get agents on both lang sides unable to speak in other lang
            idxs_real_monolings_l1 = [idx for idx, pct in enumerate(l2_pcts) if pct < 0.025]
            idxs_real_monolings_l2 = [idx for idx, pct in enumerate(l1_pcts) if pct < 0.025]

            if not idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # No complete monolinguals on either side
                # All agents partially understand each other langs, but some can't speak l1 and some can't speak l2
                # Conversation is possible when each agent picks their favorite lang
                lang_group = fav_lang_per_agent
                conv_params.update({'lang_group': lang_group, 'multilingual':True, 'long':False})

            elif idxs_real_monolings_l1 and not idxs_real_monolings_l2:
                # There are real L1 monolinguals in the group
                # Everybody partially understands L1, but some agents don't understand L2 at all
                # Some agents only understand and speak L1, while others partially understand but can't speak L1
                # slight bias towards l1 => conversation in l1 but some speakers will stay mute = > short conversation
                mute_type = 2
                if ag_init.info['language'] != mute_type:
                    lang_group = 0
                else:
                    lang_group, mute_type = 1, 0
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'long': False})

            elif not idxs_real_monolings_l1 and idxs_real_monolings_l2:
                # There are real L2 monolinguals in the group
                # Everybody partially understands L2, but some agents don't understand L1 at all
                # Some agents only understand and speak l2, while others partially understand but can't speak l2
                # slight bias towards l2 => conversation in L2 but some speakers will stay mute = > short conversation
                mute_type = 0
                if ag_init.info['language'] != mute_type:
                    lang_group = 1
                else:
                    lang_group, mute_type = 0, 2
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'long': False})

            else:
                # There are agents on both lang sides unable to follow other's speech.
                # Initiator agent will speak with whom understands him, others will listen but understand nothing

                if ag_init.info['language'] == 1:
                    # init agent is bilingual
                    # pick majority lang
                    num_l1_speakers = sum([1 if pct >= 0.1 else 0 for pct in l1_pcts])
                    num_l2_speakers = sum([1 if pct >= 0.1 else 0 for pct in l2_pcts])
                    if num_l1_speakers > num_l2_speakers:
                        lang_group, mute_type = 0, 2
                    elif num_l1_speakers < num_l2_speakers:
                        lang_group, mute_type = 2, 0
                    else:
                        lang_group = fav_lang_per_agent[0]
                        mute_type = 2 if lang_group == 0 else 0
                else:
                    # init agent is monolang
                    lang_group = fav_lang_per_agent[0]
                    mute_type = 2 if lang_group == 0 else 0
                conv_params.update({'lang_group': lang_group, 'mute_type': mute_type, 'long': False})

        if not conv_params['multilingual']:
            conv_params['lang_group'] = [conv_params['lang_group']] * len(ags)

        conv_params['min_group_age'] = min([ag.info['age'] for ag in ags])

        return conv_params

    def remove_after_death(self, agent):
        """ Removes agent object from all places where it belongs.
            It makes sure no references to agent object are left after removal,
            so that garbage collector can free memory
            Call this function if death conditions for agent are verified
        """
        # Remove agent from all networks
        for network in [self.nws.family_network,
                        self.nws.known_people_network,
                        self.nws.friendship_network]:
            try:
                network.remove_node(agent)
            except nx.NetworkXError:
                continue
        # remove agent from all locations where it might be
        for loc, attr in zip([agent.loc_info['home'], agent.loc_info['job'], agent.loc_info['school']],
                             ['occupants','employees','students']):
            try:
                getattr(loc, attr).remove(agent)
                loc.agents_in.remove(agent)
            except:
                continue
        # remove agent from city
        self.geo.clusters_info[agent.loc_info['city_idx']]['agents'].remove(agent)
        # remove agent from grid and schedule
        self.grid._remove_agent(agent.pos, agent)
        self.schedule.remove(agent)
        # make id from deceased agent available
        self.set_available_ids.add(agent.unique_id)

    def step(self):
        self.data_process.collect()
        self.schedule.step()

    def run_model(self, steps, recording_steps_period=None, save_dir=''):
        """ Run model and save frames if required
            Args
                * steps: integer. Total steps to run
                * recording_steps_period : integer. Save frames every specified number of steps
                * save_dir : string. It specifies directory where frames will be saved
        """
        pbar = pyprind.ProgBar(steps)
        self.save_dir = save_dir
        if recording_steps_period:
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, save_dir)
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
        for _ in range(steps):
            self.step()
            if recording_steps_period:
                if not self.schedule.steps%recording_steps_period:
                    self.show_results(step=self.schedule.steps, plot_results=False, save_fig=True)
            pbar.update()

    def run_and_animate(self, steps, plot_type='imshow'):
        fig = plt.figure()
        grid_size = (3, 5)
        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        ax1.set_xlim(0, steps)
        ax1.set_ylim(0, 1)
        ax1.xaxis.tick_bottom()
        line10, = ax1.plot([], [], lw=2, label='count_spa', color='darkblue')
        line11, = ax1.plot([], [], lw=2, label='count_bil', color='g')
        line12, = ax1.plot([], [], lw=2, label='count_cat', color='y')
        ax1.tick_params('x', labelsize='small')
        ax1.tick_params('y', labelsize='small')
        ax1.legend(loc='best', prop={'size': 8})
        ax1.set_title("lang_groups")
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        ax2.set_xlim(0, steps)
        ax2.set_ylim(0, self.max_people_factor * self.num_people)
        line2, = ax2.plot([], [], lw=2, label = "total_num_agents", color='k')
        ax2.tick_params('x', labelsize='small')
        ax2.tick_params('y', labelsize='small')
        ax2.legend(loc='best', prop={'size': 8})
        ax2.set_title("num_agents")
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        ax3.set_xlim(0, steps)
        ax3.set_ylim(0, 1)
        ax3.tick_params('x', labelsize='small')
        ax3.tick_params('y', labelsize='small')
        line3, = ax3.plot([], [], lw=2, label='biling_evol')
        ax3.legend(loc='best', prop={'size': 8})
        ax3.set_title("biling_quality")
        ax4 = plt.subplot2grid(grid_size, (0, 0), rowspan=3, colspan=3)
        ax4.set_xlim(0, self.grid_width-1)
        ax4.set_ylim(0, self.grid_height-1)
        if plot_type == 'imshow':
            im_2D = ax4.imshow(np.zeros((self.grid_width, self.grid_height)),
                               vmin=0, vmax=2, cmap='viridis',
                               interpolation='nearest', origin='lower')
            fig.colorbar(im_2D)
        elif plot_type == 'scatter':
            dots = ax4.scatter([], [], c=[], vmin=0, vmax=2, cmap='viridis')
            fig.colorbar(dots)
        time_text = ax4.text(0.02, 0.95, '', transform=ax4.transAxes)

        def init_show():
            if plot_type == 'imshow':
                im_2D.set_array(np.random.choice([np.nan, 0], p=[1, 0], size=(self.grid_width, self.grid_height)))
                return im_2D,
            elif plot_type == 'scatter':
                dots.set_offsets([0,0])
                return dots,

        def run_and_update(i):
            #run model step
            self.step()

            #create plots and data for 1D plots
            data = self.data_process.get_model_vars_dataframe()
            line10.set_data(data.index, data['count_spa'])
            line11.set_data(data.index, data['count_bil'])
            line12.set_data(data.index, data['count_cat'])

            line2.set_data(data.index, data['total_num_agents'])

            line3.set_data(data.index, data['biling_evol'])
            # generate data for 2D representation
            self.create_agents_attrs_data('language')
            # create 2D plot
            time_text.set_text('time = %.1f' % i)
            if plot_type == 'imshow':
                im_2D.set_array(self.df_attrs_avg.unstack('x'))
                return line10, line11, line12, line2, line3, im_2D, time_text
            else:
                data = np.hstack((self.df_attrs_avg.reset_index()['x'][:, np.newaxis],
                                  self.df_attrs_avg.reset_index()['y'][:, np.newaxis]))
                dots.set_offsets(data)
                dots.set_array(self.df_attrs_avg.reset_index()['values'])
                return line10, line11, line12, line2, line3, dots, time_text

        # generate persistent animation object
        ani = animation.FuncAnimation(fig, run_and_update, init_func=init_show,
                                      frames=steps, interval=100, blit=True, repeat=False)
        #plt.tight_layout()
        plt.show()

