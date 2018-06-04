from mesa.datacollection import DataCollector

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from collections import Counter
import deepdish as dd


class DataProcessor(DataCollector):
    def __init__(self, model):
        self.model = model
        super().__init__(model_reporters={"pct_spa": lambda dp: dp.get_lang_stats(0),
                                          "pct_bil": lambda dp: dp.get_lang_stats(1),
                                          "pct_cat": lambda dp: dp.get_lang_stats(2),
                                          "total_num_agents": lambda dp: len(dp.model.schedule.agents),
                                          "pct_cat_in_biling": lambda dp: dp.get_global_bilang_inner_evol()
                                         },
                         agent_reporters={"pct_cat_knowledge": lambda a: a.lang_stats['L2']['pct'][a.info['age']],
                                          "pct_L21_knowledge": lambda a: a.lang_stats['L21']['R'].mean(),
                                          "pct_spa_knowledge": lambda a: a.lang_stats['L1']['pct'][a.info['age']],
                                          "pct_L12_knowledge": lambda a: a.lang_stats['L12']['R'].mean(),
                                          "tokens_per_step_spa": lambda a: (a.wc_final['L1'] - a.wc_init['L1']).sum(),
                                          "tokens_per_step_cat": lambda a: (a.wc_final['L2'] - a.wc_init['L2']).sum(),
                                          "x": lambda a: a.pos[0],
                                          "y": lambda a: a.pos[1],
                                          "age": lambda a: a.info['age'],
                                          "language": lambda a: a.info['language'],
                                          "excl_c": lambda a: a.lang_stats['L1']['excl_c'][a.info['age']] if a.info['language'] == 2 else a.lang_stats['L2']['excl_c'][a.info['age']],
                                          "clust_id": lambda a: a.loc_info['home'].clust
                                          }
                         )

        self.save_dir = ''

    def collect(self):
        """ Collect all the data for the given model object. """
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(self))

        if self.agent_reporters:
            for var, reporter in self.agent_reporters.items():
                agent_records = []
                for agent in self.model.schedule.agents:
                    agent_records.append((agent.unique_id, reporter(agent)))
                self.agent_vars[var].append(agent_records)

    def get_lang_stats(self, lang_type):
        """Method to get counts of each type of lang agent

        Arguments:
            * lang_type : integer from [0,1,2] that specifies agent linguistic type

        Returns:
            * lang type count as percentage of total

        """
        ag_lang_list = [ag.info['language'] for ag in self.model.schedule.agents]
        lang_counts = Counter(ag_lang_list)
        return lang_counts[lang_type] / len(ag_lang_list)

    def get_global_bilang_inner_evol(self):
        """Method to compute internal linguistic structure of all bilinguals,
        expressed as average amount of Catalan heard or spoken as % of total

         Returns:
             * float representing the AVERAGE percentage of Catalan in bilinguals

        """
        list_biling = [ag.lang_stats['L2']['pct'][ag.info['age']] for ag in self.model.schedule.agents
                       if ag.info['language'] == 1]
        if list_biling:
            return np.array(list_biling).mean()
        else:
            if self.get_lang_stats(2) > self.get_lang_stats(0):
                return 1
            else:
                return 0

    def get_tokens_per_day(self):
        pass

    def get_agents_attrs_value(self, ag_attr, plot=False):
        """ Get value of specific attribute for all lang agents in model """
        ag_and_coords = [(ag.info[ag_attr], ag.pos[0], ag.pos[1])
                         for ag in self.model.schedule.agents]
        ag_and_coords = np.array(ag_and_coords)
        df_attrs = pd.DataFrame({'values': ag_and_coords[:, 0],
                                 'x': ag_and_coords[:, 1],
                                 'y': ag_and_coords[:, 2]})
        self.df_attrs_avg = df_attrs.groupby(['x', 'y']).mean()

        if plot:
            s = plt.scatter(self.df_attrs_avg.reset_index()['x'],
                            self.df_attrs_avg.reset_index()['y'],
                            c=self.df_attrs_avg.reset_index()['values'],
                            marker='s',
                            vmin=0, vmax=2, s=30,
                            cmap='viridis')
            plt.colorbar(s)
            plt.show()

    def save_model_data(self):
        self.model_data = {'initial_conditions': {'num_clusters': self.model.geo.num_clusters,
                                                  'cluster_sizes': self.model.geo.cluster_sizes,
                                                  'cluster_centers': self.model.geo.clust_centers,
                                                  'init_num_people': self.model.num_people,
                                                  'grid_width': self.model.grid.width,
                                                  'grid_height': self.model.grid.height,
                                                  'init_lang_distrib': self.model.init_lang_distrib,
                                                  'sort_by_dist': self.model.lang_ags_sorted_by_dist,
                                                  'sort_within_clust': self.model.lang_ags_sorted_in_clust},
                           'model_results': self.get_model_vars_dataframe(),
                           'agent_results': self.get_agent_vars_dataframe()}
        if self.save_dir:
            dd.io.save(self.save_dir + '/model_data.h5', self.model_data)
        else:
            dd.io.save('model_data.h5', self.model_data)

    @staticmethod
    def load_model_data(data_filename, key='/'):
        return dd.io.load(data_filename, key)


class DataViz:
    def __init__(self, model):
        self.model = model

    def show_results(self, ag_attr='language', step=None,
                     plot_results=True, plot_type='scatter', save_fig=False):
        grid_size = (3, 5)
        self.model.data_process.get_agents_attrs_value(ag_attr)

        data_2_plot = self.model.data_process.get_model_vars_dataframe()[:step]
        data_2D = self.model.data_process.df_attrs_avg.reset_index()

        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        data_2_plot[["pct_bil", "pct_cat", "pct_spa"]].plot(ax=ax1,
                                                            title='lang_groups',
                                                            color=['darkgreen','y','darkblue'])
        ax1.xaxis.tick_bottom()
        ax1.tick_params('x', labelsize='small')
        ax1.tick_params('y', labelsize='small')
        ax1.legend(loc='best', prop={'size': 8})
        ax2 = plt.subplot2grid(grid_size, (1, 3), rowspan=1, colspan=2)
        data_2_plot['total_num_agents'].plot(ax=ax2, title='num_agents')
        ax2.tick_params('x', labelsize='small')
        ax2.tick_params('y', labelsize='small')
        ax3 = plt.subplot2grid(grid_size, (2, 3), rowspan=1, colspan=2)
        data_2_plot['pct_cat_in_biling'].plot(ax=ax3, title='biling_quality')
        ax3.tick_params('x', labelsize='small')
        ax3.tick_params('y', labelsize='small')
        ax3.legend(loc='best', prop={'size': 8})
        ax4 = plt.subplot2grid(grid_size, (0, 0), rowspan=3, colspan=3)
        if plot_type == 'imshow':
            data = self.model.data_process.df_attrs_avg.unstack('x')
            s = ax4.imshow(data, vmin=0, vmax=2, cmap='viridis',
                           interpolation='nearest', origin='lower')
            ax4.set_xlim(0, data.shape[0])
            ax4.set_ylim(0, data.shape[1])

        else:
            s = ax4.scatter(data_2D['x'], data_2D['y'],
                            c=data_2D['values'], marker='s',
                            vmin=0, vmax=2, s=25, cmap='viridis')
            ax4.set_xlim(0, 100)
            ax4.set_ylim(0, 100)
        ax4.text(0.02, 1.04, 'time = %.1f' % self.model.schedule.steps, transform=ax4.transAxes)

        plt.colorbar(s)
        plt.suptitle(self.model.data_process.save_dir)
        plt.tight_layout()
        if save_fig:
            if self.model.data_process.save_dir:
                plt.savefig(self.model.data_process.save_dir + '/step' + str(step) + '.png')
                plt.close()
            else:
                plt.savefig('step' + str(step) + '.png')
                plt.close()
        if plot_results:
            plt.show()


class VizImpData:

    def __init__(self, file_name=None, key='/'):
        self.data = dd.io.load(file_name, key)

    def show_imported_results(self, key='/'):
        self.model


class PostProcessor:
    def __init__(self, data_filename):
        self.results = self.load_model_data(data_filename)

    @staticmethod
    def load_model_data(data_filename, key='/'):
        return dd.io.load(data_filename, key)

    def ag_results_by_id(self, ag_id):

        idx = pd.IndexSlice
        # return self.results['agent_results'].unstack().loc[idx[:], idx[['tokens_per_day_cat', 'tokens_per_day_spa',
        #                                                                 'pct_cat_knowledge', 'pct_spa_knowledge',
        #                                                                 'pct_L21_knowledge', 'pct_L12_knowledge', ], ag_id]]
        return self.results['agent_results'].unstack().loc[idx[:], idx[:, ag_id]]




