from mesa.datacollection import DataCollector

import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from collections import Counter
#import library to save any python data type to HDF5
import deepdish as dd


class DataProcessor(DataCollector):
    def __init__(self, model):
        self.model = model
        super().__init__(model_reporters={"count_spa": self.get_lang_stats(0),
                                          "count_bil": self.get_lang_stats(1),
                                          "count_cat": self.get_lang_stats(2),
                                          "total_num_agents": lambda m: m.schedule.get_agent_count(),
                                          "biling_evol": self.get_bilingual_global_evol()
                                         },
                         agent_reporters={"pct_cat_in_biling": lambda a: a.lang_stats['L2']['pct'][a.info['age']],
                                          "pct_spa_in_biling": lambda a: a.lang_stats['L1']['pct'][a.info['age']]})

    def get_lang_stats(self, i):
        """Method to get counts of each type of lang agent

        Arguments:
            * i : integer from [0,1,2] that specifies agent lang type

        Returns:
            * lang type count as percentage of total

        """
        ag_lang_list = [ag.info['language'] for ag in self.model.schedule.agents]
        num_ag = len(ag_lang_list)
        lang_counts = Counter(ag_lang_list)
        return lang_counts[i]/num_ag

    def get_bilingual_global_evol(self):
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

    def get_agents_attrs_value(self, ag_attr, plot=False):
        """ Get value of specific attribute for all lang agents in model """
        ag_and_coords = [(getattr(ag, ag_attr), ag.pos[0], ag.pos[1])
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
        self.model_data = {'initial_conditions':{'num_clusters': self.model.gm.num_clusters,
                                                 'cluster_sizes': self.model.gm.cluster_sizes,
                                                 'cluster_centers': self.model.gm.clust_centers,
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

    def load_model_data(self, data_filename, key='/' ):
        return dd.io.load(data_filename, key)


class DataViz:
    def __init__(self, model):
        self.model = model

    def show_results(self, ag_attr='language', step=None,
                     plot_results=True, plot_type='imshow', save_fig=False):
        grid_size = (3, 5)
        self.model.data_process.get_agents_attrs_value(ag_attr)

        data_2_plot = self.model.data_process.datacollector.get_model_vars_dataframe()[:step]
        data_2D = self.model.data_process.df_attrs_avg.reset_index()

        ax1 = plt.subplot2grid(grid_size, (0, 3), rowspan=1, colspan=2)
        data_2_plot[["count_bil", "count_cat", "count_spa"]].plot(ax=ax1,
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
        data_2_plot['biling_evol'].plot(ax=ax3, title='biling_quality')
        ax3.tick_params('x', labelsize='small')
        ax3.tick_params('y', labelsize='small')
        ax3.legend(loc='best', prop={'size': 8})
        ax4 = plt.subplot2grid(grid_size, (0, 0), rowspan=3, colspan=3)
        if plot_type == 'imshow':
            s = ax4.imshow(self.model.data_process.df_attrs_avg.unstack('x'), vmin=0, vmax=2, cmap='viridis',
                           interpolation='nearest', origin='lower')
        else:
            s = ax4.scatter(data_2D['x'],
                            data_2D['y'],
                            c=data_2D['values'],
                            marker='s',
                            vmin=0, vmax=2, s=35,
                            cmap='viridis')
        ax4.text(0.02, 1.04, 'time = %.1f' % self.model.schedule.steps, transform=ax4.transAxes)
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        plt.colorbar(s)
        plt.suptitle(self.model.data_process.save_dir)
        #plt.tight_layout()
        if save_fig:
            if self.model.data_process.save_dir:
                plt.savefig(self.model.data_process.save_dir + '/step' + str(step) + '.png')
                plt.close()
            else:
                plt.savefig('step' + str(step) + '.png')
                plt.close()

        if plot_results:
            plt.show()


