import pytest
from unittest.mock import patch
import os, sys
from imp import reload
import numpy as np
from scipy.spatial.distance import pdist

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import agent
import model
reload(agent)
reload(model)
from agent import Adolescent, Young, Adult, Teacher
from model import LanguageModel
from city_objects import Job, School


@pytest.fixture(scope="module")
def model():
    return LanguageModel(100, num_clusters=1)


@pytest.fixture(scope="module", params=[(457, 3), (521, 2), (897, 5), (1501, 6)])
def model_param(request):
    return LanguageModel(request.param[0], num_clusters=request.param[1])

# define cases to test => Inputs and optionally expected results


test_data_comp_cl_centers = [(10, 0.2), (20, 0.1), (26, 0.2)]

test_data_get_conv_params = [
            ([0, 0], [0.5, 0.5], [0.0, 0.0], False, ((0, 0), None)),
            ([1, 1], [0.5, 0.4], [0.4, 0.5], False, ((0, 0), None)),
            ([0, 0, 1], [0.5, 0.5, 0.5], [0.02, 0.04, 0.4], False, ((0, 0, 0), None)),
            ([1, 1, 2], [0.5, 0.5, 0.04], [0.4, 0.4, 0.5], False, ((1, 1, 1), None)),
            ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4], False, ((0, 0, 0), None)),
            ([1, 1], [0.4, 0.4], [0.5, 0.5], True, ((1, 1), None)),
            ([0, 2], [0.5, 0.04], [0.04, 0.5], True, ((0, 1), None)),
            ([0, 2], [0.5, 0.04], [0.02, 0.5], True, ((0, 0), 2)),
            ([1, 1, 1], [0.5, 0.5, 0.4], [0.4, 0.5, 0.5], True, ((0, 0, 0), None)),
            ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4], True, ((1, 1, 1), None)),
            ([0, 2, 2], [0.5, 0.02, 0.02], [0.02, 0.5, 0.5], True, ((0, 0, 0), 2)),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.5, 0.5], True, ((0, 0, 1), None)),
            ([0, 1, 2], [0.5, 0.5, 0.02], [0.02, 0.4, 0.5], True, ((0, 0, 0), 2)),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.4, 0.5], True, ((0, 0, 1), None)),
            ([2, 1, 0], [0.02, 0.5, 0.5], [0.5, 0.4, 0.04], True, ((1, 1, 1), 0)),
            ([2, 1, 0, 0], [0.04, 0.4, 0.5, 0.5], [0.5, 0.5, 0.02, 0.02], True, ((1, 1, 1, 1), 0)),
            ([0, 1, 2, 2], [0.5, 0.5, 0.04, 0.02], [0.04, 0.4, 0.5, 0.5], True, ((0, 0, 0, 0), 2))
]

test_data_run_conversation = [([1, 1, 1, 2, 2], True)]

test_data_remove_after_death = [Teacher, Adult, Adolescent]

test_data_remove_from_locations = [False, True]


def test_model_consistency(model_param):
    """ Check that all agents live in the same
        cluster as that of their occupation """
    for ag in model_param.schedule.agents:
        if ag.info['age'] > 36:
            if isinstance(ag, Young):
                if ag.loc_info['job']:
                    if isinstance(ag, Teacher):
                        assert ag['clust'] == ag.loc_info['job'][0].info['clust']
                    else:
                        assert ag['clust'] == ag.loc_info['job'].info['clust']
            else:
                if ag.loc_info['school']:
                    assert ag['clust'] == ag.loc_info['school'][0].info['clust']

            # assert blocked property is non existent
            assert not hasattr(ag, 'blocked')



@pytest.mark.parametrize("langs, pcts_1, pcts_2, delete_edges, expected", 
                         test_data_get_conv_params)
def test_get_conv_params(model, langs, pcts_1, pcts_2, delete_edges, expected): 
    agents = model.schedule.agents[:len(langs)]
    for idx, agent in enumerate(agents):
        agent.info['language'] = langs[idx]
        agent.lang_stats['L1']['pct'][agent.info['age']] = pcts_1[idx]
        agent.lang_stats['L2']['pct'][agent.info['age']] = pcts_2[idx]
    if delete_edges:
        model.nws.known_people_network.remove_edges_from(model.nws.known_people_network.edges())

    params = model.get_conv_params(agents)
    assert np.all(expected == (params['lang_group'], params['mute_type']))


@pytest.mark.parametrize("agent_type", test_data_remove_after_death)
def test_remove_after_death(model, agent_type):
    #corr_dict = {'Adult': 'Young', 'YoungUniv': 'Adolescent', 'Pensioner': 'Adult'}
    for ag in model.schedule.agents:
        if isinstance(ag, agent_type):
            break
    model.remove_after_death(ag)


@pytest.mark.parametrize("replace", test_data_remove_from_locations)
def test_remove_from_locations(model, replace):
    corr_dict = {'Baby': 'Child', 'Child': 'Adolescent', 'Adolescent': 'Young',
                 'YoungUniv': 'Adult', 'Adult': 'Pensioner'}
    # for ag in model.schedule.agents:
    #     if isinstance(ag, corr_dict[agent_type]):
    #         pass # TODO