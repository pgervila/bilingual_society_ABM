import pytest
from unittest.mock import patch
import os, sys
from imp import reload
import numpy as np
from scipy.spatial.distance import pdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import agent_simple
import model_simple
reload(agent_simple)
reload(model_simple)
from agent_simple import Language_Agent
from model_simple import Language_Model

@pytest.fixture(scope="module")
def model():
    return Language_Model(50, num_clusters=2)

@pytest.fixture(scope="module", params=[(50, 3), (50, 2), (197, 5), (1000, 15)])
def model_param(request):
    return Language_Model(request.param[0], num_clusters=request.param[1])

# define cases to test => Inputs and optionally expected results

test_data_comp_cl_centers = [(10, 0.2), (20, 0.1), (26, 0.2)]

test_data_get_conv_params = [
            ([0, 0], [0.5, 0.5], [0.0, 0.0], False, (0, None)),
            ([1, 1], [0.5, 0.4], [0.4, 0.5], False, (0, None)),
            ([0, 0, 1], [0.5, 0.5, 0.5], [0.02, 0.04, 0.4], False, (0, None)),
            ([1, 1, 2], [0.5, 0.5, 0.04], [0.4, 0.4, 0.5], False, (1, None)),
            ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4], False, (0, None)),
            ([1, 1], [0.4, 0.4], [0.5, 0.5], True, (1, None)),
            ([0, 2], [0.5, 0.04], [0.04, 0.5], True, ([0, 1], None)),
            ([0, 2], [0.5, 0.04], [0.02, 0.5], True, (0, 2)),
            ([1, 1, 1], [0.5, 0.5, 0.4], [0.4, 0.5, 0.5], True, (0, None)),
            ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4], True, (1, None)),
            ([0, 2, 2], [0.5, 0.02, 0.02], [0.02, 0.5, 0.5], True, (0, 2)),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.5, 0.5], True, ([0, 0, 1], None)),
            ([0, 1, 2], [0.5, 0.5, 0.02], [0.02, 0.4, 0.5], True, (0, 2)),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.4, 0.5], True, ([0, 0, 1], None)),
            ([2, 1, 0], [0.02, 0.5, 0.5], [0.5, 0.4, 0.04], True, (1, 0)),
            ([2, 1, 0, 0], [0.04, 0.4, 0.5, 0.5], [0.5, 0.5, 0.02, 0.02], True, (1, 0)),
            ([0, 1, 2, 2], [0.5, 0.5, 0.04, 0.02], [0.04, 0.4, 0.5, 0.5], True, (0, 2))
]

test_data_run_conversation = [([1, 1, 1, 2, 2], True)]


# @pytest.mark.parametrize("num_clusters, min_dist", test_data_comp_cl_centers)
# def test_compute_cluster_centers(min_dist, num_clusters):
#     max_pts_per_side = (int(0.8 / min_dist) + 1)
#     with patch('model_simple.Simple_Language_Model') as mock_class:
#         if num_clusters >= max_pts_per_side**2:
#             with pytest.raises(ValueError):
#                 mock_class.num_clusters = num_clusters
#                 Simple_Language_Model.compute_cluster_centers(mock_class, min_dist=min_dist)
#         else:
#             mock_class.num_clusters = num_clusters
#             Simple_Language_Model.compute_cluster_centers(mock_class, min_dist=min_dist)
#             assert mock_class.num_clusters == len(mock_class.clust_centers)
#             # check min distance
#             assert np.min(pdist(mock_class.clust_centers)) > min_dist * 0.8
        

# def test_compute_cluster_sizes(model_param):
#     with pytest.raises(Exception) as e_info:
#         for size in model_param.cluster_sizes:
#             assert size >= 20
#         assert np.sum(model_param.cluster_sizes) == model_param.num_people

# def test_generate_cluster_points_coords():
#     pass

# def test_set_clusters_info:
#     pass

# def test_sort_coords_in_clust():
#     pass

# def test_map_jobs():
#     pass

# def test_map_schools():
#     pass

# def test_map_homes():
#     pass
    
# def test_generate_lang_distrib():
#     pass


# def test_job_school_home_assignment(model):
#     agents = model.schedule.agents
#     working_agents = [ag for ag in agents if ag.age >= 36 * 20]
#     school_agents = [ag for ag in agents if ag.age < 36 * 20]
#     for agent in working_agents:
#         assert agent.loc_info['job'] != None
#     for agent in school_agents:
#         assert agent.loc_info['school'] != None
#     for agent in agents:
#         assert agent.loc_info['home'] != None
        
# def test_agent_activation(model_param):
#     assert model_param.num_people == 197
#     assert model_param.num_people == len(model_param.schedule.agents)
        
# def test_define_family_networks(model):
#     pass

# def test_define_friendship_networks(model):
#     agents = model.schedule.agents
#     for ag in agents:
#         assert ag not in model.friendship_network[ag]
#         # check relatives are not friends
#         for relat in model.family_network[ag]:
#             assert relat not in model.friendship_network[ag]


@pytest.mark.parametrize("langs, pcts_1, pcts_2, delete_edges, expected", test_data_get_conv_params)
def test_get_conv_params(model, langs, pcts_1, pcts_2, delete_edges, expected): 
    agents = model.schedule.agents[:len(langs)]
    for idx, agent in enumerate(agents):
        agent.language = langs[idx]
        agent.lang_stats['L1']['pct'][agent.age] = pcts_1[idx]
        agent.lang_stats['L2']['pct'][agent.age] = pcts_2[idx]
    if delete_edges:
        model.known_people_network.remove_edges_from(model.known_people_network.edges())
    params = model.get_conv_params(agents)
    
    assert np.all(expected == (params['lang_group'], params['mute_type']))

# @pytest.mark.parametrize("langs, delete_edges", test_data_run_conversation)
# def test_run_conversation(model, langs, delete_edges):
#     agents = model.schedule.agents[:len(langs)]
#     for idx, agent in enumerate(agents):
#         agent.language = langs[idx]
#     if delete_edges:
#         model.known_people_network.remove_edges_from(model.known_people_network.edges())
#     with patch('model_simple.Simple_Language_Model.get_conv_params') as mock_method1:
#         mock_method1.return_value = (agents, {'lang_group': 1, 
#                                              'mute_type': None, 
#                                              'long': True,
#                                              'bilingual': False}
#                                      )
#         with patch('agent_simple.Simple_Language_Agent.vocab_choice_model') as mock_method2:
#             mock_method2.return_value = (np.array([0, 1, 34, 435]), np.array([3, 2, 1, 1]))
#             with patch('agent_simple.Simple_Language_Agent.update_lang_arrays') as mock_call:
#                 model.run_conversation(agents[0], agents[1:])
#                 for ix, lang in enumerate(mock_method1.return_value[1]['lang_group']):
#                     mock_call.assert_any_call(lang, speak=False)