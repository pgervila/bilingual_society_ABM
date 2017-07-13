import pytest
import numpy as np
import os, sys
from imp import reload
#sys.path.append("/Users/PG/Paolo/python_repos/language_proj/lang_model_simple/")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import agent_simple
import model_simple
reload(agent_simple)
reload(model_simple)
from agent_simple import Simple_Language_Agent
from model_simple import Simple_Language_Model

@pytest.fixture(scope="module")
def model():
    return Simple_Language_Model(20, num_cities=1)


test_data_group_conv = [
            ([0, 0], [0.5, 0.5], [0.0, 0.0], False, 0),
            ([1, 1], [0.5, 0.4], [0.4, 0.5], False, 0),
            ([0, 0, 1], [0.5, 0.5, 0.5], [0.02, 0.04, 0.4], False, 0),
            ([1, 1, 2], [0.5, 0.5, 0.04], [0.4, 0.4, 0.5], False, 1),
            ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4], False, 0),
            ([1, 1], [0.4, 0.4], [0.5, 0.5], True, 1),
            ([0, 2], [0.5, 0.04], [0.04, 0.5], True, [0, 1]),
            ([0, 2], [0.5, 0.04], [0.02, 0.5], True, [(0, 0), (1, 'mute')]),
            ([1, 1, 1], [0.5, 0.5, 0.4], [0.4, 0.5, 0.5], True, 0),
            ([1, 1, 1], [0.4, 0.5, 0.5], [0.5, 0.5, 0.4], True, 1),
            ([0, 2, 2], [0.5, 0.02, 0.02], [0.02, 0.5, 0.5], True, [(0, 0), (1, 'mute and excluded'), (2, 'mute and excluded')]),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.5, 0.5], True, [0, 0, 1]),
            ([0, 1, 2], [0.5, 0.5, 0.02], [0.02, 0.4, 0.5], True, [(0, 0), (1, 0), (2, 'mute and excluded')]),
            ([0, 1, 2], [0.5, 0.5, 0.04], [0.04, 0.4, 0.5], True, [0, 0, 1]),
            ([2, 1, 0], [0.02, 0.5, 0.5], [0.5, 0.4, 0.04], True, [(0, 1), (1, 1), (2, 'mute')])
]
@pytest.mark.parametrize("langs, pcts_1, pcts_2, delete_edges, expected", test_data_group_conv)
def test_group_conv_lang(model, langs, pcts_1, pcts_2, delete_edges, expected): 
    agents = model.schedule.agents[:len(langs)]
    for idx, agent in enumerate(agents):
        agent.language = langs[idx]
        agent.lang_stats['L1']['pct'][agent.age] = pcts_1[idx]
        agent.lang_stats['L2']['pct'][agent.age] = pcts_2[idx]
    if delete_edges:
        model.known_people_network.remove_edges_from(model.known_people_network.edges())
    lang_conv = agents[0].get_conv_lang(agents[0], agents[1:], ret_results=True)

    assert np.all(expected == lang_conv)
        
        
#@pytest.mark.parametrize("num_agents, num_cities", [()]    
