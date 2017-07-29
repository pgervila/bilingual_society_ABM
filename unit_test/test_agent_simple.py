import pytest
from unittest.mock import patch
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


@pytest.mark.parametrize("langs, pcts_1, pcts_2, delete_edges, expected", test_data_group_conv)
def test_group_conv_lang(model, langs, pcts_1, pcts_2, delete_edges, expected): 
    agents = model.schedule.agents[:len(langs)]
    for idx, agent in enumerate(agents):
        agent.language = langs[idx]
        agent.lang_stats['L1']['pct'][agent.age] = pcts_1[idx]
        agent.lang_stats['L2']['pct'][agent.age] = pcts_2[idx]
    if delete_edges:
        model.known_people_network.remove_edges_from(model.known_people_network.edges())
    with patch('agent_simple.Simple_Language_Agent.vocab_choice_model') as mock_method:
        lang_conv, mute_type = agents[0].get_conv_params(agents[0], agents[1:], ret_results=True)

        assert np.all(expected == (lang_conv, mute_type))
        if not isinstance(lang_conv, list):
            if all([x in langs for x in [0,2]]):
                for idx, _ in enumerate([ag for ag in agents if ag.language != mute_type]):
                    mock_method.assert_any_call(lang_conv, 
                                                agents[:idx] + agents[idx + 1:], 
                                                long=False)
            else:
                for idx, _ in enumerate(agents):
                    mock_method.assert_any_call(lang_conv, 
                                                agents[:idx] + agents[idx + 1:], 
                                                long=True)
                    
        else:
            for idx, _ in enumerate(agents):
                mock_method.assert_any_call(lang_conv[idx], 
                                            agents[:idx] + agents[idx + 1:], 
                                            long=False)