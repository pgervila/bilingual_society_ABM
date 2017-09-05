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
    return Simple_Language_Model(20, num_clusters=1)


test_data_update_lang_arrays = [(0, [np.array([0, 1, 10, 450, 470, 490]), np.array([3, 2, 1, 2, 1, 3])], False), 
                                (0, [np.array([0, 1, 5, 10]), np.array([3, 2, 1, 1])], True)]

test_data_vocab_choice_model = [True, False]

def test_set_lang_ics():
    pass

def test_speak():
    pass

def test_vocab_choice_model():
    pass

@pytest.mark.parametrize("lang, sample_words, speak", test_data_update_lang_arrays)
def test_update_lang_arrays(model, lang, sample_words, speak):
    agent = model.schedule.agents[0]
    if not speak:
        agent.lang_stats['L1']['R'][:100] = 1.
        agent.lang_stats['L1']['R'][-100:] = 0.
        agent.lang_stats['L1']['wc'][-100:] = 10
    R_init = agent.lang_stats['L1']['R'][sample_words[0]].copy()
    S_init = agent.lang_stats['L1']['S'][sample_words[0]].copy()
    wc_init = agent.lang_stats['L1']['wc'][sample_words[0]].copy()
    # call method to test
    agent.update_lang_arrays(lang, sample_words, speak)
    if not speak:
        assert np.all(agent.lang_stats['L1']['S'][[0, 1, 10]] > S_init[[0, 1, 2]])
        assert agent.lang_stats['L1']['S'][490] == S_init[5]
        assert np.all(agent.lang_stats['L1']['wc'][[0, 1, 10, 490]] > wc_init[[0, 1, 2, 5]])
    else:
        assert np.all(agent.lang_stats['L1']['S'][sample_words[0]] > S_init)

@pytest.mark.parametrize("long", test_data_vocab_choice_model)
def test_vocab_choice_model(model, long):
    agent = model.schedule.agents[0]
    lang = 0 if agent.language == 0 else 1
    act, act_c = agent.vocab_choice_model(lang, long)
    assert len(act) > 0
        