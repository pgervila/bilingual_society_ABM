import random
import pytest
from unittest.mock import patch
import numpy as np
import os, sys
from imp import reload
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import agent
import model
reload(agent)
reload(model)
from agent import Child, Adolescent, YoungUniv, Young, Adult, Worker, Teacher
from model import LanguageModel

@pytest.fixture(scope="module")
def model():
    return LanguageModel(400, num_clusters=3) 


def test_home_assign_to_agent(model):
    """ Method to test home assignment to a random agent """
    i = random.choice(range(model.num_clusters))
    for ag in model.geo.clusters_info[i]['agents']:
        if isinstance(ag, Adult):
            agent = ag
            break
    h_1 = agent.loc_info['home']
    i = random.choice(range(model.num_clusters))
    for h in model.geo.clusters_info[i]['homes']:
        if not h.info['occupants']:
            h_2 = h
            break
    h_2.assign_to_agent(agent)
    
    assert agent not in h_1.info['occupants']
    assert agent not in h_1.agents_in
    assert agent in h_2.info['occupants']


def test_remove_agent(model):
    i = random.choice(range(model.num_clusters))
    agent = model.geo.clusters_info[i]['agents'][0]
    h = agent.loc_info['home']
    h.remove_agent(agent)
    assert agent not in h.info['occupants']
    assert not agent.loc_info['home']
    

    