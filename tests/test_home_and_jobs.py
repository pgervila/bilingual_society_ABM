import random
import pytest
from bilangsim.agent import Adult
from bilangsim.model import BiLangModel


@pytest.fixture(scope="module")
def model():
    return BiLangModel(400, num_clusters=3)


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