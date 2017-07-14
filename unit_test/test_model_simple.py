import pytest
import os, sys
from imp import reload
import numpy as np
from scipy.spatial.distance import pdist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import agent_simple
import model_simple
reload(agent_simple)
reload(model_simple)
from agent_simple import Simple_Language_Agent
from model_simple import Simple_Language_Model

@pytest.fixture(scope="module")
def model():
    return Simple_Language_Model(50, num_cities=2)

@pytest.fixture(scope="module", params=[(50, 2), (197, 5), (1000, 15)])
def model_param(request):
    return Simple_Language_Model(request.param[0], num_cities=request.param[1])


def test_compute_cluster_centers(model_param):
    assert model_param.num_cities == len(model_param.clust_centers)
    assert np.all(pdist(model_param.clust_centers) >= 0.16)
    
def test_compute_cluster_sizes(model_param):
    assert np.sum(model_param.cluster_sizes) == model_param.num_people

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
        
        