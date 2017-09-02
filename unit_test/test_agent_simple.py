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


def test_set_lang_ics():
    pass

def test_speak():
    pass

def test_vocab_choice_model():
    pass

def test_update_lang_arrays():
    pass



  


                