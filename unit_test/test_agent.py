import pytest
from unittest.mock import patch
import numpy as np
import os, sys
from importlib import reload
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import agent
import model
reload(agent)
reload(model)
from agent import Baby, Child, Adolescent, Young, YoungUniv, Adult, Worker, Teacher, TeacherUniv, Pensioner
from model import LanguageModel

@pytest.fixture(scope="module")
def model():
    return LanguageModel(250, num_clusters=2)

@pytest.fixture(scope="module")
def city_places(model):
    h1 = model.geo.clusters_info[0]['homes'][0]
    sc1 = model.geo.clusters_info[0]['schools'][0]
    un1 = model.geo.clusters_info[np.argmax([model.geo.cluster_sizes])]['university']
    j1 = model.geo.clusters_info[0]['jobs'][0]
    
    places = {'home':h1, 'school':sc1, 'university':un1, 'job':j1}
    
    return places


@pytest.fixture(scope="module")
def jobs_in_diff_clusts(model):
    h1 = model.geo.clusters_info[0]['homes'][1]
    h2 = model.geo.clusters_info[1]['homes'][1]
    j1 = model.geo.clusters_info[0]['jobs'][0]
    j2 = model.geo.clusters_info[1]['jobs'][0]
    

test_data_evolve = [(Baby, Child, ('school', 'school', False)),
                    (Child, Adolescent, ('school', 'school', False)),
                    (Adolescent, Young, ('school', 'job', False)), 
                    (Adolescent, YoungUniv, ('school', 'university', True)),
                    (Young, Adult, ('job', 'job', False)),
                    (YoungUniv, Young, ('university', 'job', False)),
                    (Adult, Pensioner, ('job', None, False)),
                    (Adult, Teacher, ('job', 'job', False)),
                    (Adult, TeacherUniv, ('job', 'job', False)),
                    (Teacher, Pensioner, ('job', None, False))]


test_data_update_lang_arrays = [({0: (np.array([0, 1, 10, 450, 470, 490]), 
                                      np.array([3, 2, 1, 2, 1, 3]))}, False), 
                                ({0: (np.array([0, 1, 5, 10]), 
                                      np.array([3, 2, 1, 1]))}, True)]

test_data_vocab_choice_model = [True, False]

test_data_move_new_home = [(True, None, None), (True, True, None), (True, True, True)]


@pytest.mark.parametrize("origin_class, new_class, labels", test_data_evolve)
def test_evolve(model, city_places, origin_class, new_class, labels):
    if origin_class == Baby:
        father, mother = None, None
        for ag in model.schedule.agents:
            if isinstance(ag, Young) and ag.info['sex'] == 'M':
                father = ag
            elif isinstance(ag, Young) and ag.info['sex'] == 'F':
                mother = ag
            if father and mother:
                break
        ag_id = model.set_available_ids.pop()
        old_ag = origin_class(father, mother, 0, 0, model, ag_id, 0, 'M', age=40,
                              home=city_places['home'],
                              **{labels[0]:city_places[labels[0]]})
    elif origin_class == Teacher:
        ag_id = model.set_available_ids.pop()
        old_ag = origin_class(model, ag_id , 0, 'M', age=1200, home=city_places['home'])
        school = city_places['school']
        ckey = np.random.choice(list(school.grouped_studs.keys()))
        school.assign_teacher(old_ag, ckey)
    else:
        ag_id = model.set_available_ids.pop()
        old_ag = origin_class(model, ag_id, 0, 'M', age=100, home=city_places['home'],
                              **{labels[0]:city_places[labels[0]]})
    
    city_places['home'].agents_in.add(old_ag)
    print(old_ag)
    print(city_places['home'].agents_in)
    model.geo.add_agents_to_grid_and_schedule(old_ag)
    model.geo.clusters_info[city_places['home'].clust]['agents'].append(old_ag)
    if labels[2]:
        grown_ag = old_ag.evolve(new_class, ret_output=True, 
                                 **{labels[1]:city_places[labels[1]]})
    else:
        grown_ag = old_ag.evolve(new_class, ret_output=True)
    # check references to old agent are all deleted
    assert sys.getrefcount(old_ag) == 2
    if labels[0] and labels[0] != labels[1]:
        assert labels[0] not in grown_ag.loc_info
    if labels[1]:
        assert labels[1] in grown_ag.loc_info
    print(city_places['home'].agents_in)
    print('*****')
        
def test_look_for_partner(model):
    pass
            
def test_listen(model):
    pass


@pytest.mark.parametrize("job1, job2, j2_teach", test_data_move_new_home)
def test_move_to_new_home(model, job1, job2, j2_teach):
    
    h1 = model.geo.clusters_info[0]['homes'][1]
    j1 = model.geo.clusters_info[0]['jobs'][0] if job1 else None
    h2 = model.geo.clusters_info[1]['homes'][1]
    j2 = model.geo.clusters_info[1]['jobs'][0] if job2 else None

    consort1 = Young(model, 1500, 0, 'M', age=1200, home=h1, job=j1)
    if j2_teach:
        # TODO: TO BE FIXED
        sc1 = model.geo.clusters_info[0]['schools'][0]
        consort2 = Teacher(model, 1500, 0, 'F', age=1200, home=h2, job=j2)
    else:
        consort2 = Young(model, 1500, 0, 'F', age=1200, home=h2, job=j2)
    
    ags = [consort1, consort2]
    model.geo.add_agents_to_grid_and_schedule(ags)
    model.geo.clusters_info[0]['agents'].append(consort1)
    model.geo.clusters_info[1]['agents'].append(consort2)
    model.nws.add_ags_to_networks(ags)
    consort1.update_acquaintances(consort2, 0)
    
    # get married
    consort1.info['married'] = True
    consort2.info['married'] = True
    fam_nw = model.nws.family_network
    lang = model.nws.known_people_network[consort1][consort2]['lang']
    # family network is directed Graph !!
    fam_nw.add_edge(consort1, consort2, lang=lang, fam_link='consort')
    fam_nw.add_edge(consort2, consort1, lang=lang, fam_link='consort')
    
    # find appartment to move in together
    consort1.move_to_new_home(consort2)
    
    if consort1.loc_info['job'] == j1:
        assert consort1  in consort1.loc_info['job'].info['employees']
    if consort2.loc_info['job'] == j2 and j2:
        assert consort2  in consort2.loc_info['job'].info['employees']
    

                  
# @pytest.mark.parametrize("sample_words, speak", test_data_update_lang_arrays)
# def test_update_lang_arrays(model, sample_words, speak):
#     agent = model.schedule.agents[0]
#     if not speak:
#         agent.lang_stats['L1']['R'][:100] = 1.
#         agent.lang_stats['L1']['R'][-100:] = 0.
#         agent.lang_stats['L1']['wc'][-100:] = 10
#     R_init = agent.lang_stats['L1']['R'][sample_words[0]].copy()
#     S_init = agent.lang_stats['L1']['S'][sample_words[0]].copy()
#     wc_init = agent.lang_stats['L1']['wc'][sample_words[0]].copy()
#     # call method to test
#     agent.update_lang_arrays(lang, sample_words, speak)
#     if not speak:
#         assert np.all(agent.lang_stats['L1']['S'][[0, 1, 10]] > S_init[[0, 1, 2]])
#         assert agent.lang_stats['L1']['S'][490] == S_init[5]
#         assert np.all(agent.lang_stats['L1']['wc'][[0, 1, 10, 490]] > wc_init[[0, 1, 2, 5]])
#     else:
#         assert np.all(agent.lang_stats['L1']['S'][sample_words[0]] > S_init)

# @pytest.mark.parametrize("long", test_data_vocab_choice_model)
# def test_vocab_choice_model(model, long):
#     agent = model.schedule.agents[0]
#     lang = 0 if agent.language == 0 else 1
#     act, act_c = agent.vocab_choice_model(lang, long)
#     assert len(act) > 0
        