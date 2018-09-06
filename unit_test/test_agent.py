import pytest
from unittest.mock import patch
import numpy as np
import os, sys
import objgraph
from importlib import reload
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import agent
import model
reload(agent)
reload(model)
from agent import Baby, Child, Adolescent, Young, YoungUniv, Adult, Teacher, TeacherUniv, Pensioner
from model import LanguageModel

np_seed = np.random.randint(10000)
#np_seed = 1454
np.random.seed(np_seed)
print('test seed is {}'.format(np_seed))


@pytest.fixture(scope="module")
def model():
    return LanguageModel(251, num_clusters=2)


@pytest.fixture(scope='module')
def dummy_agent(model):
    # find agent that speaks L1
    for ag in model.schedule.agents:
        if ag.info['language'] == 0:
            agent = ag
            break
    return agent


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
                    (Teacher, Pensioner, ('job', None, False)),
                    (Adult, TeacherUniv, ('job', 'job', False))]


test_data_update_lang_arrays = [({'L1': (np.array([0, 1, 10, 450, 470, 490]),
                                         np.array([3, 2, 1, 2, 1, 3]))}, 'listen'),
                                ({'L1': (np.array([0, 1, 5, 10]),
                                         np.array([3, 2, 1, 1]))}, 'speak'),
                                ({'L1': (np.array([450, 470, 490]),
                                         np.array([3, 2, 1]))}, 'listen')]

test_data_vocab_choice_model = [True, False]

test_data_move_new_home = [(True, None, None), (True, True, None), (True, True, True)]

test_data_get_job = [Young, Teacher, TeacherUniv]


def add_ag_to_world(m, city_places, ag_to_add):
    """ add agent to grid, schedule, clusters, networks """
    m.geo.add_agents_to_grid_and_schedule(ag_to_add)
    m.geo.clusters_info[city_places['home'].info['clust']]['agents'].append(ag_to_add)
    m.nws.add_ags_to_networks(ag_to_add)


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
        add_ag_to_world(model, city_places, old_ag)
    elif origin_class == Teacher:
        ag_id = model.set_available_ids.pop()
        old_ag = origin_class(model, ag_id, 0, 'M', age=1200, home=city_places['home'])
        add_ag_to_world(model, city_places, old_ag)
        school = city_places['school']
        ckey = np.random.choice(list(school.grouped_studs.keys()))
        school.assign_teacher(old_ag, ckey)
    else:
        ag_id = model.set_available_ids.pop()

        old_ag = origin_class(model, ag_id, 0, 'M', age=100, home=city_places['home'],
                              **{labels[0]: city_places[labels[0]]})
        add_ag_to_world(model, city_places, old_ag)
    
    city_places['home'].agents_in.add(old_ag)
    # print(old_ag)
    # print(city_places['home'].agents_in)

    if labels[2]:
        grown_ag = old_ag.evolve(new_class, ret_output=True,
                                 **{labels[1]: city_places[labels[1]]})
    else:
        grown_ag = old_ag.evolve(new_class, ret_output=True)

    # if isinstance(old_ag, Teacher):
    #     city_places['home'].info['occupants'].add(old_ag)
    #     objgraph.show_backrefs([old_ag], filename=str(old_ag) + '_backrefs.png')

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
    # define home and job pairs for each agent so that each pair is in a different cluster
    h1 = model.geo.clusters_info[0]['homes'][1]
    j1 = model.geo.clusters_info[0]['jobs'][0] if job1 else None
    h2 = model.geo.clusters_info[1]['homes'][1]
    j2 = model.geo.clusters_info[1]['jobs'][0] if job2 else None

    new_ags_ids = [model.set_available_ids.pop() for _ in range(2)]
    new_ags_age = 1200

    consort1 = Young(model, new_ags_ids[0],
                     1, 'M', age=new_ags_age, home=h1, job=j1)
    if j2_teach:
        # TODO: TO BE FIXED
        sc1 = model.geo.clusters_info[0]['schools'][0]
        consort2 = Teacher(model, new_ags_ids[1], 1, 'F', age=new_ags_age, home=h2, job=j2)
    else:
        consort2 = Young(model, new_ags_ids[1], 1, 'F', age=new_ags_age, home=h2, job=j2)
    
    ags = [consort1, consort2]
    # add agents to model entities
    for ag in ags:
        model.add_new_agent_to_model(ag)
    # update acquaintance status between agents
    consort1.update_acquaintances(consort2, 0)

    # get married
    consort1.get_married(consort2)
    
    if consort1.loc_info['job'] == j1:
        assert consort1 in consort1.loc_info['job'].info['employees']
    if consort2.loc_info['job'] == j2 and j2:
        assert consort2 in consort2.loc_info['job'].info['employees']


@pytest.mark.parametrize('agent_class', test_data_get_job)
def test_get_job(model, city_places, agent_class):
    ag_id = model.set_available_ids.pop()
    ag = agent_class(model, ag_id, 1, 'M', age=1200, home=city_places['home'])
    add_ag_to_world(model, city_places, ag)
    # get a job
    ag.get_job()

    if ag.loc_info['job']:
        if type(ag) is Adult:
            assert ag in ag.loc_info['job'].info['employees']
        elif type(ag) is Teacher:
            assert ag in ag.loc_info['job'][0].info['employees']
        elif type(ag) is TeacherUniv:
            assert ag in ag.loc_info['job'][0].info['employees']


@pytest.mark.parametrize("sample_words, mode_type", test_data_update_lang_arrays)
def test_update_lang_arrays(dummy_agent, sample_words, mode_type):

    ag = dummy_agent

    # adapt agent retention levels
    ag.lang_stats['L1']['R'][-100:] = 0.
    # get words whose counting will be updated
    act, act_c = sample_words['L1']
    known_words = np.nonzero(ag.lang_stats['L1']['R'] > 0.1)
    kn_act_bool = np.in1d(act, known_words, assume_unique=True)
    act_upd, act_upd_c = act[kn_act_bool], act_c[kn_act_bool]

    wc_init = ag.lang_stats['L1']['wc'][act_upd].copy()

    # check calls through patches
    with patch.object(ag, 'process_unknown_words') as proc_uw:
        with patch.object(ag, 'update_words_memory') as uwm:
            if np.all(kn_act_bool):
                # call method to test
                ag.update_lang_arrays(sample_words, mode_type=mode_type)
                assert proc_uw.call_count == 0
                assert uwm.call_count == 1
            elif np.all(~kn_act_bool):
                proc_uw.return_value = np.array([], dtype=np.int32)
                # call method to test
                ag.update_lang_arrays(sample_words, mode_type=mode_type)
                assert uwm.call_count == 0
            else:
                proc_uw.return_value = np.array([ix for ix, val in enumerate(act)
                                                 if val not in known_words[0]])
                # call method to test
                ag.update_lang_arrays(sample_words, mode_type=mode_type)
                assert proc_uw.call_count == 1
                assert uwm.call_count == 1

    # check updated counts
    if mode_type == 'speak':
        assert np.all(ag.lang_stats['L1']['wc'][act_upd] > wc_init)
    elif mode_type == 'listen':
        assert np.all(ag.lang_stats['L1']['wc'][act_upd] > wc_init)


@pytest.mark.parametrize("sample_words, mode_type", test_data_update_lang_arrays)
def test_process_unknown_words(model, sample_words, mode_type):
    pass



@pytest.mark.parametrize("sample_words, mode_type", test_data_update_lang_arrays)
def test_update_words_memory(model, dummy_agent, sample_words, mode_type):

    agent = dummy_agent

    for lang_label, (act, act_c) in sample_words.items():
        R_init = agent.lang_stats[lang_label]['R'][act].copy()
        S_init = agent.lang_stats[lang_label]['S'][act].copy()
        t_init = agent.lang_stats[lang_label]['t'][act].copy()

        agent.update_words_memory(lang_label, act, act_c)

        assert np.all(agent.lang_stats[lang_label]['R'][act] >= R_init)
        assert np.all(agent.lang_stats[lang_label]['S'][act] > S_init)
        assert np.all(agent.lang_stats[lang_label]['t'][act] <= t_init)

# @pytest.mark.parametrize("long", test_data_vocab_choice_model)
# def test_vocab_choice_model(model, long):
#     agent = model.schedule.agents[0]
#     lang = 0 if agent.language == 0 else 1
#     act, act_c = agent.vocab_choice_model(lang, long)
#     assert len(act) > 0
        