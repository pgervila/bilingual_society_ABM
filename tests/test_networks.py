import pytest
from bilangsim.model import BiLangModel
from bilangsim.agent import Child, Adolescent, Young, Pensioner


@pytest.fixture(scope="module")
def model():
    return BiLangModel(100, num_clusters=1, init_lang_distrib=[0.5, 0.5, 0.])


@pytest.fixture(scope="function")
def newborn_family(model):
    father = [ag for ag in model.schedule.agents if type(ag) == Child and
              ag.info['sex'] == 'M' and ag.info['language'] in [0, 1]][0]
    mother = [ag for ag in model.schedule.agents if type(ag) == Child
              and ag.info['sex'] == 'F' and ag.info['language'] in [0, 1] and
              ag not in father.get_family_relative('sibling')][0]
    father.grow(growth_inc=model.steps_per_year*20 - father.info['language'])
    mother.grow(growth_inc=model.steps_per_year*20 - mother.info['language'])

    father = father.evolve(Adolescent, ret_output=True)
    father = father.evolve(Young, ret_output=True)

    mother = mother.evolve(Adolescent, ret_output=True)
    mother = mother.evolve(Young, ret_output=True)

    father.get_job(ignore_lang_constraint=True)
    father.update_acquaintances(mother, 0)
    father.get_married(mother)
    father.reproduce(day_prob=1.)

    child = father.get_family_relative('child')[0]

    grandfathers = [father.get_family_relative('father'), mother.get_family_relative('father')]
    grandmothers = [father.get_family_relative('mother'), mother.get_family_relative('mother')]
    uncles_and_aunts = father.get_family_relative('sibling') + mother.get_family_relative('sibling')
    uncles = [x for x in uncles_and_aunts if x.info['sex'] == 'M']
    aunts = [x for x in uncles_and_aunts if x.info['sex'] == 'F']
    cousins = [cousin for x in uncles_and_aunts for cousin in x.get_family_relative('child')]

    return father, mother, child, grandfathers, grandmothers, uncles, aunts, cousins


def test_networks_consistency(model):
    family = model.schedule.agents[:4]
    # test family network
    father, mother, child_1, child_2 = family
    assert model.nws.family_network[father][mother]['fam_link'] == 'consort'
    assert model.nws.family_network[father][child_1]['fam_link'] == 'child'
    assert model.nws.family_network[child_1][mother]['fam_link'] == 'mother'
    assert model.nws.family_network[child_1][child_2]['fam_link'] == 'sibling'


def test_newborn_family_links(newborn_family):

    father, mother, child, grandfathers, grandmothers, uncles, aunts, cousins = newborn_family

    assert child.get_family_relative('father') == father
    assert child in father.get_family_relative('child')
    assert child.get_family_relative('mother') == mother
    assert child in mother.get_family_relative('child')
    assert all(x in child.get_family_relative('grandfather') for x in grandfathers)
    assert child in grandfathers[0].get_family_relative('grandchild')
    assert all(x in child.get_family_relative('grandmother') for x in grandmothers)
    assert child in grandmothers[0].get_family_relative('grandchild')
    if uncles:
        assert all(x in child.get_family_relative('uncle') for x in uncles)
        assert child in uncles[0].get_family_relative('nephew')
    if aunts:
        assert all(x in child.get_family_relative('aunt') for x in aunts)
        assert child in aunts[0].get_family_relative('nephew')
    if cousins:
        assert all(x in child.get_family_relative('cousin') for x in cousins)
        assert child in cousins[0].get_family_relative('cousin')


def test_set_link_with_relatives(model):
    home = [home for home in model.geo.clusters_info[0]['homes'] if not home.info['occupants']][0]
    new_agent = Pensioner(model, 1234, 0, sex='M', age=2000,
                          home=home)
    father = model.schedule.agents[0]
    child = father.get_family_relative('child')[0]

    model.nws.set_link_with_relatives(father, new_agent, 'father', lang_with_relatives=0)
    model.nws.set_link_with_relatives(child, new_agent, 'grandfather', lang_with_relatives=0)

    assert new_agent == father.get_family_relative('father')
    assert father in new_agent.get_family_relative('child')
    assert new_agent in child.get_family_relative('grandfather')
    assert child in new_agent.get_family_relative('grandchild')


def test_friends_per_agent(model):
    assert max([len(model.nws.friendship_network[ag]) for ag in model.schedule.agents]) <= 10



