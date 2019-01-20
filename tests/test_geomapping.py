import pytest
from bilangsim import BiLangModel
from bilangsim.agent import Child, Adolescent


@pytest.fixture(scope="module")
def model():
    return BiLangModel(400, num_clusters=1, init_lang_distrib=[0.25, 0.5, 0.25])


def test_add_new_school(model):

    num_schools = len(model.geo.clusters_info[0]['schools'])
    new_school = model.geo.add_new_school(0)
    new_num_schools = len(model.geo.clusters_info[0]['schools'])

    assert new_num_schools == num_schools + 1

