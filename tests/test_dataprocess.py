import pytest

from bilangsim import BiLangModel


@pytest.fixture(scope="module")
def model():
    return BiLangModel(251, num_clusters=2, check_setup=True)


def test_save_model_data():
    pass
