import numpy as np
import pytest
from experiment_runner import experiment,validate_data

def test_reproducible(tmp_path):
    a=experiment(tmp_path/'a');b=experiment(tmp_path/'b')
    assert a==b and a['experiments']==6

def test_invalid_data_rejected():
    with pytest.raises(ValueError):
        validate_data(np.array([[np.nan]]),np.array([2.]))
