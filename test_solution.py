import sys
sys.path.append('/Users/JDMS/Downloads/coding exercise')

import numpy as np

from solution import FixedEffectsOLS, FEParams


def test_initialization():
    data = [{'city': 'Gotham', 'ltv': 100, 'ldx': 10, 'ls': 5}, {'city': 'Smallville', 'ltv': 200, 'ldx': 20, 'ls': 10}]
    model = FixedEffectsOLS(data, 'city', 'ltv', ['ldx', 'ls'])
    assert set(model.group_names) == set(['Gotham', 'Smallville'])
    assert model.y_by_group['Gotham'] == np.array([[100]])
    assert model.x_by_group['Gotham'].all() == np.array([[10, 5]]).all()


def test_collect_measures_by_group():
    data = [{'city': 'Gotham', 'ltv': 100, 'ldx': 10, 'ls': 5}, {'city': 'Gotham', 'ltv': 150, 'ldx': 15, 'ls': 7.5}]
    model = FixedEffectsOLS(data, 'city', 'ltv', ['ldx', 'ls'])
    measures_y = model.collect_measures_by_group(['ltv'], 'Gotham')
    assert np.array_equal(measures_y, np.array([[100], [150]]))
    measures_x = model.collect_measures_by_group(['ldx', 'ls'], 'Gotham')
    assert np.array_equal(measures_x, np.array([[10, 5], [15, 7.5]]))


def test_bootstrap_sample():
    data_dict = {'Gotham': np.array([1, 2, 3])}
    index_dict = {'Gotham': np.array([0, 2])}
    sample = FixedEffectsOLS.draw_bootstrap_sample(data_dict, index_dict)
    assert np.array_equal(sample['Gotham'], np.array([1, 3]))


def test_param_estimates():
    bootstrap_params = [np.array([1, 2, 5]).reshape(-1, 1), np.array([1, 3, 7]).reshape(-1, 1)]    
    params = FEParams(['param1', 'param2', 'param3'], bootstrap_params)
    assert params.param_estimates['param1'] == 1
    assert params.param_estimates['param2'] == 2.5
    assert params.param_estimates['param3'] == 6


if __name__ == "__main__":
    test_initialization()
    test_collect_measures_by_group()
    test_bootstrap_sample()
    test_param_estimates()