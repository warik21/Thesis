###
#
#
#
# THIS IS NOT FUNCTIONAL YET
#
#
#
###
import numpy as np

import pytest
from utils.utils import calc_transport_pot_sinkhorn, is_valid_transport_plan, calc_transport_pot_emd

"""
Code Analysis:
- The main goal of the function is to calculate the optimal transport plan between two probability distributions using the Sinkhorn algorithm.
- It takes in three inputs: the source distribution, the target distribution, and the cost matrix.
- The function also has an optional input for the regularization parameter.
- It first calculates the kernel matrix K_t using the cost matrix and the regularization parameter.
- It then uses the Sinkhorn algorithm from the ot library to calculate the optimal transport cost and the logs.
- The logs contain the dual variables u and v, which are flattened and stored in separate numpy arrays.
- The function then calculates the transport plan using the flattened u and v arrays and the kernel matrix K_t.
- The function returns four outputs: the transport plan, the transport cost, the u array, and the v array.
"""

"""
Test Plan:
- test_valid_inputs(): tests that the function returns the expected outputs when valid inputs are provided. Tags: [happy path]
- test_regularization_param(): tests that the function returns the expected outputs when a valid regularization parameter is provided. Tags: [happy path]
- test_empty_source_distribution(): tests that the function handles the case when the source distribution is empty. Tags: [edge case]
- test_empty_target_distribution(): tests that the function handles the case when the target distribution is empty. Tags: [edge case]
- test_unbalanced_data(): tests that the function handles unbalanced data. Tags: [general behavior]
- test_empty_cost_matrix(): tests that the function handles the case when the cost matrix is empty. Tags: [edge case]
- test_different_cost_matrix_dimensions(): tests that the function handles the case when the cost matrix has different dimensions than the source and target distributions. Tags: [edge case]
- test_zero_regularization_param(): tests that the function handles the case when the regularization parameter is zero. Tags: [edge case]
- test_negative_regularization_param(): tests that the function handles the case when the regularization parameter is negative. Tags: [edge case]
- test_large_inputs(): tests that the function handles large inputs. Tags: [general behavior]
"""



class Test_calc_transport_pot_sinkhorn:
    def test_valid_inputs(self):
        source = np.array([1, 0, 2, 0])
        target = np.array([0, 1, 0, 2])
        costs = np.array([[0.        , 0.33333333, 0.66666667, 1.        ],
                          [0.33333333, 0.        , 0.33333333, 0.66666667],
                          [0.66666667, 0.33333333, 0.        , 0.33333333],
                          [1.        , 0.66666667, 0.33333333, 0.        ]])
        Transport_plan, Transport_cost, u, v = calc_transport_pot_sinkhorn(source, target, costs)
        assert is_valid_transport_plan(Transport_plan)

    def test_regularization_param(self):
        source = np.array([0.2, 0.3, 0.5])
        target = np.array([0.1, 0.4, 0.5])
        costs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Transport_plan, Transport_cost, u, v = calc_transport_pot_sinkhorn(source, target, costs, reg_param=1.e-2)
        assert np.allclose(Transport_plan, np.array([[0.02, 0., 0.], [0., 0.12, 0.], [0., 0., 0.25]]))
        assert np.isclose(Transport_cost, 2.6)

    def test_empty_source_distribution(self):
        source = np.array([])
        target = np.array([0.1, 0.4, 0.5])
        costs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            calc_transport_pot_sinkhorn(source, target, costs)

    def test_empty_target_distribution(self):
        source = np.array([0.2, 0.3, 0.5])
        target = np.array([])
        costs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            calc_transport_pot_sinkhorn(source, target, costs)

    def test_unbalanced_data(self):
        source = np.array([0.2, 0.3, 0.5])
        target = np.array([0.1, 0.4])
        costs = np.array([[1, 2], [4, 5], [7, 8]])
        Transport_plan, Transport_cost, u, v = calc_transport_pot_sinkhorn(source, target, costs)
        assert np.allclose(Transport_plan, np.array([[0.02, 0.], [0., 0.12], [0., 0.]]))
        assert np.isclose(Transport_cost, 2.6)

    def test_empty_cost_matrix(self):
        source = np.array([0.2, 0.3, 0.5])
        target = np.array([0.1, 0.4, 0.5])
        costs = np.array([])
        with pytest.raises(ValueError):
            calc_transport_pot_sinkhorn(source, target, costs)


"""
Code Analysis:
- The main goal of the function is to calculate the optimal transport plan and cost between two distributions using Earth Mover's Distance (EMD) algorithm.
- The function takes three inputs: source distribution, target distribution, and cost matrix.
- The source and target distributions are numpy arrays representing the probability distribution of two sets of data.
- The cost matrix is a numpy array representing the cost of transporting a unit of mass from each element in the source distribution to each element in the target distribution.
- The function uses the ot.emd() function from the ot package to calculate the optimal transport plan between the two distributions.
- The ot.emd() function takes the flattened source and target distributions and the cost matrix as inputs and returns the optimal transport plan as a numpy array.
- The function then calculates the total cost of transporting the mass according to the optimal transport plan by multiplying the transport plan with the cost matrix and summing the result.
- The function returns two outputs: the optimal transport plan and the total cost of transporting the mass according to the plan.
"""

"""
Test Plan:
- test_valid_input(): tests that the function works correctly with valid input. Tags: [happy path]
- test_empty_source(): tests that the function returns an error when the source is an empty numpy array. Tags: [edge case]
- test_empty_target(): tests that the function returns an error when the target is an empty numpy array. Tags: [edge case]
- test_non_symmetric_costs(): tests that the function returns an error when the cost matrix is not symmetric. Tags: [general behavior]
- test_empty_costs(): tests that the function returns an error when the costs are an empty numpy array. Tags: [edge case]
- test_single_source(): tests that the function returns an error when the source has only one element. Tags: [edge case]
- test_single_target(): tests that the function returns an error when the target has only one element. Tags: [edge case]
- test_different_shapes(): tests that the function returns an error when the source and target have different shapes than the costs. Tags: [edge case]
- test_negative_values(): tests that the function returns an error when the cost matrix or the distributions have negative values. Tags: [general behavior]
- test_sum_to_one(): tests that the function returns an error when the source or target distributions do not sum to 1. Tags: [general behavior]
"""



class Test_calc_transport_pot_emd:
    def test_valid_input(self):
        source = np.array([1, 0, 2, 0])
        target = np.array([0, 1, 0, 2])
        costs = np.array([[0., 0.33333333, 0.66666667, 1.],
                          [0.33333333, 0., 0.33333333, 0.66666667],
                          [0.66666667, 0.33333333, 0., 0.33333333],
                          [1., 0.66666667, 0.33333333, 0.]])
        Transport_plan, Transport_cost, u, v = calc_transport_pot_sinkhorn(source, target, costs)
        assert is_valid_transport_plan(Transport_plan)
        assert np.isclose(Transport_cost, 1.3333333333333333)

    def test_empty_source(self):
        source = np.array([])
        target = np.array([0.1, 0.4, 0.5])
        costs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            calc_transport_pot_emd(source, target, costs)

    def test_empty_target(self):
        source = np.array([0.2, 0.3, 0.5])
        target = np.array([])
        costs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            calc_transport_pot_emd(source, target, costs)

    def test_non_symmetric_costs(self):
        source = np.array([0.2, 0.3, 0.5])
        target = np.array([0.1, 0.4, 0.5])
        costs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        with pytest.raises(ValueError):
            calc_transport_pot_emd(source, target, costs)

    def test_empty_costs(self):
        source = np.array([0.2, 0.3, 0.5])
        target = np.array([0.1, 0.4, 0.5])
        costs = np.array([])
        with pytest.raises(ValueError):
            calc_transport_pot_emd(source, target, costs)

    def test_single_source(self):
        source = np.array([0.2])
        target = np.array([0.1, 0.4, 0.5])
        costs = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            calc_transport_pot_emd(source, target, costs)

