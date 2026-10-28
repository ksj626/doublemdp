import numpy as np
import pytest
from doublemdp.diagnostics import native_progress_indices, grid_dynamic_programming, aggregate_training_seeds


def test_native_steps_not_interpolated():
    np.testing.assert_array_equal(native_progress_indices([0,.2,.4,.6,.8,1]),[0,1,2,4,5])


def test_grid_reference_has_path_and_bellman_convergence():
    result=grid_dynamic_programming(31)
    assert result["bellman_residual"]<1e-8
    assert np.max(result["values"])>0
    assert np.all(result["values"][~result["valid"]]==0)


def test_uncertainty_unit_is_training_seed():
    with pytest.raises(ValueError,match="Repeated training seed"):
        aggregate_training_seeds([{"seed":0,"score":1},{"seed":0,"score":0}])
    assert aggregate_training_seeds([{"seed":0,"score":1}])["ci95"] is None
