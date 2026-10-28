import numpy as np
from doublemdp.visualize import _final_actions


def test_decoded_actions_are_distinct_from_unbounded_final_latents():
    data={"probe0_latents":np.array([[[3.,-2.]]]),"probe0_final_actions":np.array([[[1.,-1.]]])}
    np.testing.assert_array_equal(_final_actions(data,0),[[1.,-1.]])


def test_old_bandit_grid_collision_recovers_saved_actual_candidates():
    data={"probe0_actions":np.zeros((41,41,2)),"probe0_candidate_actions":np.array([[[.1,.2]],[[.3,.4]]])}
    np.testing.assert_array_equal(_final_actions(data,0),[[.1,.2],[.3,.4]])
