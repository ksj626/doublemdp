import os
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np
import pytest
pytest.importorskip("ogbench")
from doublemdp.envs import make_env
from doublemdp.agents.dppo_env_state import StatefulOGBench


@pytest.mark.parametrize("task", ["pointmaze-medium-navigate-singletask-v0", "cube-double-play-singletask-v0"])
def test_ogbench_exact_simulator_rng_wrapper_resume(task):
    original=StatefulOGBench(make_env(task))
    restored=StatefulOGBench(make_env(task))
    original.reset(seed=12)
    restored.reset(seed=83)
    rng=np.random.RandomState(7)
    shape=original.action_space.shape
    for _ in range(3):
        original.step(rng.uniform(-.5,.5,shape))
    snapshot=original.state_dict()
    actions = [rng.uniform(-.5,.5,shape) for _ in range(5)]
    expected = [original.step(action) for action in actions]
    expected_reset = original.reset()[0]
    restored.load_state_dict(snapshot)
    for action, a in zip(actions, expected):
        b=restored.step(action)
        np.testing.assert_allclose(a[0],b[0],rtol=0,atol=1e-10)
        assert a[1:4]==b[1:4]
    # Reset consumes task RNG; it must resume identically as well.
    np.testing.assert_allclose(expected_reset,restored.reset()[0],rtol=0,atol=1e-10)
    original.close()
    restored.close()
