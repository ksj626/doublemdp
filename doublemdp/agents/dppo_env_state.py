"""Exact state adapter for tested OGBench 1.2.0 state-observation MuJoCo tasks.

Save task variables and wrapper counters in addition to full integration state.
MuJoCo integration state includes warm starts, control, mocap, time, and RNG-
independent simulator state. A qpos/qvel-only checkpoint is insufficient.
"""
import copy
import importlib.metadata
import numpy as np


def _plain(value):
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic, np.ndarray,
                                           np.random.RandomState, np.random.Generator)):
        return True
    if isinstance(value, (tuple, list)):
        return all(_plain(v) for v in value)
    if isinstance(value, dict):
        return all(_plain(k) and _plain(v) for k,v in value.items())
    return False


def _objects(env):
    result = []
    while True:
        result.append(env)
        if "env" not in env.__dict__:
            return result
        env = env.__dict__["env"]


class StatefulOGBench:
    def __init__(self, env):
        if importlib.metadata.version("ogbench") != "1.2.0":
            raise ValueError("Checkpoint adapter audited for OGBench 1.2.0")
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def state_dict(self):
        import mujoco
        base = self.env.unwrapped
        if getattr(base, "_ob_type", "states") != "states":
            raise NotImplementedError("Pixel/rendering-state resume is not validated")
        model, data = base.model, base.data
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        physics = np.empty(mujoco.mj_stateSize(model, specification), dtype=np.float64)
        mujoco.mj_getState(model, data, physics, specification)
        # Goal markers, task locks and physical parameters can change on reset.
        model_arrays = {}
        for name in dir(model):
            value = getattr(model, name)
            if isinstance(value, np.ndarray) and value.flags.writeable:
                model_arrays[name] = value.copy()
        # mj_step leaves some derived fields at the last integration substep.
        # Controllers read those fields before the next mj_forward; recomputing
        # them here would subtly change the next action (notably manipulation).
        data_arrays = {}
        for name in dir(data):
            value = getattr(data, name)
            if isinstance(value, np.ndarray) and value.flags.writeable:
                data_arrays[name] = value.copy()
        python = []
        for obj in _objects(self.env):
            python.append({name:copy.deepcopy(value) for name,value in vars(obj).items()
                           if name != "env" and _plain(value)})
        return {"physics":physics, "specification":int(specification), "python":python,
                "model_arrays":model_arrays, "data_arrays":data_arrays,
                "global_numpy_rng":np.random.get_state(), "mujoco_version":mujoco.__version__,
                "env_id":getattr(self.env.spec,"id",None),
                "wrapper_types":[type(obj).__qualname__ for obj in _objects(self.env)]}

    def load_state_dict(self, state):
        import mujoco
        if state["mujoco_version"] != mujoco.__version__:
            raise ValueError("Exact simulator resume requires the saved MuJoCo version")
        objects = _objects(self.env)
        if state["wrapper_types"] != [type(obj).__qualname__ for obj in objects]:
            raise ValueError("Checkpoint wrapper stack differs")
        if state["env_id"] != getattr(self.env.spec,"id",None):
            raise ValueError("Checkpoint environment differs")
        # The caller resets once before restore so manipulation models compile.
        base = self.env.unwrapped
        for obj, variables in zip(objects, state["python"]):
            for name,value in variables.items():
                setattr(obj,name,copy.deepcopy(value))
        for name,value in state["model_arrays"].items():
            getattr(base.model,name)[:] = value
        mujoco.mj_setState(base.model,base.data,state["physics"],state["specification"])
        mujoco.mj_forward(base.model,base.data)
        # Forward initializes variable-size caches; overwrite saved native
        # arrays afterward, preserving their original integration timing.
        for name,value in state["data_arrays"].items():
            target = getattr(base.data,name)
            if target.shape != value.shape:
                raise ValueError(f"Simulator cache shape changed for {name}; exact resume unavailable")
            target[:] = value
        np.random.set_state(state["global_numpy_rng"])

