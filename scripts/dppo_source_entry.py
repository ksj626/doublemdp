"""Pinned source entry point with observation-only physical-step telemetry.

The source counts configured chunk lengths, including actions skipped after
termination. Count actual inner environment.step calls without changing any
observation, reward, model, optimizer, RNG draw, or action execution.
"""
import functools
import json
import os
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
upstream=Path(os.environ.get("DPPO_UPSTREAM_ROOT",root/"third_party/dppo"))
sys.path.insert(0,str(upstream))
sys.path.insert(0,str(upstream/"script"))

if os.environ.get("DPPO_COUNT_PHYSICAL_STEPS")=="1":
    from env.gym_utils.wrapper.mujoco_locomotion_lowdim import MujocoLocomotionLowdimWrapper
    from env.gym_utils.wrapper.multi_step import MultiStep
    from agent.finetune.train_agent import TrainAgent
    from agent.eval.eval_agent import EvalAgent
    low_step=MujocoLocomotionLowdimWrapper.step
    def counted_low_step(self,action):
        result=low_step(self,action)
        self._doublemdp_physical_steps=getattr(self,"_doublemdp_physical_steps",0)+1
        return result
    MujocoLocomotionLowdimWrapper.step=counted_low_step
    chunk_step=MultiStep.step
    def counted_chunk_step(self,action):
        previous=getattr(self.env,"_doublemdp_physical_steps",0)
        result=chunk_step(self,action)
        result[-1]["doublemdp_physical_steps"]=getattr(self.env,"_doublemdp_physical_steps",0)-previous
        return result
    MultiStep.step=counted_chunk_step
    def attach_counter(self):
        counts={"training_env_steps":0,"evaluation_env_steps":0,"by_iteration":{},
                "kind":"observed physical environment.step calls; excludes skipped chunk actions"}
        vector_step=self.venv.step
        calls=0
        def counted_vector_step(actions):
            nonlocal calls
            result=vector_step(actions)
            physical=sum(int(info["doublemdp_physical_steps"]) for info in result[-1])
            mode="training" if self.model.training else "evaluation"
            counts[mode+"_env_steps"]+=physical
            key=str(getattr(self,"itr",0))
            counts["by_iteration"].setdefault(key,{"mode":mode,"env_steps":0})
            counts["by_iteration"][key]["env_steps"]+=physical
            calls+=1
            if calls%self.n_steps==0:
                Path(self.logdir,"physical_interactions.json").write_text(json.dumps(counts,indent=2)+"\n")
            return result
        self.venv.step=counted_vector_step
    def wrap_init(original_init):
        @functools.wraps(original_init)
        def counted_init(self,cfg):
            original_init(self,cfg)
            attach_counter(self)
        return counted_init
    TrainAgent.__init__=wrap_init(TrainAgent.__init__)
    EvalAgent.__init__=wrap_init(EvalAgent.__init__)

runpy.run_path(str(upstream/"script/run.py"),run_name="__main__")
