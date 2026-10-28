from types import SimpleNamespace
import numpy as np
import pytest
from doublemdp.agents.bdpo_probe import regularized_maze_continuation


class FakeAgent:
    def __init__(self):
        self.cfg=SimpleNamespace(critic=SimpleNamespace(rho=0.,maxQ=False,q_target='lcb',eta=.1,discount=.5))
        self._is_pretraining=False
    def continue_from_latent(self,obs,latent,time,seed,return_info):
        return {'actions':np.zeros_like(latent),'remaining_path_kl':np.full(len(obs),3. if time else 0.)}
    def sample_trace(self,obs,seed,num_samples):
        return {'actions':np.zeros_like(obs),'step_path_kl':np.full((1,len(obs),1),2.)}
    def q0_target(self,obs,actions):
        return np.zeros((2,len(obs),1))
    def vt_target(self,obs,actions,times):
        return np.zeros((2,len(obs),1))


class TwoStepEnvironment:
    def transition(self,states,actions):
        next_states=states+1
        done=next_states[:,0]>=2
        return next_states,np.ones(len(states)),np.full(len(states),2.),done,np.zeros(len(states),bool)


def test_current_remaining_and_future_outer_penalties_have_correct_discounts():
    agent=FakeAgent()
    result=regularized_maze_continuation(agent,np.zeros((2,2)),np.ones((2,2)),[1,0],
                                        repetitions=3,horizon=5,environment=TwoStepEnvironment())
    # V1 = -eta*3 + r0 + gamma*(r1-eta*2); Q0 omits current generation cost.
    np.testing.assert_allclose(result['mean'],[1.1,1.4])
    np.testing.assert_allclose(result['discounted_path_penalty_mean'],[4.,1.])
    np.testing.assert_allclose(result['unfinished_fraction'],0.)
    np.testing.assert_allclose(result['se'],0.,atol=1e-15)
    result=regularized_maze_continuation(agent,np.zeros((1,2)),np.ones((1,2)),1,
        repetitions=3,horizon=1,environment=TwoStepEnvironment())
    np.testing.assert_allclose(result['mean'],[.7])
    np.testing.assert_allclose(result['unfinished_fraction'],[1.])


def test_pessimistic_heads_cannot_claim_unbiased_continuation_calibration():
    agent=FakeAgent();agent.cfg.critic.rho=.5
    with pytest.raises(ValueError,match='rho=0'):
        regularized_maze_continuation(agent,np.zeros((1,2)),np.zeros((1,2)),1)
