"""Policy-specific evaluation must respect physical and genuine terminal states."""
import numpy as np
from doublemdp.toy import PointMaze, segments_hit_boxes
from doublemdp.diagnostics import rollout_continuations


def test_terminal_and_invalid_initial_states_do_not_query_policy(monkeypatch):
    from doublemdp import evaluate
    def forbidden(*args,**kwargs):
        raise AssertionError('No action can execute from terminal/invalid starts')
    monkeypatch.setattr(evaluate,'policy_actions',forbidden)
    states=np.array([PointMaze.goal,[.45,.55],[-.1,.2],[1.1,.2]])
    result=rollout_continuations(None,'rql',states,repetitions=3,horizon=5)
    np.testing.assert_array_equal(result['valid_start'],[True,False,False,False])
    assert result['terminal_start'][0]
    assert result['mean'][0]==0 and result['se'][0]==0
    assert np.isnan(result['mean'][1:]).all()
    assert not result['unfinished_fraction'].any()


def test_mixed_rollouts_only_execute_actions_at_valid_nonterminal_starts(monkeypatch):
    from doublemdp import evaluate
    observed=[]
    def zero_actions(agent,states,method,*args,**kwargs):
        observed.extend(np.asarray(states))
        return np.zeros((len(states),1,2))
    monkeypatch.setattr(evaluate,'policy_actions',zero_actions)
    states=np.array([PointMaze.goal,PointMaze.start,[.45,.55]])
    result=rollout_continuations(None,'rql',states,repetitions=2,horizon=4)
    assert len(observed)==8
    np.testing.assert_allclose(observed,np.repeat(PointMaze.start[None],8,axis=0))
    assert result['mean'][0]==0 and result['mean'][1]==0
    assert np.isnan(result['mean'][2])
    np.testing.assert_array_equal(result['unfinished_fraction'],[0,1,0])


def test_all_fixed_probes_are_valid_and_nonterminal():
    probes=PointMaze.probes
    assert not segments_hit_boxes(probes,probes,PointMaze.boxes).any()
    assert np.all(np.linalg.norm(probes-PointMaze.goal,axis=-1)>=.055)
