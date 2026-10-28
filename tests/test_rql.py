"""Numerical parity and scientific regression checks for pinned native RQL."""
import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from doublemdp.agents.rql import RQLAgent, get_config

ROOT = Path(__file__).resolve().parents[1]


def upstream_module():
    source = ROOT / 'third_party' / 'rql'
    if not source.exists():
        pytest.fail('Pinned official reference missing; run scripts/setup_references.sh')
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location('pinned_upstream_rql', source / 'agents/rql.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(source))
    return module


def small_config(h=3):
    config = get_config().to_dict()
    config.update(h=h, batch_size=8, actor_hidden_dims=(16,16), value_hidden_dims=(16,16),
                  ensemble_ct=3, flow_steps=4, rho=.5, expectile=.7)
    return config


def batch_for(config):
    rng = np.random.RandomState(781)
    h, b = config['h'], config['batch_size']
    batch = dict(observations=rng.normal(size=(h+1,b,3)).astype('float32'),
                 actions=np.tanh(rng.normal(size=(h+1,b,2))).astype('float32'),
                 rewards=-rng.uniform(size=(h+1,b)).astype('float32'),
                 terminals=np.zeros((h+1,b),dtype='float32'),
                 masks=np.ones((h+1,b),dtype='float32'))
    batch['terminals'][-1,:] = 1
    if h > 1:
        batch['terminals'][1:,0] = 1
        batch['masks'][1:,0] = 0
        batch['terminals'][-2:,1] = 1  # Logging timeout keeps bootstrap.
    return batch


def tree_close(a, b, atol=2e-6, rtol=2e-6):
    aa, ta = jax.tree_util.tree_flatten(a)
    bb, tb = jax.tree_util.tree_flatten(b)
    assert ta == tb
    for x,y in zip(aa,bb):
        np.testing.assert_allclose(x,y,atol=atol,rtol=rtol)


def max_abs(tree):
    return max(float(jnp.max(jnp.abs(v))) for v in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize('h', [1,3])
def test_pinned_loss_gradient_optimizer_target_sampler_parity(h):
    cfg=small_config(h)
    batch=batch_for(cfg)
    local=RQLAgent.create(7,batch['observations'][0],batch['actions'][0],cfg)
    reference=upstream_module().RQLAgent.create(7,batch['observations'][0],batch['actions'][0],dict(cfg))
    tree_close(local.network.params,reference.network.params,0,0)
    key=jax.random.PRNGKey(18)
    local_loss=local.total_loss(batch,local.network.params,key)
    ref_loss=reference.total_loss(batch,reference.network.params,key)
    tree_close(local_loss,ref_loss)
    gl=jax.grad(lambda p:local.total_loss(batch,p,key)[0])(local.network.params)
    gr=jax.grad(lambda p:reference.total_loss(batch,p,key)[0])(reference.network.params)
    tree_close(gl,gr)
    for _ in range(3):
        local, lm=local.update(batch)
        reference, rm=reference.update(batch)
        tree_close(lm,rm)
        tree_close(local.state_dict(), __import__('flax').serialization.to_state_dict(reference))
    for temperature in [0.,1.]:
        tree_close(local.sample_actions(batch['observations'][0,0],key,temperature),
                   reference.sample_actions(batch['observations'][0,0],key,temperature))


def test_gradient_boundaries_and_old_parameter_target_order():
    cfg=small_config()
    batch=batch_for(cfg)
    agent=RQLAgent.create(9,batch['observations'][0],batch['actions'][0],cfg)
    key=jax.random.PRNGKey(19)
    critic_grad=jax.grad(lambda p:agent.total_loss(batch,p,key)[1]['critic_loss'])(agent.network.params)
    actor_grad=jax.grad(lambda p:agent.total_loss(batch,p,key)[1]['actor_loss'])(agent.network.params)
    assert max_abs(critic_grad['modules_actor']) == 0
    assert max_abs(critic_grad['modules_value']) > 0
    assert max_abs(actor_grad['modules_value']) == 0
    assert max_abs(actor_grad['modules_actor']) > 0
    assert max_abs(actor_grad['modules_target_value']) == 0
    for _ in range(2):
        old=agent
        agent,_=agent.update(batch)
        for name,decay in [('value',cfg['tau']),('actor',1-cfg['ema'])]:
            expected=jax.tree_util.tree_map(lambda p,t:decay*p+(1-decay)*t,
                old.network.params['modules_'+name],old.network.params['modules_target_'+name])
            tree_close(agent.network.params['modules_target_'+name],expected)


def test_trace_preserves_actions_and_native_continuations():
    cfg=small_config(1)
    batch=batch_for(cfg)
    agent=RQLAgent.create(3,batch['observations'][0],batch['actions'][0],cfg)
    obs=batch['observations'][0,0]
    key=jax.random.PRNGKey(12)
    trace=agent.sample_trace(obs,key)
    tree_close(trace['final_actions'][0],agent.sample_actions(obs,key))
    assert trace['latents'].shape == (cfg['flow_steps']+1,1,2)
    assert trace['native_values'].shape == (cfg['flow_steps']+1,3,1)
    for step in range(cfg['flow_steps']+1):
        tree_close(agent.continue_flow(obs,trace['latents'][step],step),trace['final_actions'])
    # No plotting-time latent clipping: the supplied >1 Gaussian-like noise is retained.
    trace=agent.sample_trace(obs,key,noise=jnp.array([[3.,-4.]]))
    tree_close(trace['latents'][0],jnp.array([[3.,-4.]]),0,0)
    cycle=agent.reverse_cycle(batch['observations'][0],batch['actions'][0],1.)
    assert np.isfinite(cycle['rmse']) and float(cycle['rmse']) > 1e-7
    zero=agent.reverse_cycle(batch['observations'][0],batch['actions'][0],0.)
    assert float(zero['rmse']) == 0
    print('cycle_rmse=',float(cycle['rmse']))


def test_terminal_bandit_adapter_retains_reward_and_zero_bootstrap():
    from doublemdp.data import OfflineDataset
    cfg=small_config(1)
    data=dict(observations=np.zeros((8,3)),actions=np.ones((8,2))*.2,
              rewards=np.arange(8,dtype=np.float32),masks=np.zeros(8),terminals=np.ones(8),
              next_observations=np.ones((8,3))*1e3)
    dataset=OfflineDataset(data,seed=4,metadata={'terminal_bandit':True})
    batch=dataset.sample_rql(8,h=1)
    agent=RQLAgent.create(3,batch['observations'][0],batch['actions'][0],cfg)
    zeros=jax.tree_util.tree_map(jnp.zeros_like,agent.network.params)
    agent=agent.replace(network=agent.network.replace(params=zeros))
    _,info=agent.total_loss(batch,agent.network.params)
    np.testing.assert_allclose(info['critic_loss'],cfg['expectile']*np.mean(batch['rewards'][0]**2),atol=1e-6)
    altered={**batch,'rewards':batch['rewards'].copy(),'observations':batch['observations'].copy()}
    altered['rewards'][1]=1e6
    altered['observations'][1]=1e6
    _,altered_info=agent.total_loss(altered,agent.network.params)
    tree_close(info['critic_loss'],altered_info['critic_loss'],0,0)
    with pytest.raises(ValueError,match='h=1'):
        dataset.sample_rql(8,h=3)
    dataset.verify_immutable()


def test_checkpoint_restores_rng_targets_optimizer_and_next_update():
    cfg=small_config()
    batch=batch_for(cfg)
    agent=RQLAgent.create(4,batch['observations'][0],batch['actions'][0],cfg)
    agent,_=agent.update(batch)
    template=RQLAgent.create(15,batch['observations'][0],batch['actions'][0],cfg)
    restored=template.restore_bytes(agent.to_bytes())
    tree_close(agent.state_dict(),restored.state_dict(),0,0)
    updated,metrics=agent.update(batch)
    resumed,resume_metrics=restored.update(batch)
    tree_close(updated.state_dict(),resumed.state_dict(),0,0)
    tree_close(metrics,resume_metrics,0,0)


def test_ogbench_manifest_is_fifty_separate_tasks():
    import json
    manifest=json.loads((ROOT/'configs/rql_ogbench_manifest.json').read_text())
    tasks=manifest['tasks']
    assert len(tasks)==50
    assert len({t['task_id'] for t in tasks})==50
    assert all(len(t['seeds'])==4 for t in tasks)
    assert all(t['paper_reference']['score_percent'] is not None for t in tasks)
    assert sum(t['dataset']['nominal_transitions']==100_000_000 for t in tasks)==10
    assert all(t['training_budgets']['official_code']==1_000_000 for t in tasks)
    assert all(t['training_budgets']['paper_budget']==2_000_000 for t in tasks)


def test_ogbench_manifest_ids_are_registered():
    import json
    import gymnasium
    import ogbench  # Registers pinned task IDs.
    manifest=json.loads((ROOT/'configs/rql_ogbench_manifest.json').read_text())
    for task in manifest['tasks']:
        spec=gymnasium.spec(task['gym_id'])
        assert spec.kwargs['reward_task_id']==task['task_number']
        assert spec.kwargs['success_timing']=='pre'


@pytest.mark.parametrize('batch_size',[4,8])
def test_sequence_sampler_matches_pinned_episode_uniform_random_starts(batch_size):
    from doublemdp.data import OfflineDataset
    source=ROOT/'third_party/rql/utils/datasets.py'
    spec=importlib.util.spec_from_file_location('pinned_upstream_datasets',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n=28
    arrays=dict(observations=np.arange(n*3,dtype=np.float32).reshape(n,3),
                actions=np.arange(n*2,dtype=np.float32).reshape(n,2),
                rewards=np.arange(n,dtype=np.float32), masks=np.ones(n,dtype=np.float32),
                terminals=np.zeros(n,dtype=np.float32),next_observations=np.zeros((n,3),dtype=np.float32))
    arrays['terminals'][[3,10,12,20,27]]=1
    arrays['masks'][10]=0  # Other boundaries retain bootstrap.
    local=OfflineDataset(arrays,seed=123)
    reference=module.Dataset.create(**arrays)
    reference.config={'h':5}
    host_state=np.random.get_state()
    np.random.seed(123)
    try:
        for _ in range(4):
            expected=reference.sample_traj(batch_size)
            actual=local.sample_rql(batch_size,h=5)
            for key in expected:
                np.testing.assert_array_equal(actual[key],expected[key])
    finally:
        np.random.set_state(host_state)
