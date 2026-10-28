"""Pinned-reference numerical parity and checks for scientific failure modes."""
import ast
import importlib.util
import pickle
from pathlib import Path
import sys
import types

import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp
from doublemdp.agents.bdpo import BDPOAgent, default_config, get_penalty, get_target, jit_update_critic
from doublemdp.agents.bdpo_data import generate_8gaussians, qlearning_dataset, ScanSampler
from doublemdp.agents.bdpo_diffusion import vp_beta_schedule, cosine_beta_schedule, linear_beta_schedule
from doublemdp.agents.bdpo_nn import Batch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/flow-rl'


def upstream_module():
    sys.path.insert(0, str(UPSTREAM))
    # Import exactly the pinned BDPO module; the package's __init__ otherwise
    # eagerly imports unrelated algorithms and dependencies such as Diffrax.
    name = 'flowrl.agent.offline'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(UPSTREAM/'flowrl/agent/offline')]
        sys.modules[name] = package
    path = UPSTREAM/'flowrl/agent/offline/bdpo/bdpo.py'
    spec = importlib.util.spec_from_file_location('pinned_bdpo_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_tree_equal(a, b, atol=2e-6):
    a_leaves, b_leaves = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(a_leaves) == len(b_leaves)
    for x,y in zip(a_leaves, b_leaves):
        np.testing.assert_allclose(x, y, atol=atol, rtol=2e-5)


def examples():
    return generate_8gaussians(8, 17)


def make(config=None):
    batch = examples()
    return BDPOAgent.create(5, batch['observations'], batch['actions'], config or default_config('smoke'))


def native_batch(batch):
    return Batch(batch['observations'], batch['actions'], batch['rewards'][:,None],
                 1-batch['masks'][:,None], batch['next_observations'], None)


def test_actual_upstream_parameters_losses_optimizer_ema_sampler_parity():
    upstream = upstream_module()
    agent = make()
    reference = upstream.BDPOAgent(1,2,agent.cfg,5)
    data = examples()
    local_batch = native_batch(data)
    up_batch = upstream.Batch(**local_batch.__dict__)
    for name in agent.model_names:
        assert_tree_equal(getattr(agent,name).params, getattr(reference,name).params)
    for step in range(2):
        got = agent.pretrain_step(local_batch,step)
        expected = reference.pretrain_step(up_batch,step)
        assert_tree_equal(got,expected)
    agent.prepare_training(); reference.prepare_training()
    for step in range(6):
        got = agent.train_step(local_batch,step)
        expected = reference.train_step(up_batch,step)
        assert_tree_equal(got,expected)
        for name in agent.model_names:
            assert_tree_equal(getattr(agent,name).state.params, getattr(reference,name).state.params)
            assert_tree_equal(getattr(agent,name).state.opt_state, getattr(reference,name).state.opt_state)
            assert_tree_equal(getattr(agent,name).dropout_rng, getattr(reference,name).dropout_rng)
    assert_tree_equal(agent.rng,reference.rng)
    rng = agent.rng
    expected,_ = reference.sample_actions(jnp.asarray(data['observations']), num_samples=3)
    got = agent.sample_actions(data['observations'],seed=rng,num_samples=3)
    assert_tree_equal(got,expected)
    import flowrl.flow.ddpm as ref_ddpm
    for schedule in (vp_beta_schedule,cosine_beta_schedule,linear_beta_schedule):
        assert_tree_equal(schedule(50),getattr(ref_ddpm,schedule.__name__)(50))
    # Full paths and one-step gradients use the same supplied PRNG draws.
    model, ref_model = agent.actor, reference.actor
    key = jax.random.PRNGKey(92)
    x = jnp.asarray(data['actions']); obs = jnp.asarray(data['observations'])
    assert_tree_equal(model.sample(key,x,obs),ref_model.sample(key,x,obs))
    def loss(params, model):
        result = model.onestep_sample(key,x,obs,True,2,'ddpm',True,None,params)
        return jnp.square(result[1]).mean()
    grad = jax.grad(loss)(model.params,model)
    ref_grad = jax.grad(loss)(ref_model.params,ref_model)
    assert_tree_equal(grad,ref_grad)
    assert sum(float(jnp.sum(g*g)) for g in jax.tree_util.tree_leaves(grad)) > 0


def test_kl_coefficients_reductions_and_gradients():
    agent = make()
    model = agent.actor
    t = jnp.array([[1],[3],[5]])
    actor = jnp.array([[.3,-.4],[.1,.9],[-.8,.2]])
    behavior = jnp.zeros_like(actor)
    coefficient = .5*model.betas[t]/model.alphas[t]/(1-model.alpha_hats[t])
    actual = get_penalty(actor,behavior,t,5,model.alphas,model.alpha_hats,model.betas)
    np.testing.assert_allclose(actual,coefficient*actor**2,rtol=1e-6)
    gradient = jax.grad(lambda a: get_penalty(a,behavior,t,5,model.alphas,model.alpha_hats,model.betas).sum())(actor)
    np.testing.assert_allclose(gradient,2*coefficient*actor,rtol=1e-6)
    # Sample reduction MUST precede ensemble reduction.
    values = jnp.array([[[[0.],[8.]]],[[[6.],[2.]]]])
    _, target = get_target(jax.random.PRNGKey(0),values,True,'lcb',1.)
    np.testing.assert_allclose(target,[[6.]])


def critic_update(agent, batch, discount):
    c = agent.cfg.critic
    return jit_update_critic(agent.rng,agent.q0,agent.q0_target,agent.vt,agent.vt_target,
        agent.actor_target,agent.behavior_target,batch,agent.cfg.diffusion.steps,
        discount,c.eta,c.rho,c.num_samples,c.q_target,c.maxQ,c.solver,c.ema,True)


def test_no_environment_discount_in_inner_backup_and_terminal_outer_target():
    agent = make(); agent.prepare_training()
    batch = native_batch(examples())
    no_discount = critic_update(agent,batch,0.)
    discounted = critic_update(agent,batch,.99)
    # All samples are terminal: Q0 only sees the saved dataset reward.
    assert_tree_equal(no_discount[1].params,discounted[1].params)
    assert_tree_equal(no_discount[3].params,discounted[3].params)
    continuing = batch.replace(terminal=jnp.zeros_like(batch.terminal))
    a = critic_update(agent,continuing,0.)
    b = critic_update(agent,continuing,.99)
    assert_tree_equal(a[3].params,b[3].params)
    assert any(not np.allclose(x,y) for x,y in zip(jax.tree_util.tree_leaves(a[1].params),jax.tree_util.tree_leaves(b[1].params)))


def test_terminal_noise_endpoint_trace_rng_and_checkpoint():
    agent = make(); agent.prepare_training()
    batch = examples(); obs = jnp.asarray(batch['observations']); x = jnp.asarray(batch['actions'])
    t = jnp.ones((len(x),1),jnp.int32)
    a = agent.actor.onestep_sample(jax.random.PRNGKey(1),x,obs,False,4,'ddpm',False,t)[1]
    b = agent.actor.onestep_sample(jax.random.PRNGKey(2),x,obs,False,4,'ddpm',False,t)[1]
    assert_tree_equal(a,b)
    trace = agent.sample_trace(obs,seed=29,num_samples=3)
    assert_tree_equal(trace['actions'],agent.sample_actions(obs,seed=29,num_samples=3))
    assert trace['latents'].shape == (6,8,3,2)
    assert np.array_equal(trace['native_times'],[5,4,3,2,1,0])
    assert_tree_equal(agent.continue_from_latent(obs,x,0,seed=3),x)
    final = agent.continue_from_latent(obs,x,1,seed=3)
    assert_tree_equal(final,a[:,0])
    saved = pickle.loads(pickle.dumps(agent))
    assert_tree_equal(saved.sample_actions(obs),agent.sample_actions(obs))
    for value in batch.values():
        value.setflags(write=False)
    _,a = agent.update(batch); _,b = saved.update(batch)
    assert_tree_equal(a,b)
    assert_tree_equal(agent.state_dict()['models'],saved.state_dict()['models'])
    assert saved._n_training_steps == agent._n_training_steps


def test_behavior_pretrain_handoff_and_delayed_actor():
    cfg = default_config('smoke'); cfg['pretrain_steps'] = 2
    agent = make(cfg); batch = examples()
    agent.update(batch)
    restored = pickle.loads(pickle.dumps(agent))
    agent.update(batch); restored.update(batch)
    assert not agent._is_pretraining
    assert_tree_equal(agent.actor.params,agent.behavior_target.params)
    assert_tree_equal(agent.actor.params,restored.actor.params)
    agent.update(batch) # training step zero: actor is updated
    actor = agent.actor.params
    behavior = agent.behavior.params
    agent.update(batch) # step one: critic only
    assert_tree_equal(actor,agent.actor.params)
    assert_tree_equal(behavior,agent.behavior.params)


def test_original_toy_draws_and_d4rl_boundary_semantics():
    source=(UPSTREAM/'flowrl/dataset/toy2d.py').read_text()
    tree=ast.parse(source)
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_compute_energy','inf_train_gen')],type_ignores=[])
    namespace={'np': np}; exec(compile(selected,'pinned_toy2d','exec'),namespace)
    np.random.seed(17)
    reference_actions, reference_rewards=namespace['inf_train_gen']('8gaussians',batch_size=8)
    data=examples()
    np.testing.assert_array_equal(data['actions'],reference_actions)
    np.testing.assert_array_equal(data['rewards'],reference_rewards[:,0])
    raw={'observations':np.arange(7,dtype=np.float32)[:,None], 'actions':np.zeros((7,1)),
         'rewards':np.ones(7), 'terminals':np.array([0,0,0,0,1,0,0]),
         'timeouts':np.array([0,0,1,0,0,0,0])}
    processed=qlearning_dataset(types.SimpleNamespace(_max_episode_steps=1000),raw)
    np.testing.assert_array_equal(processed['observations'][:,0],[0,1,3,4,5])
    np.testing.assert_array_equal(processed['terminals'],[0,0,0,1,0])
    np.testing.assert_array_equal(processed['ends'],[0,1,0,1,1])
    sampler=ScanSampler(data,seed=5); sampler.sample(3)
    saved=pickle.loads(pickle.dumps(sampler.state_dict()))
    other=ScanSampler(data,seed=8); other.load_state_dict(saved)
    for _ in range(4):
        assert_tree_equal(sampler.sample(3),other.sample(3))


def test_d4rl_reward_transform_matches_upstream_column_accumulation(tmp_path):
    import h5py
    from doublemdp.agents.bdpo_data import load_d4rl_hdf5, normalize_reward
    rng=np.random.RandomState(19)
    n=1500
    raw={'observations':rng.randn(n,3).astype('float32'),
         'actions':rng.randn(n,2).astype('float32'),
         'rewards':rng.randn(n).astype('float32'),
         'terminals':np.zeros(n,bool),'timeouts':np.zeros(n,bool)}
    raw['timeouts'][499::500]=True
    path=tmp_path/'fixture.hdf5'
    with h5py.File(path,'w') as stream:
        for key,value in raw.items():
            stream[key]=value
    processed=qlearning_dataset(types.SimpleNamespace(_max_episode_steps=1000),raw)
    expected=normalize_reward(processed['rewards'][:,None],processed['ends'],'iql_mujoco')[:,0]
    local,metadata=load_d4rl_hdf5(path)
    np.testing.assert_array_equal(local['rewards'],expected)
    np.testing.assert_array_equal(local['masks'],1-processed['terminals'])
    assert metadata['size']==1497


def test_explicit_stable_zero_rho_profile_at_degenerate_ensemble():
    values=jnp.ones((2,3,1,1))
    key=jax.random.PRNGKey(1)
    native=jax.grad(lambda x:get_target(key,x,False,'lcb',0.)[1].sum())(values)
    # Preserve and document the source behavior rather than silently changing it.
    assert not np.isfinite(native).all()
    stable=jax.grad(lambda x:get_target(key,x,False,'mean',0.)[1].sum())(values)
    np.testing.assert_array_equal(stable,np.full(values.shape,.5))
    assert_tree_equal(get_target(key,values,False,'lcb',0.)[1],
                      get_target(key,values,False,'mean',0.)[1])
