"""Scientific identity, resumability and self-contained artifact regressions."""
import hashlib
import json
from pathlib import Path
import random
import signal

import numpy as np
import pytest

from doublemdp.data import OfflineDataset, save_dataset
from doublemdp.logging import save_checkpoint, load_checkpoint


def arrays():
    return dict(observations=np.zeros((3,2),dtype='f'),next_observations=np.ones((3,2),dtype='f'),
                actions=np.zeros((3,2),dtype='f'),rewards=np.zeros(3,dtype='f'),
                terminals=np.array([0,0,1],dtype='f'),masks=np.ones(3,dtype='f'))


def test_sampler_rejects_semantic_metadata_change_with_identical_arrays():
    source=OfflineDataset(arrays(),metadata={'kind':'bandit','gap':False})
    modified=OfflineDataset(arrays(),metadata={'kind':'bandit','gap':True})
    assert source.digest==modified.digest
    with pytest.raises(ValueError,match='semantic metadata'):
        modified.load_state_dict(source.state_dict())
    # Historical checkpoints lacked this digest; backward compatibility is
    # explicit, with no claim that their old metadata can be reconstructed.
    legacy=source.state_dict();legacy.pop('metadata_sha256')
    source.load_state_dict(legacy)


def test_checkpoint_hash_tracks_open_snapshot_during_atomic_replacement(tmp_path,monkeypatch):
    from doublemdp import logging
    path=tmp_path/'checkpoint.pkl'
    save_checkpoint(path,{'step':1})
    old_digest=hashlib.sha256(path.read_bytes()).hexdigest()
    real_load=logging.pickle.load
    def replace_during_load(stream):
        state=real_load(stream)
        save_checkpoint(path,{'step':2})
        return state
    monkeypatch.setattr(logging.pickle,'load',replace_during_load)
    state,identity=load_checkpoint(path,with_identity=True)
    assert state['step']==1
    assert identity['sha256']==old_digest
    assert identity['sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('change',['arrays','metadata'])
def test_evaluation_rejects_changed_dataset_identity(tmp_path,monkeypatch,change):
    from doublemdp.evaluate import load_run
    from doublemdp import train
    path=tmp_path/'data.npz'
    save_dataset(path,arrays(),{'kind':'bandit','gap':False})
    original=OfflineDataset.load(path)
    checkpoint=tmp_path/'checkpoint.pkl'
    save_checkpoint(checkpoint,{'config':{'dataset':str(path),'method':'rql'},'sampler':original.state_dict(),'agent':{},'step':1})
    monkeypatch.setattr(train,'restore_agent',lambda *a:object())
    if change=='arrays':
        changed=arrays();changed['actions'][0]=.3
        save_dataset(path,changed,{'kind':'bandit','gap':False})
    else:
        sidecar=json.loads(path.with_suffix('.json').read_text());sidecar['gap']=True
        path.with_suffix('.json').write_text(json.dumps(sidecar))
    with pytest.raises(ValueError,match='differs from checkpoint'):
        load_run(checkpoint)


def test_maze_figures_regenerate_without_external_dataset(tmp_path):
    from doublemdp.visualize import render
    np.savez(tmp_path/'plot_inputs.npz',probe_states=np.array([[.2,.5]]),
             probe0_latents=np.zeros((2,2,2)),probe0_progress=np.array([0.,1.]),
             probe0_native_times=np.array([0.,1.]),probe0_final_actions=np.zeros((2,1,2)),
             oracle_values=np.zeros((2,2)),dataset_coverage_observations=np.array([[.2,.5],[.3,.5]]))
    meta={'env':'maze','method':'rql','profile':'toy_diagnostic','training_step':2,
          'dataset':str(tmp_path/'unavailable.npz'),'native_value_semantics':'Not an MC value claim'}
    (tmp_path/'plot_metadata.json').write_text(json.dumps(meta))
    render(tmp_path,movie=False)
    assert (tmp_path/'atlas_probe0.png').exists()
    assert (tmp_path/'index.html').exists()


def test_shard_resume_rejects_change_in_inactive_shard_identity(tmp_path):
    from doublemdp.envs import ShardedOGBench
    shard=ShardedOGBench.__new__(ShardedOGBench)
    shard.paths=[tmp_path/'active.npz',tmp_path/'inactive.npz']
    shard.shards_digest='new full-shard identity'
    shard._load=lambda *_:pytest.fail('Identity must be rejected before loading active shard')
    state={'paths':[str(p) for p in shard.paths],'index':0,'dataset':{},'shards_sha256':'old full-shard identity'}
    with pytest.raises(ValueError,match='inactive shard'):
        shard.load_state_dict(state)


def test_queue_rejects_conflicting_config_and_output_reuse(tmp_path):
    from doublemdp.sweep import connect,enqueue
    db=connect(tmp_path/'queue.db')
    cfg={'output':str(tmp_path/'run'),'agent':{'discount':.99}}
    enqueue(db,'run',cfg)
    with pytest.raises(ValueError,match='frozen configuration'):
        enqueue(db,'run',dict(cfg,agent={'discount':.9}))
    with pytest.raises(ValueError,match='Output directory'):
        enqueue(db,'different',cfg)
    enqueue(db,'next',{'output':str(tmp_path/'next')})
    assert db.execute('SELECT count(*) FROM jobs').fetchone()[0]==2


def test_queue_resume_uses_recorded_interpreter_and_environment(tmp_path):
    from doublemdp.sweep import connect,enqueue,status
    db=connect(tmp_path/'queue.db');output=tmp_path/'run';output.mkdir()
    enqueue(db,'run',{'output':str(output)})
    assert status(db)[0]['resume_command'] is None
    (output/'checkpoint.pkl').write_bytes(b'existence only')
    recorded={'argv':['/native/env/bin/python','-m','doublemdp.train','--config','run.json'],
              'environment':{'CUDA_VISIBLE_DEVICES':'1'},'cwd':'/native/workspace'}
    (output/'command.json').write_text(json.dumps(recorded))
    record=status(db)[0]
    assert record['resume_command']==recorded['argv']+['--resume']
    assert record['resume_environment']==recorded['environment']
    assert record['resume_cwd']==recorded['cwd']


def test_failed_budget_extension_clears_stale_completion_and_restores_signals(tmp_path,monkeypatch):
    from doublemdp import train
    path=tmp_path/'data.npz';save_dataset(path,arrays(),{'kind':'maze','gamma':.99})
    dataset=OfflineDataset.load(path)
    config={'method':'rql','profile':'toy_diagnostic','strict_offline':True,'env':'maze',
            'dataset':str(path),'output':str(tmp_path/'run'),'seed':0,'batch_size':2,'steps':2,'agent':{}}
    output=Path(config['output']);output.mkdir()
    save_checkpoint(output/'checkpoint.pkl',{'config':config,'agent':{},'sampler':dataset.state_dict(),
                    'step':2,'elapsed_seconds':1.,'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
    (output/'status.json').write_text(json.dumps({'status':'complete','step':2,'total_steps':2}))
    class BrokenAgent:
        config={'h':1}
        def update(self,batch):raise RuntimeError('injected update failure')
    monkeypatch.setattr(train,'restore_agent',lambda *a:BrokenAgent())
    old_handlers={s:signal.getsignal(s) for s in [signal.SIGTERM,signal.SIGINT]}
    with pytest.raises(RuntimeError,match='injected'):
        train.run(dict(config,steps=4),str(output/'checkpoint.pkl'))
    status=json.loads((output/'status.json').read_text())
    assert (status['status'],status['step'],status['total_steps'])==('running',2,4)
    assert all(signal.getsignal(s)==h for s,h in old_handlers.items())
    with pytest.raises(ValueError,match='below the restored checkpoint'):
        train.run(dict(config,steps=1),str(output/'checkpoint.pkl'))


def test_100m_rejects_repeated_file_substitution(tmp_path):
    from doublemdp.envs import ShardedOGBench
    path=tmp_path/'one.npz';np.savez(path,terminals=np.array([0,1]))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'total_transitions':100_000_000,'shards':[path.name]*100}))
    with pytest.raises(ValueError,match='100 distinct'):
        ShardedOGBench('puzzle-4x4-play-singletask-task1-v0',manifest)


@pytest.mark.parametrize('problem',['count','duplicate_contents','total'])
def test_100m_checks_actual_contents_and_counts(tmp_path,problem):
    from doublemdp.envs import ShardedOGBench
    paths=[];records=[]
    for i in range(100):
        path=tmp_path/f'{i:03d}.npz'
        np.savez(path,terminals=np.array([0,0,1],dtype='f'),unique=np.array(0 if problem=='duplicate_contents' else i))
        path.with_name(path.stem+'-val.npz').write_bytes(b'partner existence')
        paths.append(path.name)
        records.append({'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'transitions':1_000_000 if problem=='count' else 2})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'total_transitions':100_000_000,'shards':paths,'records':records}))
    expected={'count':'transition count differs','duplicate_contents':'Duplicate shard contents','total':'Verified processed transition count'}[problem]
    with pytest.raises(ValueError,match=expected):
        ShardedOGBench('puzzle-4x4-play-singletask-task1-v0',manifest)


def test_dense_shaping_discount_guard_precedes_agent_creation(tmp_path,monkeypatch):
    from doublemdp import train
    data=arrays();data['rewards_dense']=np.zeros(3,dtype='f')
    path=tmp_path/'dense.npz';save_dataset(path,data,{'kind':'maze','gamma':.99})
    monkeypatch.setattr(train,'create_agent',lambda *_:pytest.fail('Mismatched shaping must fail before training'))
    config={'method':'rql','profile':'toy_diagnostic','strict_offline':True,'env':'maze','reward':'dense',
            'dataset':str(path),'output':str(tmp_path/'run'),'steps':2,'agent':{'discount':.95}}
    with pytest.raises(ValueError,match='gamma must equal'):
        train.run(config)


def test_download_resume_checks_existing_frozen_bytes(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('dataset_downloader',Path(__file__).resolve().parents[1]/'scripts/download_dataset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'data.npz';path.write_bytes(b'original')
    record={'url':'https://example.test/data.npz','size':8,'sha256':hashlib.sha256(b'original').hexdigest()}
    path.with_suffix('.download.json').write_text(json.dumps(record))
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *a,**kw:pytest.fail('Valid existing file needs no network'))
    module.download(record['url'],path)
    path.write_bytes(b'modified')
    with pytest.raises(ValueError,match='frozen download identity'):
        module.download(record['url'],path)


def test_toy_sequence_extension_cannot_change_ogbench_profile():
    from doublemdp.train import validate_config
    with pytest.raises(ValueError,match='restricted'):
        validate_config({'method':'rql','profile':'official_code',
                         'env':'antmaze-large-navigate-singletask-task1-v0','rql_sequence_adapter':'transition_h1'})
    validate_config({'method':'rql','profile':'toy_diagnostic','env':'maze','rql_sequence_adapter':'transition_h1'})
