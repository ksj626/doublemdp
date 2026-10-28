"""Readiness, orphan recovery and lease tests use isolated queues only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pytest

from doublemdp.logging import save_checkpoint
from doublemdp.readiness import BASE, DatasetReadiness
from doublemdp.sweep import (connect, enqueue, exclusive_lock, recover_orphans,
                            refresh_readiness)


def archive(path):
    np.savez(path,observations=np.zeros((3,1)),actions=np.zeros((3,1)),terminals=np.array([0,0,1]))
    path.with_suffix('.download.json').write_text(json.dumps({
        'url':BASE+path.name,'size':path.stat().st_size,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))


def standard_config(tmp_path):
    return {'env':'scene-play-singletask-task2-v0','dataset_dir':str(tmp_path),
            'output':str(tmp_path/'run'),'steps':10}


def test_standard_readiness_checks_both_files_and_invalidates_cached_bytes(tmp_path,monkeypatch):
    check=DatasetReadiness();cfg=standard_config(tmp_path)
    assert check(cfg).startswith('Dataset not ready:')
    train=tmp_path/'scene-play-v0.npz';val=tmp_path/'scene-play-v0-val.npz'
    archive(train)
    assert check(cfg) is not None
    archive(val)
    assert check(cfg) is None
    # Explicitly simulate a coarse filesystem clock for this same-length write.
    from doublemdp import readiness
    stamps={p:readiness.signature(p) for p in [train,val,train.with_suffix('.download.json'),val.with_suffix('.download.json')]}
    monkeypatch.setattr(readiness,'signature',lambda path:stamps[Path(path)])
    recent=max(max(stamp[3],stamp[4]) for stamp in stamps.values())/1e9+.5
    monkeypatch.setattr(readiness.time,'time',lambda:recent)
    raw=train.read_bytes();train.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    assert 'checksum mismatch' in check(cfg)


def test_quiescent_cache_avoids_rehashing_but_expires(tmp_path,monkeypatch):
    from doublemdp import readiness
    cfg=standard_config(tmp_path)
    for name in ['scene-play-v0.npz','scene-play-v0-val.npz']:archive(tmp_path/name)
    wall=time.time();mono=time.monotonic()
    monkeypatch.setattr(readiness.time,'time',lambda:wall+10)
    monkeypatch.setattr(readiness.time,'monotonic',lambda:mono)
    check=DatasetReadiness();assert check(cfg) is None
    real_load=np.load
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('quiescent unchanged bytes were reloaded before expiry'))
    assert check(cfg) is None
    monkeypatch.setattr(np,'load',real_load)
    monkeypatch.setattr(readiness.time,'monotonic',lambda:mono+readiness.MAX_CACHE_AGE_SECONDS+1)
    calls=[]
    def reload(*args,**kwargs):
        calls.append(args[0]);return real_load(*args,**kwargs)
    monkeypatch.setattr(np,'load',reload)
    assert check(cfg) is None
    assert len(calls)==2


def test_readiness_does_not_release_unidentified_or_wrong_source_files(tmp_path):
    cfg=standard_config(tmp_path)
    for name in ['scene-play-v0.npz','scene-play-v0-val.npz']:archive(tmp_path/name)
    side=tmp_path/'scene-play-v0.download.json';record=json.loads(side.read_text())
    record['url']='https://different.test/data.npz';side.write_text(json.dumps(record))
    assert 'identity/size mismatch' in DatasetReadiness()(cfg)
    side.unlink()
    assert DatasetReadiness()(cfg) is not None


def test_readiness_cas_never_modifies_simultaneously_claimed_job(tmp_path):
    db=connect(tmp_path/'queue.db');enqueue(db,'a',standard_config(tmp_path))
    def claim_during_check(config):
        db.execute("UPDATE jobs SET status='running',pid=? WHERE id='a'",(os.getpid(),));db.commit()
        return 'missing bytes'
    refresh_readiness(db,claim_during_check)
    assert db.execute("SELECT status FROM jobs WHERE id='a'").fetchone()[0]=='running'


def test_readiness_blocks_and_releases_only_verified_data(tmp_path):
    db=connect(tmp_path/'queue.db');enqueue(db,'a',standard_config(tmp_path));check=DatasetReadiness()
    refresh_readiness(db,check)
    assert db.execute('SELECT status FROM jobs').fetchone()[0]=='blocked'
    for name in ['scene-play-v0.npz','scene-play-v0-val.npz']:archive(tmp_path/name)
    refresh_readiness(db,check)
    assert db.execute('SELECT status,error FROM jobs').fetchone()[:]==('pending',None)


def test_worker_never_starts_upstream_downloader_for_missing_standard_data(tmp_path,monkeypatch):
    from doublemdp import sweep
    db=connect(tmp_path/'queue.db');enqueue(db,'a',standard_config(tmp_path))
    monkeypatch.setattr(sweep.signal,'signal',lambda *args:None)
    monkeypatch.setattr(sweep.subprocess,'Popen',lambda *a,**kw:pytest.fail('Missing data must block before trainer launch'))
    assert sweep.worker(db,'5','python')==0
    row=db.execute('SELECT status,error,pid FROM jobs').fetchone()
    assert row['status']=='blocked' and row['pid'] is None
    assert 'Dataset not ready' in row['error']


def orphan(db,tmp_path,name,pid=99999999,age=120,checkpoint=True):
    config={'output':str(tmp_path/name),'steps':10,'seed':0}
    enqueue(db,name,config)
    db.execute("UPDATE jobs SET status='running',pid=?,gpu='1',started=? WHERE id=?",(pid,time.time()-age,name));db.commit()
    if checkpoint:
        save_checkpoint(Path(config['output'])/'checkpoint.pkl',{
            'config':config,'agent':{'params':'saved'},'sampler':{'rng':'saved'},'step':5,
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'elapsed_seconds':2.})
    return config


def test_orphan_recovery_requires_dead_process_and_complete_matching_checkpoint(tmp_path):
    db=connect(tmp_path/'queue.db')
    orphan(db,tmp_path,'valid');orphan(db,tmp_path,'missing',checkpoint=False)
    invalid=orphan(db,tmp_path,'corrupt');(Path(invalid['output'])/'checkpoint.pkl').write_bytes(b'broken')
    mismatched=orphan(db,tmp_path,'mismatch');path=Path(mismatched['output'])/'checkpoint.pkl'
    import pickle
    state=pickle.loads(path.read_bytes());state['config']['seed']=77;save_checkpoint(path,state)
    orphan(db,tmp_path,'live',pid=os.getpid())
    orphan(db,tmp_path,'fresh',pid=None,age=0)
    report=recover_orphans(db)
    assert report['recovered']==['valid']
    assert set(report['unrecoverable'])=={'missing','corrupt','mismatch'}
    assert {r['id']:r['status'] for r in db.execute('SELECT id,status FROM jobs')}=={
        'valid':'pending','missing':'failed','corrupt':'failed','mismatch':'failed','live':'running','fresh':'running'}
    for row in db.execute("SELECT error,pid FROM jobs WHERE status='failed'"):
        assert row['error'].startswith('Manual retry required: orphan checkpoint')
        assert row['pid'] is None
    refresh_readiness(db,lambda config:None)
    assert db.execute("SELECT count(*) FROM jobs WHERE status='failed'").fetchone()[0]==3


def test_orphan_failure_cas_preserves_replaced_live_process(tmp_path,monkeypatch):
    from doublemdp import logging
    db=connect(tmp_path/'queue.db');orphan(db,tmp_path,'changed')
    def replaced_during_load(path):
        db.execute("UPDATE jobs SET pid=?,started=? WHERE id='changed'",(os.getpid(),time.time()));db.commit()
        raise ValueError('old checkpoint was invalid')
    monkeypatch.setattr(logging,'load_checkpoint',replaced_during_load)
    assert recover_orphans(db)=={'recovered':[],'unrecoverable':{}}
    row=db.execute("SELECT status,pid FROM jobs WHERE id='changed'").fetchone()
    assert (row['status'],row['pid'])==('running',os.getpid())


def test_data_readiness_preserves_manual_block(tmp_path):
    db=connect(tmp_path/'queue.db');enqueue(db,'a',standard_config(tmp_path))
    db.execute("UPDATE jobs SET status='blocked',error='Manual intervention required'");db.commit()
    refresh_readiness(db,lambda config:pytest.fail('Manual block is outside data readiness'))
    row=db.execute('SELECT status,error FROM jobs').fetchone()
    assert tuple(row)==('blocked','Manual intervention required')


def test_live_worker_lease_protects_exit_recording_window(tmp_path):
    db=connect(tmp_path/'queue.db');orphan(db,tmp_path,'valid')
    assert recover_orphans(db,protected_gpus={'1'})['recovered']==[]
    assert db.execute('SELECT status FROM jobs').fetchone()[0]=='running'


def test_exclusive_leases_prevent_duplicate_supervisors_and_workers(tmp_path):
    path=tmp_path/'queue.supervisor.lock';first=exclusive_lock(path)
    assert first is not None
    assert exclusive_lock(path) is None
    first.close()
    second=exclusive_lock(path);assert second is not None;second.close()


def test_large_manifest_is_not_released_from_existence_alone(tmp_path,monkeypatch):
    manifest=tmp_path/'shards.json';config={'env':'puzzle-4x4-play-singletask-task1-v0','shard_manifest':str(manifest)}
    check=DatasetReadiness();manifest.write_text('{')
    assert check(config) is not None
    paths=[str(tmp_path/f'{i:03d}.npz') for i in range(100)]
    records=[{'path':p,'sha256':f'{i:064x}','transitions':1_000_000} for i,p in enumerate(paths)]
    data={'source':BASE+'puzzle-4x4-play-100m-v0/','total_transitions':100_000_000,
          'shards':paths,'records':records,'validation_shards':[p.replace('.npz','-val.npz') for p in paths]}
    manifest.write_text(json.dumps(data))
    calls=[]
    def checked(path,**kw):
        calls.append(path)
        return (kw.get('sha256','validation'),kw.get('transitions',100_000))
    monkeypatch.setattr(check,'_file',checked)
    assert check(config) is None
    assert len(calls)==200
    data['validation_shards'].pop();manifest.write_text(json.dumps(data))
    assert 'every validation partner' in check(config)
    data['validation_shards']=[p.replace('.npz','-val.npz') for p in paths]
    data['total_transitions']+=1;manifest.write_text(json.dumps(data))
    assert 'Actual shard total differs' in check(config)


def test_supervisor_module_import_is_side_effect_free_and_checks_worker_identity(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('supervisor',Path(__file__).resolve().parents[1]/'scripts/continue_reproduction.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'process_alive',lambda pid:True)
    queue=tmp_path/'queue.db'
    command=['python','-m','doublemdp.sweep','work','--queue',str(queue),'--gpu','5','']
    monkeypatch.setattr(Path,'read_bytes',lambda p:'\0'.join(command).encode())
    assert module.worker_matches(123,queue,'5')
    assert not module.worker_matches(123,queue,'6')
    assert not module.worker_matches(123,tmp_path/'different.db','5')
    command[2]='unrelated.module'
    assert not module.worker_matches(123,queue,'5')
