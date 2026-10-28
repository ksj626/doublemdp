"""Scientific identity and elapsed-time regressions for the original-task runner."""
from copy import deepcopy
import json
import pytest
from doublemdp.agents.bdpo_experiment import validate_resume, resume_elapsed


def record():
    config={'critic':{'eta':.06},'train_steps':5000}
    identity={'task':'8gaussians','seed':2,'dataset':{'sha256':'content-A','size':1000,'path':'old.npz'}}
    saved={'agent':{'config':config,'pretrain_counter':1000,'train_counter':5000},
           'experiment':identity,'elapsed_seconds':123.5}
    return saved,identity,config


def test_resume_rejects_changed_algorithm_task_dataset_and_seed_before_writes():
    saved,previous,cfg=record()
    requested=dict(requested_config=cfg,task='8gaussians',dataset={'sha256':'content-A','size':1000,'path':'moved.npz'},seed=2)
    validate_resume(saved,previous,**requested)
    for key,value,match in [('task','halfcheetah-medium-v2','task'),
            ('dataset',{'sha256':'content-B','size':1000},'dataset'),('seed',1,'seed')]:
        with pytest.raises(ValueError,match=match):
            validate_resume(saved,previous,**(requested|{key:value}))
    changed=deepcopy(cfg);changed['critic']['eta']=.05
    with pytest.raises(ValueError,match='configuration'):
        validate_resume(saved,previous,**(requested|{'requested_config':changed}))
    changed=deepcopy(cfg);changed['train_steps']=100000
    with pytest.raises(ValueError,match='budget'):
        validate_resume(saved,previous,**(requested|{'requested_config':changed}))
    # No explicit algorithm overrides means resume the checkpoint configuration.
    validate_resume(saved,previous,**(requested|{'requested_config':None}))
    assert saved['agent']['pretrain_counter']==1000
    assert saved['agent']['train_counter']==5000


def test_resume_preserves_elapsed_and_migrates_old_checkpoints_at_saved_counter(tmp_path):
    saved,previous,_=record()
    path=tmp_path/'metrics.jsonl'
    rows=[{'step':1000,'elapsed_seconds':20.}, {'step':6000,'elapsed_seconds':120.},
          {'step':7000,'elapsed_seconds':150.}]
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert resume_elapsed(saved,previous,path)==123.5
    del saved['elapsed_seconds']
    previous['elapsed_seconds']=150.
    assert resume_elapsed(saved,previous,path)==120.
    # Old runner checkpoints are acceptable with their original run identity.
    del saved['experiment']
    validate_resume(saved,previous,requested_config=None,task='8gaussians',dataset=previous['dataset'],seed=2)
    with pytest.raises(ValueError,match='identity'):
        validate_resume(saved,{},requested_config=None,task='8gaussians',dataset=previous['dataset'],seed=2)


def test_runner_phase_handoff_resume_accumulates_time_and_rejects_eta_relabel(tmp_path,monkeypatch):
    import pickle,sys
    import doublemdp.agents.bdpo_experiment as experiment
    from doublemdp.agents.bdpo import default_config
    cfg=default_config('smoke');cfg.update(pretrain_steps=2,train_steps=2,batch_size=8)
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(cfg))
    out=tmp_path/'run'
    base=['bdpo_experiment','--task','8gaussians','--config',str(config_path),
          '--out',str(out),'--data-size','32','--checkpoint-interval','1']
    monkeypatch.setattr(experiment,'evaluate_toy',lambda *args,**kwargs:{'test_evaluation':True})
    monkeypatch.setattr(sys,'argv',base+['--max-seconds','0.00000001'])
    experiment.main()
    with (out/'checkpoint.pkl').open('rb') as stream:first=pickle.load(stream)
    assert first['agent']['pretrain_counter']==1
    assert first['agent']['train_counter']==0
    assert first['elapsed_seconds']>0
    monkeypatch.setattr(sys,'argv',base+['--resume'])
    experiment.main()
    with (out/'checkpoint.pkl').open('rb') as stream:resumed=pickle.load(stream)
    report=json.loads((out/'run.json').read_text())
    assert report['status']=='complete'
    assert resumed['agent']['pretrain_counter']==2
    assert resumed['agent']['train_counter']==2
    assert resumed['resume_count']==1
    assert resumed['elapsed_seconds']>first['elapsed_seconds']
    assert report['elapsed_seconds']>=resumed['elapsed_seconds']
    before=(out/'run.json').read_bytes()
    cfg['critic']['eta']+=.01;config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys,'argv',base+['--resume'])
    with pytest.raises(ValueError,match='configuration'):
        experiment.main()
    assert (out/'run.json').read_bytes()==before
