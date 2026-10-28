#!/usr/bin/env python
"""Summarize completed fixed final checkpoints; do not select best evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
from doublemdp.diagnostics import aggregate_training_seeds
from doublemdp.logging import write_json
from doublemdp.sweep import connect,status

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--output-root",default=str(Path.home()/"doublemdp_outputs"))
args=p.parse_args(); root=Path(args.output_root)
groups={}; excluded=[]; completed=[]
for run in sorted((root/'runs').iterdir()):
    status_path=run/'status.json'; config_path=run/'config.json'
    if not status_path.exists() or not config_path.exists():
        continue
    state=json.loads(status_path.read_text()); config=json.loads(config_path.read_text())
    if state.get('status')!='complete' or 'method' not in config or state.get('step') != config.get('steps'):
        continue
    if (run/'SCIENTIFIC_LIMITATIONS.json').exists():
        excluded.append({'run':str(run),'limitation':json.loads((run/'SCIENTIFIC_LIMITATIONS.json').read_text())});continue
    evaluation=run/f"evaluation_{state['step']}.json"
    if not evaluation.exists():continue
    metrics=json.loads(evaluation.read_text())
    dataset_meta=json.loads((run/'dataset.json').read_text())
    algorithm=json.loads((run/'resolved_agent.json').read_text()) if (run/'resolved_agent.json').exists() else config.get('agent',{})
    algorithm_fingerprint=hashlib.sha256(json.dumps(algorithm,sort_keys=True).encode()).hexdigest()
    record={'run':str(run),'method':config['method'],'profile':config['profile'],'env':config['env'],
            'seed':config['seed'],'steps':state['step'],'seconds':state['elapsed_seconds'],'metrics':metrics,
            'reward':config.get('reward','sparse'),'dataset':config.get('dataset'),
            'adapter':config.get('rql_sequence_adapter','official'),'online_interactions':0,
            'dataset_arrays_sha256':dataset_meta['arrays_sha256'],'algorithm_fingerprint':algorithm_fingerprint,
            'evaluation_policy':{'seed':config.get('evaluation_seed',1729),'episodes':config.get('eval_episodes',50),
                                 'samples':config.get('eval_samples',2048),'candidates':config.get('agent',{}).get('eval_num_samples',1)}}
    completed.append(record)
    # Include the dataset/adapter/budget/reward definition in each comparison.
    key=json.dumps({k:record[k] for k in ('method','profile','env','steps','reward','dataset_arrays_sha256','adapter','algorithm_fingerprint','evaluation_policy')},sort_keys=True)
    metric='reward_mean' if config['env']=='bandit' else 'success'
    if metric in metrics:
        groups.setdefault(key,[]).append({'seed':config['seed'],'score':metrics[metric]})
aggregates=[dict(json.loads(key),**aggregate_training_seeds(rows)) for key,rows in groups.items()]
queues={}
for path in root.glob('*.sqlite'):
    queues[path.name]=status(connect(path))
write_json(root/'summary.json',{'completed_offline_runs':completed,'training_seed_aggregates':aggregates,
    'excluded_runs':excluded,'queues':queues,'online_results':'See DPPO completed.json and figures/dppo/summary.json; kept outside offline comparison'})
print(json.dumps({'completed_offline_runs':len(completed),'groups':len(aggregates),'excluded':len(excluded),'summary':str(root/'summary.json')},indent=2))
