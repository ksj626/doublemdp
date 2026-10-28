#!/usr/bin/env python
"""Disaggregate fixed final-policy bandit evaluations and saved behavior data."""
import argparse
import json
from pathlib import Path
import numpy as np
from doublemdp.evaluate import load_run,evaluate_bandit
from doublemdp.data import OfflineDataset
from doublemdp.logging import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-root',type=Path,default=Path.home()/'doublemdp_outputs')
args=p.parse_args();root=args.output_root
results=[]
for method in ('rql','bdpo'):
    for condition in ('','_gap'):
        for seed in range(3):
            run=root/'runs'/f'{method}_bandit{condition}_seed{seed}'
            agent,config,metadata,checkpoint=load_run(run)
            metrics=evaluate_bandit(agent,config,metadata)
            ds=OfflineDataset.load(config['dataset'])
            contexts=ds.arrays['observations'].argmax(-1)
            baseline={f'context_{c}_saved_behavior_reward_mean':float(ds.arrays['rewards'][contexts==c].mean()) for c in range(3)}
            record={'method':method,'condition':condition or 'main','seed':seed,'checkpoint':str(run/'checkpoint.pkl'),
                    'step':checkpoint['step'],'metrics':metrics,'saved_behavior_baseline':baseline,'reward_access':'evaluation oracle only'}
            write_json(run/'context_evaluation.json',record);results.append(record)
write_json(root/'figures/comparison/bandit_contexts.json',results)
print(root/'figures/comparison/bandit_contexts.json')
