#!/usr/bin/env python
"""Run held-out original D4RL evaluation only after its full training budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from doublemdp.logging import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',required=True,type=Path)
p.add_argument('--gpu',default='6')
args=p.parse_args()
status_path=args.run/'postprocess.json'
while True:
    try:
        metadata=json.loads((args.run/'run.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        time.sleep(30);continue
    status=metadata.get('status')
    if status=='failed':
        write_json(status_path,{'status':'blocked','reason':'Training failed; no final-score evaluation claimed','pid':os.getpid()});break
    if status=='complete':
        cfg=metadata['config']
        if metadata['completed_pretrain_steps']!=cfg['pretrain_steps'] or metadata['completed_train_steps']!=cfg['train_steps']:
            raise RuntimeError('Completion flag disagrees with source budget counters')
        command=[sys.executable,'-m','doublemdp.agents.bdpo_d4rl_evaluate','--checkpoint',str(args.run/'checkpoint.pkl'),
                 '--task',metadata['task'],'--episodes','10','--seed',str(metadata['seed']),
                 '--output',str(args.run/'original_evaluation.json')]
        write_json(status_path,{'status':'running_evaluation','command':command,'pid':os.getpid()})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,XLA_PYTHON_CLIENT_PREALLOCATE='false')
        with (args.run/'postprocess.log').open('a') as log:
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).resolve().parents[1])
        write_json(status_path,{'status':'complete' if result.returncode==0 else 'failed','returncode':result.returncode,'command':command})
        break
    write_json(status_path,{'status':'waiting_for_training','pretrain_steps':metadata.get('completed_pretrain_steps',0),
        'train_steps':metadata.get('completed_train_steps',0),'pid':os.getpid(),'updated_unix':time.time()})
    time.sleep(30)
