#!/usr/bin/env python
"""Download every upstream 100M shard and certify actual transition counts."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import urllib.request
import numpy as np
from download_dataset import download
from doublemdp.data import file_sha256
from doublemdp.logging import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("family",choices=["puzzle-4x4","cube-quadruple"])
p.add_argument("--dataset-dir",default=str(Path.home()/"doublemdp_outputs"/"datasets"/"ogbench"))
p.add_argument("--file-workers",type=int,default=2)
p.add_argument("--range-workers",type=int,default=4)
args=p.parse_args()
base=f"https://rail.eecs.berkeley.edu/datasets/ogbench/{args.family}-play-100m-v0/"
with urllib.request.urlopen(base,timeout=60) as response:
    names=sorted(set(re.findall(r'href="([^"/]+\.npz)"',response.read().decode())))
train=[n for n in names if not n.endswith('-val.npz')]
if len(train)!=100 or len(names)!=200:
    raise RuntimeError("Expected exactly100train+100validation shards; inspect changed upstream listing")
directory=Path(args.dataset_dir)/(args.family+'-play-100m-v0'); directory.mkdir(parents=True,exist_ok=True)
def fetch(name):
    download(base+name,directory/name,workers=args.range_workers)
    return name
with concurrent.futures.ThreadPoolExecutor(max_workers=args.file_workers) as pool:
    list(pool.map(fetch,names))
records=[]
for name in train:
    path=directory/name
    with np.load(path,allow_pickle=False) as z:
        terminals=z['terminals']
        transitions=int(np.sum(1-terminals))
    records.append({'path':str(path),'transitions':transitions,'sha256':file_sha256(path),'source':base+name})
total=sum(r['transitions'] for r in records)
if total<100_000_000:
    raise RuntimeError(f"Only{total}processed transitions; refusing100Mlabel")
manifest={'source':base,'total_transitions':total,'shards':[r['path'] for r in records],
          'records':records,'validation_shards':[str(directory/n) for n in names if n.endswith('-val.npz')],
          'sampling':'sorted cyclic rotation before every1000thupdate; source action clipping bypass'}
target=Path(args.dataset_dir)/(args.family+'-100M-shards.json')
write_json(target,manifest)
print(target)
