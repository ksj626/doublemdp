#!/usr/bin/env python
"""Fetch standard benchmark datasets from the frozen task manifest."""
import argparse
import concurrent.futures
import json
from pathlib import Path
from download_dataset import download

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest",default="configs/rql_ogbench_manifest.json")
p.add_argument("--dataset-dir",default=str(Path.home()/"doublemdp_outputs"/"datasets"/"ogbench"))
args=p.parse_args()
manifest=json.loads(Path(args.manifest).read_text())
urls=set()
for task in manifest["tasks"]:
    data=task["dataset"]
    if not data.get("shard_manifest_required"):
        urls.update([data["source"],data["validation_source"]])
def fetch(url):
    download(url,Path(args.dataset_dir)/url.rsplit('/',1)[1],workers=4)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(fetch,sorted(urls)))
