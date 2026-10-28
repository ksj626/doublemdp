#!/usr/bin/env python
"""Resumable bounded-concurrency HTTP range download with content validation."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request


def download(url, destination, workers=8, chunk_bytes=8<<20):
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        provenance=destination.with_suffix(".download.json")
        if provenance.exists():
            recorded=json.loads(provenance.read_text())
            digest=hashlib.sha256()
            with destination.open("rb") as stream:
                for block in iter(lambda:stream.read(8<<20),b""):
                    digest.update(block)
            if (recorded["url"]!=url or recorded["size"]!=destination.stat().st_size
                    or recorded["sha256"]!=digest.hexdigest()):
                raise ValueError("Existing dataset differs from its frozen download identity")
        print(f"Already exists: {destination}",flush=True)
        return
    with urllib.request.urlopen(urllib.request.Request(url,method="HEAD"),timeout=60) as response:
        length=int(response.headers["Content-Length"])
        etag=response.headers.get("ETag")
    parts=destination.with_suffix(destination.suffix+".parts"); parts.mkdir(exist_ok=True)
    identity={"url":url,"size":length,"etag":etag}
    record=parts/"source.json"
    if record.exists() and json.loads(record.read_text())!=identity:
        raise ValueError("Remote dataset changed during download; preserve old parts and investigate")
    record.write_text(json.dumps(identity,indent=2))
    def fetch(start):
        end=min(length-1,start+chunk_bytes-1)
        path=parts/str(start)
        if path.exists() and path.stat().st_size==end-start+1:
            return path
        for retry in range(5):
            try:
                headers={"Range":f"bytes={start}-{end}"}
                if etag:
                    headers["If-Range"]=etag
                request=urllib.request.Request(url,headers=headers)
                with urllib.request.urlopen(request,timeout=120) as response:
                    expected=f"bytes {start}-{end}/{length}"
                    if response.status!=206 or response.headers.get("Content-Range")!=expected:
                        raise RuntimeError(f"Server did not honor range: {response.status}, {response.headers}")
                    with path.with_suffix(".tmp").open("wb") as target:
                        shutil.copyfileobj(response,target,1<<20)
                if path.with_suffix(".tmp").stat().st_size != end-start+1:
                    raise IOError("Incomplete range")
                os.replace(path.with_suffix(".tmp"),path)
                print(f"{destination.name}: completed bytes {start}-{end} of {length}",flush=True)
                return path
            except Exception:
                if retry==4:
                    raise
                time.sleep(2**retry)
    starts=list(range(0,length,chunk_bytes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch,starts))
    temporary=destination.with_suffix(destination.suffix+".assembling")
    digest=hashlib.sha256()
    with temporary.open("wb") as stream:
        for start in starts:
            with (parts/str(start)).open("rb") as source:
                for block in iter(lambda:source.read(8<<20),b""):
                    digest.update(block); stream.write(block)
        stream.flush(); os.fsync(stream.fileno())
    assert temporary.stat().st_size==length
    os.replace(temporary,destination)
    destination.with_suffix(".download.json").write_text(json.dumps(identity|{"sha256":digest.hexdigest()},indent=2)+"\n")
    shutil.rmtree(parts)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("url"); p.add_argument("destination"); p.add_argument("--workers",type=int,default=8)
    args=p.parse_args(); download(args.url,args.destination,args.workers)
