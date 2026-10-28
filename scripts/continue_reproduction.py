#!/usr/bin/env python
"""Gate the remaining sweep on completed representatives; supervise idle GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from doublemdp.logging import write_json
from doublemdp.readiness import DatasetReadiness
from doublemdp.sweep import (connect, exclusive_lock, process_alive, status,
                            recover_orphans, refresh_readiness)


def worker_matches(pid, queue, gpu):
    """Adopt only a live worker for this exact queue and physical GPU."""
    if not process_alive(pid):
        return False
    try:
        argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        module=argv.index('-m')
        if argv[module+1:module+3] != ['doublemdp.sweep','work']:
            return False
        found_queue=Path(argv[argv.index('--queue')+1])
        if not found_queue.is_absolute():
            found_queue=Path(f'/proc/{pid}/cwd').resolve()/found_queue
        return found_queue.resolve()==Path(queue).resolve() and argv[argv.index('--gpu')+1]==str(gpu)
    except (OSError, ValueError, IndexError):
        return False


def idle(gpu):
    try:
        result=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        return result.returncode==0 and not result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',required=True)
    parser.add_argument('--gpus',default='0,1,3,4,5,7',help='GPU2/6 are reserved for the original BDPO runs')
    parser.add_argument('--output',required=True)
    parser.add_argument('--python',default=sys.executable)
    args=parser.parse_args(argv)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    queue=Path(args.queue).resolve()
    lease=exclusive_lock(queue.with_name(queue.name+'.supervisor.lock'))
    if lease is None:
        raise SystemExit('Another supervisor owns this queue')
    snapshot=output/'supervisor.json'
    previous=json.loads(snapshot.read_text()) if snapshot.exists() else {}
    # Also detect the older supervisor version which did not acquire a lease.
    if previous.get('pid')!=os.getpid() and process_alive(previous.get('pid')):
        lease.close()
        raise SystemExit('Recorded supervisor is still alive; refusing duplicate workers')
    db=connect(queue); workers={}; children={}; streams={}
    for gpu,pid in previous.get('workers',{}).items():
        if worker_matches(pid,queue,gpu):
            workers[gpu]=pid
    stopping=False; detach=False
    def stop(sig,frame):
        nonlocal stopping,detach
        stopping=True; detach=sig==signal.SIGHUP
        if not detach:
            for gpu,pid in list(workers.items()):
                if worker_matches(pid,queue,gpu):
                    try: os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError: pass
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,stop)
    representatives=['rql_antmaze-large-navigate-singletask-task1-v0_official_code_seed0',
                     'rql_scene-play-singletask-task2-v0_official_code_seed0']
    checker=DatasetReadiness()
    try:
        while not stopping:
            for gpu,pid in list(workers.items()):
                if gpu in children:
                    children[gpu].poll()  # Reap owned zombies before /proc checks.
                if not worker_matches(pid,queue,gpu):
                    workers.pop(gpu); children.pop(gpu,None)
                    if gpu in streams: streams.pop(gpu).close()
            recovery=recover_orphans(db,protected_gpus=set(workers))
            # Existing workers are preserved. Only pending/blocked rows may
            # change, and compare-and-swap protects a simultaneous live claim.
            refresh_readiness(db,checker)
            rows={r['id']:dict(r) for r in db.execute('SELECT id,status FROM jobs WHERE id IN (?,?)',representatives)}
            ready=all(rows.get(name,{}).get('status')=='complete' for name in representatives)
            live_gpus={str(r['gpu']) for r in db.execute("SELECT gpu,pid FROM jobs WHERE status='running'")
                       if process_alive(r['pid'])}
            if ready and not stopping:
                for gpu in dict.fromkeys(args.gpus.split(',')):
                    pending=db.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
                    if pending and gpu not in workers and gpu not in live_gpus and idle(gpu):
                        stream=(output/f'worker_gpu{gpu}.log').open('a'); streams[gpu]=stream
                        command=[args.python,'-m','doublemdp.sweep','work','--queue',str(queue),'--gpu',gpu]
                        child=subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],stdout=stream,stderr=subprocess.STDOUT)
                        children[gpu]=child; workers[gpu]=child.pid
            counts={r[0]:r[1] for r in db.execute('SELECT status,count(*) FROM jobs GROUP BY status')}
            write_json(snapshot,{'pid':os.getpid(),'representatives_complete':ready,'counts':counts,
                'workers':workers,'updated_unix':time.time(),'orphan_recovery':recovery,
                'resume_command':[args.python,str(Path(__file__).resolve()),'--queue',str(queue),
                                  '--gpus',args.gpus,'--output',str(output)],
                'policy':'Final source-budget representatives unlock remaining jobs. Dataset bytes are verified before queue release. Failed algorithms require inspection. Only verified orphan checkpoints recover automatically; SIGHUP detaches and preserves workers.'})
            write_json(output/'queue_status.json',status(db))
            if counts.get('complete',0)==sum(counts.values()): break
            if ready and not workers and not counts.get('pending',0) and not counts.get('blocked',0): break
            for _ in range(30):
                if stopping: break
                time.sleep(1)
    finally:
        if stopping and not detach:
            for child in children.values(): child.wait()
        for stream in streams.values(): stream.close()
        db.close(); lease.close()


if __name__=='__main__':
    main()
