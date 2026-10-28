"""SQLite experiment queue: explicit state, durable commands, resumable workers."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from .readiness import DatasetReadiness

STATUSES=("pending","running","failed","complete","blocked")


def connect(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=30)
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, config TEXT NOT NULL, status TEXT NOT NULL, priority INTEGER DEFAULT 10, pid INTEGER, started REAL, ended REAL, error TEXT, gpu TEXT)")
    db.commit()
    return db


def enqueue(db,job_id,config,priority=10):
    db.execute("BEGIN IMMEDIATE")
    try:
        previous=db.execute("SELECT config FROM jobs WHERE id=?",(job_id,)).fetchone()
        if previous is not None:
            if json.loads(previous["config"]) != config:
                raise ValueError(f"Job {job_id} already has a different frozen configuration")
        else:
            destination=Path(config["output"]).resolve()
            for row in db.execute("SELECT id,config FROM jobs"):
                if Path(json.loads(row["config"])["output"]).resolve()==destination:
                    raise ValueError(f"Output directory already belongs to job {row['id']}")
            db.execute("INSERT INTO jobs (id,config,status,priority) VALUES (?,?,?,?)",(job_id,json.dumps(config),"pending",priority))
        db.commit()
    except Exception:
        db.rollback()
        raise


def claim(db,gpu):
    db.execute("BEGIN IMMEDIATE")
    row=db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY priority,id LIMIT 1").fetchone()
    if row is not None:
        db.execute("UPDATE jobs SET status='running',started=?,gpu=?,error=NULL WHERE id=?",(time.time(),str(gpu),row["id"]))
    db.commit()
    return row


def process_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid,0)
        stat=Path(f"/proc/{pid}/stat").read_text().split()
        return stat[2]!="Z"
    except (ProcessLookupError,FileNotFoundError):
        return False
    except PermissionError:
        return True


def exclusive_lock(path):
    """Hold this descriptor for the lifetime of one supervisor/GPU worker."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a')
    try:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def refresh_readiness(db, checker):
    """CAS updates preserve jobs claimed while byte verification is in flight."""
    for row in db.execute("SELECT id,config,status,error FROM jobs WHERE status IN ('pending','blocked')").fetchall():
        if row['status']=='blocked' and not (row['error'] or '').startswith(
                ('Dataset not ready:', 'Required 100M shard manifest unavailable')):
            continue  # A manual/non-data block is not cleared by available files.
        reason=checker(json.loads(row['config']))
        if reason:
            db.execute("UPDATE jobs SET status='blocked',error=? WHERE id=? AND status=?",
                       (reason,row['id'],row['status']))
        elif row['status']=='blocked':
            db.execute("UPDATE jobs SET status='pending',error=NULL WHERE id=? AND status='blocked'",(row['id'],))
        db.commit()


def recover_orphans(db, *, protected_gpus=(), now=None):
    """Recover dead trainers only after verifying a complete durable snapshot.

    A live worker may still be recording its child's exit. Such GPU leases,
    live trainer PIDs and fresh claims are always left untouched.
    """
    from .logging import load_checkpoint
    from .readiness import signature
    now=time.time() if now is None else now
    report={'recovered':[], 'unrecoverable':{}}
    def transition(row, target, reason):
        db.execute('BEGIN IMMEDIATE')
        current=db.execute('SELECT status,pid,started,gpu FROM jobs WHERE id=?',(row['id'],)).fetchone()
        unchanged=(current['status']=='running' and all(current[k]==row[k] for k in ('pid','started','gpu')))
        changed=False
        if unchanged and not process_alive(current['pid']):
            db.execute('UPDATE jobs SET status=?,pid=NULL,error=?,ended=? WHERE id=?',
                       (target,reason,now,row['id']))
            changed=True
        db.commit()
        return changed
    for row in db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
        if str(row['gpu']) in protected_gpus or process_alive(row['pid']):
            continue
        if row['pid'] is None and now-(row['started'] or now)<60:
            continue
        config=json.loads(row['config']); path=Path(config['output'])/'checkpoint.pkl'
        try:
            before=signature(path); state=load_checkpoint(path)
            required={'config','agent','sampler','step','numpy_rng','python_rng','elapsed_seconds'}
            if not isinstance(state,dict) or not required<=state.keys() or not state['agent'] or not state['sampler']:
                raise ValueError('Incomplete training checkpoint')
            if any(state['config'].get(k)!=value for k,value in config.items()):
                raise ValueError('Checkpoint differs from frozen queue configuration')
            if not isinstance(state['step'],int) or not 0<=state['step']<=config['steps']:
                raise ValueError('Invalid checkpoint update counter')
            if signature(path)!=before:
                raise ValueError('Checkpoint changed during verification')
        except Exception as error:
            reason=f'Manual retry required: orphan checkpoint unavailable or invalid: {error}'
            if transition(row,'failed',reason):
                report['unrecoverable'][row['id']]=reason
            continue
        if transition(row,'pending',f"Recovered verified orphan from checkpoint step {state['step']}"):
            report['recovered'].append(row['id'])
    return report


def worker(db,gpu,python,max_jobs=0):
    database=Path(db.execute('PRAGMA database_list').fetchone()[2])
    lease=exclusive_lock(database.with_name(database.name+f'.gpu-{gpu}.lock'))
    if lease is None:
        return 0
    try:
        return _worker(db,gpu,python,max_jobs)
    finally:
        lease.close()


def _worker(db,gpu,python,max_jobs=0):
    completed=0
    interrupted=False
    child=None
    readiness=DatasetReadiness()
    def stop(sig,frame):
        nonlocal interrupted
        interrupted=True
        if child is not None and child.poll() is None:
            child.terminate()  # Trainer writes complete state at the next update boundary.
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    while not interrupted and (not max_jobs or completed<max_jobs):
        row=claim(db,gpu)
        if row is None:
            break
        config=json.loads(row["config"])
        output=Path(config["output"]); output.mkdir(parents=True,exist_ok=True)
        config_path=output/"queued_config.json"
        config_path.write_text(json.dumps(config,indent=2)+"\n")
        command=[python,"-m","doublemdp.train","--config",str(config_path)]
        if (output/"checkpoint.pkl").exists():
            command += ["--resume"]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE="false",MUJOCO_GL="egl",OMP_NUM_THREADS="2")
        (output/"command.json").write_text(json.dumps({"argv":command,"cwd":str(Path(__file__).resolve().parent.parent),
            "environment":{k:env[k] for k in ("CUDA_VISIBLE_DEVICES","XLA_PYTHON_CLIENT_PREALLOCATE","MUJOCO_GL","OMP_NUM_THREADS")}},indent=2)+"\n")
        reason=readiness(config)
        if reason:
            db.execute("UPDATE jobs SET status='blocked',error=?,ended=?,pid=NULL WHERE id=?",(reason,time.time(),row["id"]))
            db.commit(); continue
        with (output/"console.log").open("a") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=Path(__file__).resolve().parent.parent)
            db.execute("UPDATE jobs SET pid=? WHERE id=?",(child.pid,row["id"])); db.commit()
            print(json.dumps({"job":row["id"],"pid":child.pid,"gpu":gpu,"status":"running"}),flush=True)
            code=child.wait()
        status_file=output/"status.json"
        result=json.loads(status_file.read_text()) if status_file.exists() else {}
        status="complete" if code==0 and result.get("status")=="complete" else "pending" if interrupted else "failed"
        error=None if status=="complete" else (output/"console.log").read_text()[-4000:]
        db.execute("UPDATE jobs SET status=?,ended=?,error=?,pid=NULL WHERE id=?",(status,time.time(),error,row["id"])); db.commit()
        child=None; completed+=1
    return completed


def status(db):
    records=[]
    for row in db.execute("SELECT * FROM jobs ORDER BY priority,id"):
        row=dict(row); cfg=json.loads(row.pop("config"))
        row["output"]=cfg["output"]
        path=Path(cfg["output"])/"status.json"
        if path.exists():
            progress=json.loads(path.read_text())
            row["step"]=progress["step"]; row["total_steps"]=progress["total_steps"]
            row["elapsed_seconds"]=progress["elapsed_seconds"]
            if progress["step"]:
                row["estimated_remaining_seconds"]=(progress["total_steps"]-progress["step"])*progress["elapsed_seconds"]/progress["step"]
        command_path=Path(cfg["output"])/"command.json"
        recorded=json.loads(command_path.read_text()) if command_path.exists() else {}
        if (Path(cfg["output"])/"checkpoint.pkl").exists():
            command=list(recorded.get("argv",[sys.executable,"-m","doublemdp.train","--config",str(Path(cfg["output"])/"queued_config.json")]))
            if "--resume" not in command:
                command.append("--resume")
            row["resume_command"]=command
            row["resume_environment"]=recorded.get("environment",{})
            row["resume_cwd"]=recorded.get("cwd",str(Path(__file__).resolve().parent.parent))
        else:
            row["resume_command"]=None
            row["resume_reason"]="No checkpoint yet; launch or recover through the queue worker"
        records.append(row)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["enqueue","work","status","resume"])
    p.add_argument("--queue",required=True)
    p.add_argument("--configs",nargs="*")
    p.add_argument("--gpu",default="0")
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--max-jobs",type=int,default=0)
    p.add_argument("--retry-failed",action="store_true")
    p.add_argument("--retry-blocked",action="store_true")
    args=p.parse_args(); db=connect(args.queue)
    if args.action=="enqueue":
        for path in args.configs or []:
            config=json.loads(Path(path).read_text())
            enqueue(db,config.get("run_id",Path(config["output"]).name),config,config.get("priority",10))
    elif args.action=="work":
        worker(db,args.gpu,args.python,args.max_jobs)
    elif args.action=="resume":
        for row in db.execute("SELECT id,pid,started FROM jobs WHERE status='running'").fetchall():
            if not process_alive(row["pid"]) and (row["pid"] is not None or time.time()-row["started"]>60):
                db.execute("UPDATE jobs SET status='pending',pid=NULL,error='Recovered interrupted worker' WHERE id=?",(row["id"],))
        if args.retry_failed:
            db.execute("UPDATE jobs SET status='pending' WHERE status='failed'")
        if args.retry_blocked:
            db.execute("UPDATE jobs SET status='pending' WHERE status='blocked'")
        db.commit()
    print(json.dumps(status(db),indent=2))


if __name__=="__main__":
    main()
