#!/usr/bin/env python3
"""Regenerate DPPO seed summaries and native-latent figures from saved artifacts."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",nargs="+",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--profile-device",default=None)
    args=p.parse_args()
    runs=[Path(x) for x in args.runs]
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    records=[[json.loads(s) for s in (r/"metrics.jsonl").read_text().splitlines()] for r in runs]
    bc,online=[],[]
    fig,ax=plt.subplots(figsize=(5,3.4))
    for run,rows in zip(runs,records):
        evaluations=[r for r in rows if "evaluation" in r["phase"]]
        bc.append(evaluations[0]["success_rate"])
        online.append(evaluations[-1]["success_rate"])
        ax.plot([r["online_env_steps"] for r in evaluations],[r["success_rate"] for r in evaluations],"o-",label=run.name)
    ax.set(xlabel="New training environment transitions",ylabel="Evaluation success",ylim=(0,1.05),title="Online DPPO after offline BC: three trained seeds")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png","pdf"):
        fig.savefig(output/f"dppo_success.{suffix}",dpi=180)
    plt.close(fig)
    summaries=[]
    for run,rows in zip(runs,records):
        updates=[r for r in rows if r["phase"]=="online"]
        summary=json.loads((run/"completed.json").read_text())
        summary.update(mean_steady_update_seconds=float(np.mean([r["update_seconds"] for r in updates[1:]])),
            mean_rollout_seconds=float(np.mean([r["rollout_seconds"] for r in updates])),
            final_success=next(r["success_rate"] for r in reversed(rows) if r["phase"]=="online_evaluation"))
        summaries.append(summary)
    report={"bc_success":bc,"dppo_success":online,"mean_final_success":float(np.mean(online)),
        "sample_std_across_training_seeds":float(np.std(online,ddof=1)) if len(online)>1 else None,
        "uncertainty_unit":"training seed; evaluation episodes are not independent training seeds",
        "runs":summaries,"selection":"final update","candidate_count":1,"ddpm_steps":20,
        "actor_network_calls_per_sample":30}
    if args.profile_device:
        import torch
        from doublemdp.agents.dppo_online import DPPOAgent
        torch.set_num_threads(1)
        agent,_=DPPOAgent.load(runs[0]/"latest.pt",device=args.profile_device)
        obs=np.array([[.32,.48]],dtype=np.float32)
        for _ in range(10):
            agent.sample_actions(obs)
        if agent.device.type=="cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        for _ in range(100):
            agent.sample_actions(obs)
        if agent.device.type=="cuda":
            torch.cuda.synchronize()
        report["inference_seconds_batch1"]=(time.perf_counter()-started)/100
        report["inference_device"]=str(agent.device)
        report["inference_peak_allocated_bytes"]=torch.cuda.max_memory_allocated() if agent.device.type=="cuda" else None
    (output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    # Plot actual native states; a latent action is never plotted on the maze.
    raw=np.load(runs[0]/"traces.npz")
    latents=raw["latents"][:,:,0,:]
    indices=np.array([np.argmin(abs(raw["progress"]-v)) for v in (0,.25,.5,.75,1)])
    limit=max(1.1,float(np.max(np.abs(latents[:,indices])))*1.04)
    colors=np.arctan2(latents[:,-1,1],latents[:,-1,0])
    fig,axes=plt.subplots(4,5,figsize=(12,9),sharex=True,sharey=True)
    for row,name in enumerate(("fork","bottleneck","detour","near goal")):
        selection=slice(row*128,(row+1)*128)
        for col,idx in enumerate(indices):
            ax=axes[row,col]
            z=latents[selection,idx]
            ax.scatter(z[:,0],z[:,1],c=colors[selection],cmap="twilight",s=10,vmin=-np.pi,vmax=np.pi)
            ax.add_patch(Rectangle((-1,-1),2,2,fill=False,color="gray",ls="--",lw=.7))
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect="equal")
            if row==0:
                ax.set_title(f"progress {raw['progress'][idx]:.2f}\nnative t={raw['native_times'][idx]}")
            if col==0:
                ax.set_ylabel(f"{name}\nlatent action y")
            if row==3:
                ax.set_xlabel("latent action x")
    fig.suptitle("DPPO native generation particles · color = final action direction\nEnvironment state fixed per row; dashed box = action bounds; these are not physical trajectories")
    fig.tight_layout(rect=[0,0,1,.94])
    for suffix in ("png","pdf"):
        fig.savefig(output/f"dppo_native_latents.{suffix}",dpi=160)
    plt.close(fig)
    import imageio.v2 as imageio
    frames=[]
    for idx in indices:
        fig,axes=plt.subplots(1,4,figsize=(12,3))
        for row,ax in enumerate(axes):
            selection=slice(row*128,(row+1)*128)
            z=latents[selection,idx]
            ax.scatter(z[:,0],z[:,1],c=colors[selection],cmap="twilight",s=12,vmin=-np.pi,vmax=np.pi)
            ax.add_patch(Rectangle((-1,-1),2,2,fill=False,color="gray",ls="--"))
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect="equal",xlabel="latent action x",ylabel="latent action y")
        fig.suptitle(f"Fixed physical probes · generation progress {raw['progress'][idx]:.2f} · actual native time {raw['native_times'][idx]}")
        fig.tight_layout()
        fig.canvas.draw()
        frames.extend([np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()]*8)
        plt.close(fig)
    imageio.mimsave(output/"dppo_generation.gif",frames,duration=100,loop=0)
    try:
        imageio.mimsave(output/"dppo_generation.mp4",frames,fps=8,macro_block_size=1)
    except Exception as error:
        (output/"mp4_error.txt").write_text(str(error))
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>DPPO native diagnostic</title><h1>DPPO online diagnostic</h1><p>Three fixed training seeds; final checkpoints. Gaussian action latents are not physical maze positions.</p><img width="700" src="dppo_success.png"><img width="1000" src="dppo_native_latents.png"><video width="1000" controls src="dppo_generation.mp4"></video><p><a href="summary.json">Measured results and compute</a></p>')
    print(json.dumps({k:report[k] for k in ("bc_success","dppo_success","mean_final_success","sample_std_across_training_seeds")},indent=2))


if __name__=="__main__":
    main()
