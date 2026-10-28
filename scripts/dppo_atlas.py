#!/usr/bin/env python3
"""Collect/regenerate the DPPO continuation atlas and saved-checkpoint history."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def collect(args):
    import torch
    from doublemdp.agents.dppo_artifacts import load_evaluation_checkpoint,file_sha256
    from doublemdp.agents.dppo_diagnostics import noise_tape,rollouts,q_probe,summarize_samples
    from doublemdp.data import OfflineDataset
    from doublemdp.toy import PointMaze
    from doublemdp.diagnostics import native_progress_indices,grid_dynamic_programming,refinement_check
    torch.set_num_threads(1)
    run,output=Path(args.run),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    agents,identities=[],[]
    for filename in ("bc.pt","latest.pt"):
        agent,identity=load_evaluation_checkpoint(run/filename,device=args.device,env="maze")
        if agent.config.horizon_steps!=1:
            raise ValueError("Continuation atlas requires the declared chunk-length-one maze")
        agents.append(agent); identities.append(identity)
    if identities[0]["phase"]!="bc" or identities[1]["phase"]!="online":
        raise ValueError("History requires the actual saved BC and online checkpoints")
    if identities[0]["metadata"]["dataset_sha256"]!=identities[1]["metadata"]["dataset_sha256"]:
        raise ValueError("History checkpoints must use identical saved data")
    probes=PointMaze.probes.copy()
    count,repetitions,steps=len(probes),args.repetitions,agents[-1].config.denoising_steps
    if any(a.config.denoising_steps!=steps or a.config.gamma!=agents[-1].config.gamma for a in agents):
        raise ValueError("Common-noise history requires identical native steps and discount")
    particles=args.particles
    rng=np.random.RandomState(args.seed)
    generation=rng.normal(size=(steps+1,count*particles,1,2)).astype(np.float32)
    remaining=noise_tape(args.seed+1,1,count*repetitions,steps)[0,1:]
    continuation_noise=noise_tape(args.seed+2,args.horizon,count*repetitions,steps)
    route_noise=noise_tape(args.seed+3,args.horizon,count*4,steps)
    destination_noise=noise_tape(args.seed+6,args.horizon,count*particles,steps)
    values={"probe_states":probes,"generation_initial_noise":generation[0],"generation_noise":generation[1:],
        "remaining_generation_noise":remaining,"continuation_outer_noise":continuation_noise,"route_noise":route_noise,
        "particle_destination_noise":destination_noise}
    total_interactions=0
    history=[]
    for stage,(agent,identity) in enumerate(zip(agents,identities)):
        observations=np.repeat(probes,particles,axis=0)
        trace=agent.trace_with_noise(observations,generation[0],generation[1:],deterministic=True)
        trace["particles_per_probe"]=particles
        indices=native_progress_indices(trace["progress"])
        probe=q_probe(agent,probes,trace,indices,remaining,continuation_noise)
        routes=rollouts(agent,np.repeat(probes,4,axis=0),route_noise)
        destinations=rollouts(agent,observations,destination_noise,first_actions=trace["final_actions"][:,0])
        for key in ("latents","native_times","progress","final_actions"):
            values[f"stage{stage}_{key}"]=trace[key]
        values[f"stage{stage}_indices"]=indices
        for key in ("mean","se","samples","unfinished","endpoints"):
            values[f"stage{stage}_continuation_{key}"]=probe[key]
        values[f"stage{stage}_routes"]=routes["paths"]
        values[f"stage{stage}_route_counts"]=routes["counts"]
        values[f"stage{stage}_particle_destinations"]=destinations["paths"][-1]
        values[f"stage{stage}_particle_success"]=destinations["returns"]>0
        if trace["native_environment_values"] is not None:
            values[f"stage{stage}_native_state_value"]=trace["native_environment_values"][::particles].reshape(-1)
        total_interactions+=probe["evaluation_steps"]+int(routes["counts"].sum()+destinations["counts"].sum())
        # Same supplied remaining tape must replay the original path exactly.
        replay_errors=[]
        for index in indices:
            replay=agent.continue_generation(observations,trace["latents"][:,index],int(trace["native_times"][index]),
                deterministic=True,noise_tape=generation[1:])
            replay_errors.append(float(np.max(abs(replay-trace["final_actions"]))))
        history.append({"stage":identity["phase"],"checkpoint":identity,"mean_q_probe":probe["mean"].tolist(),
            "mc_se":probe["se"].tolist(),"unfinished_fraction":probe["unfinished"].mean(-1).tolist(),
            "same_path_replay_max_error":max(replay_errors)})
        print(json.dumps({"stage":identity["phase"],"elapsed_seconds":time.perf_counter()-started}),flush=True)
    # Q^pi and V^pi use the same final policy and exact same evaluation sampler.
    agent=agents[-1]
    action_axis=np.linspace(-1,1,args.action_resolution)
    ax,ay=np.meshgrid(action_axis,action_axis)
    action_grid=np.stack([ax.ravel(),ay.ravel()],-1).astype(np.float32)
    qstates=np.repeat(probes,len(action_grid)*repetitions,axis=0)
    first=np.tile(np.repeat(action_grid,repetitions,axis=0),(count,1))
    qnoise=noise_tape(args.seed+4,args.horizon,count*repetitions,steps)
    # Common random numbers across action-grid choices; repetitions independent.
    qgroups=np.concatenate([np.tile(np.arange(p*repetitions,(p+1)*repetitions),len(action_grid)) for p in range(count)])
    q=rollouts(agent,qstates,qnoise,first_actions=first,groups=qgroups)
    qs=q["returns"].reshape(count,args.action_resolution,args.action_resolution,repetitions)
    values.update(q_axis=action_axis,q_mean=summarize_samples(qs)["mean"],q_se=summarize_samples(qs)["se"],
        q_samples=qs,q_unfinished=q["unfinished"].reshape(qs.shape),q_outer_noise=qnoise)
    physical_axis=np.linspace(0,1,args.state_resolution)
    sx,sy=np.meshgrid(physical_axis,physical_axis)
    gridstates=np.stack([sx.ravel(),sy.ravel()],-1).astype(np.float32)
    vnoise=noise_tape(args.seed+5,args.horizon,repetitions,steps)
    v=rollouts(agent,np.repeat(gridstates,repetitions,axis=0),vnoise,groups=np.tile(np.arange(repetitions),len(gridstates)))
    vs=v["returns"].reshape(args.state_resolution,args.state_resolution,repetitions)
    values.update(value_axis=physical_axis,value_mean=summarize_samples(vs)["mean"],value_se=summarize_samples(vs)["se"],
        value_samples=vs,value_unfinished=v["unfinished"].reshape(vs.shape),value_outer_noise=vnoise)
    total_interactions+=int(q["counts"].sum()+v["counts"].sum())
    data=OfflineDataset.load(identities[-1]["metadata"]["dataset_path"],reward=identities[-1]["metadata"]["reward"])
    if data.digest!=identities[-1]["metadata"]["dataset_sha256"]:
        raise ValueError("Coverage dataset differs from checkpoint identity")
    values["coverage"]=data.arrays["observations"][::max(1,len(data)//5000)]
    oracle=grid_dynamic_programming(41,gamma=agent.config.gamma)
    values.update(oracle_axis=oracle["axis"],oracle_values=oracle["values"])
    np.savez_compressed(output/"plot_inputs.npz",**values)
    metadata={"method":"dppo_online","policy":"single candidate, native source deterministic=True evaluation sampler; intermediate DDPM noise remains stochastic",
        "quantity":"unregularized discounted sparse return; zero tail after finite rollout horizon",
        "native_head":"environment V(s) only; trained reward/scale differs from sparse MC, so no native inner calibration is available",
        "seed":args.seed,"particles_per_probe":particles,"repetitions":repetitions,"horizon":args.horizon,"gamma":agent.config.gamma,
        "evaluation_interactions":total_interactions,"training_interactions":0,"device":args.device,
        "elapsed_seconds":time.perf_counter()-started,"history":history,"grid_refinement":refinement_check(gamma=agent.config.gamma),
        "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name":torch.cuda.get_device_name(agent.device) if agent.device.type=="cuda" else None,
        "checkpoint_history":"Only saved BC and final online checkpoints exist; no intermediate checkpoints were reconstructed",
        "uncertainty":"SE is conditional Monte Carlo uncertainty across supplied independent repetitions, not training-seed uncertainty; unfinished paths incur unestimated tail bias",
        "noise":"All Gaussian draws saved in plot_inputs.npz; common draws across native steps/checkpoints and action choices; no cross-method path identity claim",
        "plot_inputs_sha256":file_sha256(output/"plot_inputs.npz")}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"evaluation_interactions":total_interactions,"elapsed_seconds":metadata["elapsed_seconds"]}),flush=True)


def render(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import imageio.v2 as imageio
    from doublemdp.toy import PointMaze
    output=Path(output)
    raw=np.load(output/"plot_inputs.npz")
    meta=json.loads((output/"metadata.json").read_text())
    from doublemdp.agents.dppo_artifacts import file_sha256
    if file_sha256(output/"plot_inputs.npz")!=meta["plot_inputs_sha256"]:
        raise ValueError("Saved plot inputs differ from their recorded artifact identity")
    names=("fork","bottleneck","detour","near goal")
    particles=meta["particles_per_probe"]
    indices=raw["stage1_indices"]
    def walls(ax):
        for x0,y0,x1,y1 in PointMaze.boxes:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color="#354052",zorder=5))
        ax.scatter(*PointMaze.goal,marker="*",s=100,c="#e6ad32",edgecolor="black",zorder=8)
        ax.set(xlim=(0,1),ylim=(0,1),aspect="equal",xlabel="physical x",ylabel="physical y")
    limits=max(1.1,*(float(np.max(np.abs(raw[f"stage{s}_latents"]))) for s in (0,1)))*1.04
    def atlas(probe_index,stage,index):
        fig,axes=plt.subplots(1,4,figsize=(17,4.2))
        p=probe_index
        ax=axes[0]
        field=ax.imshow(raw["value_mean"],origin="lower",extent=(0,1,0,1),cmap="viridis",vmin=0,vmax=1,alpha=.85)
        fig.colorbar(field,ax=ax,fraction=.046,pad=.02).ax.tick_params(labelsize=7)
        ax.scatter(*raw["coverage"].T,s=.5,c="white",alpha=.25)
        routes=raw[f"stage{stage}_routes"]
        for r in range(4):
            count=int(raw[f"stage{stage}_route_counts"][p*4+r])
            ax.plot(routes[:count+1,p*4+r,0],routes[:count+1,p*4+r,1],lw=1,alpha=.85)
        ax.scatter(*raw["probe_states"][p],s=45,c="red",zorder=9)
        walls(ax)
        ax.set_title("Physical maze · white = saved data\ncolor = final-policy sparse Vpi MC")
        ax=axes[1]
        image=ax.imshow(raw["q_mean"][p],origin="lower",extent=(-1,1,-1,1),vmin=0,vmax=1,cmap="viridis")
        fig.colorbar(image,ax=ax,fraction=.046,pad=.02).ax.tick_params(labelsize=7)
        final=raw[f"stage{stage}_final_actions"][p*particles:(p+1)*particles,0]
        destinations=raw[f"stage{stage}_particle_destinations"][p*particles:(p+1)*particles]
        succeeded=raw[f"stage{stage}_particle_success"][p*particles:(p+1)*particles]
        # Color by realized terminal/finite-tail destination, never by the latent
        # interpreted as a physical state. Goal arrivals are green; remaining
        # particles retain spatial destination colors, labeled horizon-censored.
        color=np.stack([destinations[:,0],destinations[:,1],np.full(particles,.55)],-1)
        color[succeeded]=[.05,.85,.25]
        ax.scatter(*final.T,c=color,s=10,edgecolor="white",lw=.2)
        ax.set(xlim=(-1.1,1.1),ylim=(-1.1,1.1),aspect="equal",xlabel="executed action x",ylabel="executed action y",
            title="Final-policy Qpi(s,a) MC\npoints = selected checkpoint actions")
        ax=axes[2]
        latent=raw[f"stage{stage}_latents"][p*particles:(p+1)*particles,index,0]
        ax.scatter(*latent.T,c=color,s=15)
        ax.scatter(*latent[0],facecolor="none",edgecolor="black",s=110,lw=1.5,label="probed particle 0")
        ax.add_patch(Rectangle((-1,-1),2,2,fill=False,color="gray",ls="--"))
        ax.set(xlim=(-limits,limits),ylim=(-limits,limits),aspect="equal",xlabel="native action latent x",ylabel="native action latent y",
            title=f"Generation progress {raw[f'stage{stage}_progress'][index]:.2f} · t={raw[f'stage{stage}_native_times'][index]}\ncolor = physical destination\ngreen = goal; other = horizon-censored")
        ax.legend(fontsize=7,loc="upper left")
        ax=axes[3]
        progress=raw[f"stage{stage}_progress"][indices]
        for s,label in ((0,"BC"),(1,"final online")):
            ax.errorbar(progress,raw[f"stage{s}_continuation_mean"][p],yerr=1.96*raw[f"stage{s}_continuation_se"][p],fmt="o-",label=label,capsize=2)
        ax.axvline(raw[f"stage{stage}_progress"][index],color="gray",ls="--")
        ax.set(xlabel="generation progress (native steps)",ylabel="unregularized conditional return",ylim=(-.05,1.05),
            title=f"Continue selected latent · mean ±1.96 SE\n{meta['horizon']}-step tail truncated\nNo native inner-value head")
        ax.legend(fontsize=8)
        if f"stage{stage}_native_state_value" in raw:
            native=float(raw[f"stage{stage}_native_state_value"][p])
            ax.text(.02,.02,f"Native V(s)={native:.3g}: training objective,\nnot calibrated against sparse MC",transform=ax.transAxes,fontsize=7,
                bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
        for axis in axes:
            axis.title.set_fontsize(9)
            axis.tick_params(labelsize=8)
        fig.suptitle(f"DPPO {names[p]} · checkpoint {'BC' if stage==0 else 'final online'} · physical rollout time ≠ generation time ≠ training stage",fontsize=12)
        fig.tight_layout(rect=(0,0,1,.93))
        return fig
    for p,name in enumerate(names):
        fig=atlas(p,1,int(indices[-1]))
        for suffix in ("png","pdf"):
            fig.savefig(output/f"atlas_{name.replace(' ','_')}.{suffix}",dpi=150)
        plt.close(fig)
    frames=[]
    for p in range(4):
        for stage in range(2):
            for index in indices:
                fig=atlas(p,stage,int(index))
                fig.canvas.draw()
                frames.append(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
                plt.close(fig)
    imageio.mimsave(output/"continuation_atlas.gif",frames,duration=400,loop=0)
    try:
        imageio.mimsave(output/"continuation_atlas.mp4",frames,fps=2.5,macro_block_size=1)
    except Exception as error:
        (output/"mp4_error.txt").write_text(str(error))
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    for p,ax in enumerate(axes.ravel()):
        for s,label in ((0,"BC"),(1,"final online")):
            ax.errorbar(raw[f"stage{s}_progress"][indices],raw[f"stage{s}_continuation_mean"][p],
                yerr=1.96*raw[f"stage{s}_continuation_se"][p],fmt="o-",label=label,capsize=3)
        ax.set(title=names[p],xlabel="generation progress",ylabel="q_probe sparse return",ylim=(-.05,1.05))
        ax.legend(fontsize=8)
    fig.suptitle("Saved BC → final checkpoint · common noise across stages and inner steps\nConditional Monte Carlo uncertainty; finite-tail bias remains; monotonicity is not required")
    fig.tight_layout(rect=(0,0,1,.91))
    for suffix in ("png","pdf"):
        fig.savefig(output/f"checkpoint_history.{suffix}",dpi=160)
    plt.close(fig)
    html='<!doctype html><meta charset="utf-8"><title>DPPO continuation atlas</title><h1>DPPO single-candidate continuation atlas</h1><p>Actual native particles; physical, generation and training times are distinct. The same stochastic evaluation policy defines both remaining generation and outer Qpi. Common noise is saved. Green particles reached the physical goal; other colors encode horizon-censored physical destinations. Error bars show conditional MC uncertainty; truncated unfinished paths add tail bias. DPPO has no native inner head.</p>'
    html+='<video width="1200" controls src="continuation_atlas.mp4"></video><p><a href="continuation_atlas.gif">GIF fallback</a> · <a href="metadata.json">Evidence and identities</a> · <a href="plot_inputs.npz">Regenerable inputs and noise</a></p>'
    html+='<img width="1000" src="checkpoint_history.png">'
    for name in names:
        html+=f'<img width="1300" src="atlas_{name.replace(" ","_")}.png">'
    (output/"index.html").write_text(html)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run")
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cpu")
    p.add_argument("--regenerate",action="store_true")
    p.add_argument("--seed",type=int,default=731)
    p.add_argument("--particles",type=int,default=32)
    p.add_argument("--repetitions",type=int,default=8)
    p.add_argument("--horizon",type=int,default=100)
    p.add_argument("--action-resolution",type=int,default=7)
    p.add_argument("--state-resolution",type=int,default=11)
    args=p.parse_args()
    if not args.regenerate:
        if not args.run:
            p.error("--run is required when collecting")
        collect(args)
    render(args.output)


if __name__=="__main__":
    main()
