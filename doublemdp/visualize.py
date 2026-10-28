"""Regenerable native sampling plots and a compact continuation atlas."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
from .logging import write_json
from .diagnostics import native_progress_indices


def collect(run, output, *, particles=128, continuation=False):
    import jax
    from .evaluate import load_run, evaluate_rollouts
    from .toy import Bandit, PointMaze
    agent,config,metadata,checkpoint=load_run(run)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    env_name=config["env"]
    if env_name not in ("bandit","maze"):
        raise ValueError("Toy visualization accepts bandit or maze checkpoints")
    probes=np.eye(3,dtype=np.float32) if env_name=="bandit" else PointMaze.probes
    values={"probe_states":probes,"training_step":np.array(checkpoint["step"])}
    descriptions={"method":config["method"],"env":env_name,"dataset":config["dataset"],"profile":config["profile"],
                  "seed":731,"training_step":checkpoint["step"],"policy":"single_candidate_probe",
                  "checkpoint":checkpoint["artifact_identity"],"reward":config.get("reward","sparse"),
                  "dataset_arrays_sha256":checkpoint["sampler"]["sha256"],
                  "native_value_semantics":"Native expectile/regularized head; not equated to unregularized Monte Carlo return",
                  "generation_time":"Actual native sampler steps; no interpolated states",
                  "environment_time":"Only decoded final actions execute; latents are action-space quantities"}
    for i,obs in enumerate(probes):
        if config["method"]=="rql":
            trace=agent.sample_trace(obs,jax.random.PRNGKey(731),num_samples=particles)
        else:
            trace=agent.sample_trace(np.repeat(obs[None],particles,axis=0),seed=731,num_samples=1)
        for key in ("latents","native_times","progress","native_values","initial_noise","final_actions","actions","candidate_actions","step_path_kl"):
            if key in trace:
                values[f"probe{i}_{key}"]=np.asarray(trace[key])
        if env_name=="bandit":
            ref=Bandit(gap=metadata.get("gap",False)).reference_grid(i,config.get("agent",{}).get("critic",{}).get("eta",.06))
            values.update({f"probe{i}_{'reference_actions' if k == 'actions' else k}":v for k,v in ref.items()})
        if continuation and env_name=="maze":
            from .diagnostics import continuation_probe
            result=continuation_probe(agent,config["method"],obs,trace,repetitions=16,horizon=100)
            values.update({f"probe{i}_continuation_{k}":v for k,v in result.items() if not isinstance(v,str)})
    if env_name=="maze":
        from .data import OfflineDataset
        dataset=OfflineDataset.load(config["dataset"],reward=config.get("reward","sparse"))
        # Coverage is itself a plot input. Saved figures can regenerate after
        # the external training dataset has moved or been removed.
        stride=max(1,int(np.ceil(len(dataset)/6000)))
        values["dataset_coverage_observations"]=dataset.arrays["observations"][::stride]
        from .diagnostics import grid_dynamic_programming,refinement_check,rollout_continuations
        oracle=grid_dynamic_programming(41)
        values.update({"oracle_"+k:v for k,v in oracle.items()})
        write_json(output/"grid_refinement.json",refinement_check())
        evaluation=evaluate_rollouts(agent,dict(config,eval_episodes=4),PointMaze(),save_path=output/"rollouts.npz")
        write_json(output/"rollouts.json",evaluation)
        if continuation:
            axis=np.linspace(-1,1,9); x,y=np.meshgrid(axis,axis); actions=np.stack([x.ravel(),y.ravel()],-1)
            for i,obs in enumerate(probes):
                result=rollout_continuations(agent,config["method"],np.repeat(obs[None],len(actions),axis=0),actions,repetitions=8,horizon=100)
                values[f"probe{i}_q_axis"]=axis
                values[f"probe{i}_q_mean"]=result["mean"].reshape(9,9)
                values[f"probe{i}_q_se"]=result["se"].reshape(9,9)
            physical_axis=np.linspace(0,1,13)
            x,y=np.meshgrid(physical_axis,physical_axis); states=np.stack([x.ravel(),y.ravel()],-1)
            result=rollout_continuations(agent,config["method"],states,repetitions=8,horizon=100)
            values["policy_value_axis"]=physical_axis
            values["policy_value_mean"]=result["mean"].reshape(13,13)
            values["policy_value_se"]=result["se"].reshape(13,13)
            descriptions["mc_reference"]="Single-candidate policy, 100-step truncated rollouts; SE is conditional simulation uncertainty, not training-seed uncertainty"
    np.savez_compressed(output/"plot_inputs.npz",**values)
    write_json(output/"plot_metadata.json",descriptions)
    return output


def _trace(data,index):
    latents=data[f"probe{index}_latents"]
    if latents.ndim==4:
        latents=latents[:,:,0,:]
    # Maze/bandit diagnostics use H=1. Retain unbounded native latents.
    return latents[:,:,:2]


def _final_actions(data,index):
    key=f"probe{index}_final_actions"
    if key in data:
        return data[key][:,0,:2]
    actions=data[f"probe{index}_actions"]
    if actions.ndim != 2:
        # Early plot-input artifacts had a reward-grid key collision. Actual
        # single-candidate endpoints are independently retained in the trace.
        candidates=data[f"probe{index}_candidate_actions"]
        if candidates.ndim != 3 or candidates.shape[1] != 1:
            raise ValueError("Ambiguous final actions; recollect from the checkpoint")
        actions=candidates[:,0]
    return actions[:,:2]


def draw_walls(ax):
    from matplotlib.patches import Rectangle
    from .toy import PointMaze
    for x0,y0,x1,y1 in PointMaze.boxes:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color="#354052",zorder=5))
    ax.scatter(*PointMaze.goal,marker="*",s=160,c="#edac33",edgecolor="black",zorder=10)
    ax.set(xlim=(0,1),ylim=(0,1),aspect="equal",xlabel="physical x",ylabel="physical y")


def render(output, *, movie=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import OfflineDataset
    output=Path(output)
    meta=json.loads((output/"plot_metadata.json").read_text())
    with np.load(output/"plot_inputs.npz") as file:
        data={k:file[k] for k in file.files}
    coverage=data.get("dataset_coverage_observations")
    if meta["env"]=="maze" and coverage is None:
        # Compatibility for already generated artifacts; verify identity when
        # an older metadata file contains it.
        dataset=OfflineDataset.load(meta["dataset"],reward=meta.get("reward","sparse"))
        if meta.get("dataset_arrays_sha256") not in (None,dataset.digest):
            raise ValueError("Plot coverage dataset differs from the saved checkpoint")
        coverage=dataset.arrays["observations"][::max(1,int(np.ceil(len(dataset)/6000)))]
    for i,obs in enumerate(data["probe_states"]):
        trace=_trace(data,i)
        final=_final_actions(data,i)
        idx=native_progress_indices(data[f"probe{i}_progress"])
        times=data[f"probe{i}_native_times"]
        colors=np.arctan2(final[:,1],final[:,0])
        fig=plt.figure(figsize=(16,8),layout="constrained")
        grid=fig.add_gridspec(2,5)
        if meta["env"]=="bandit":
            ax=fig.add_subplot(grid[0,0:2])
            image=ax.imshow(data[f"probe{i}_reward"],origin="lower",extent=(-1,1,-1,1),cmap="viridis")
            ax.scatter(*final.T,s=7,c="white",alpha=.35)
            final_lim=max(1.,float(np.abs(final).max()))*1.05
            ax.set(title=f"Context {i}: reward oracle + decoded actions",xlabel="action x",ylabel="action y",xlim=(-final_lim,final_lim),ylim=(-final_lim,final_lim))
            fig.colorbar(image,ax=ax,label="reward")
            ax=fig.add_subplot(grid[0,2])
            ax.imshow(data[f"probe{i}_behavior_density"],origin="lower",extent=(-1,1,-1,1),cmap="Blues")
            ax.set(title="Analytic behavior density")
            ax=fig.add_subplot(grid[0,3])
            ax.imshow(data[f"probe{i}_reference_mass"],origin="lower",extent=(-1,1,-1,1),cmap="magma")
            ax.set(title="Endpoint-KL reference\n(objective-specific)")
            ax=fig.add_subplot(grid[0,4])
            ax.hist2d(*final.T,bins=35,range=[[-final_lim,final_lim],[-final_lim,final_lim]],cmap="Blues")
            ax.set(title="Empirical final-action mass")
        else:
            ax=fig.add_subplot(grid[0,0:2])
            key="policy_value_mean" if "policy_value_mean" in data else "oracle_values"
            image=ax.imshow(data[key],origin="lower",extent=(0,1,0,1),alpha=.65,cmap="viridis")
            ax.scatter(*coverage.T,s=1,c="white",alpha=.3)
            if (output/"rollouts.npz").exists():
                with np.load(output/"rollouts.npz") as trajectories:
                    for k in trajectories.files:
                        if k.startswith("episode_"):
                            ax.plot(*trajectories[k].T,alpha=.8,lw=1)
            ax.scatter(*obs,c="red",marker="x",s=100,zorder=10)
            draw_walls(ax)
            ax.set_title("Physical maze: policy-specific MC V" if key=="policy_value_mean" else "Physical maze: approximate grid optimum")
            fig.colorbar(image,ax=ax)
            ax=fig.add_subplot(grid[0,2])
            if f"probe{i}_q_mean" in data:
                ax.imshow(data[f"probe{i}_q_mean"],origin="lower",extent=(-1,1,-1,1),cmap="viridis")
            ax.scatter(*final.T,c=colors,s=8,cmap="twilight")
            ax.set(title="Single-candidate Q^pi + final actions",xlim=(-1,1),ylim=(-1,1),xlabel="action x",ylabel="action y")
            ax=fig.add_subplot(grid[0,3:5])
            prefix=f"probe{i}_continuation_"
            if prefix+"mean" in data:
                ax.errorbar(data[prefix+"times"],data[prefix+"mean"],yerr=1.96*data[prefix+"se"],marker="o",label="q_probe ± 1.96 MC SE")
                ax.legend()
            else:
                ax.text(.5,.5,"Continuation MC not collected\nRun with --continuation",ha="center",transform=ax.transAxes)
            ax.set(title="Held-out remaining-generation continuation",xlabel="native time",ylabel="unregularized return")
        lim=max(1.,float(np.max(np.abs(trace))))*1.05
        for col,k in enumerate(idx):
            ax=fig.add_subplot(grid[1,col])
            ax.scatter(*trace[k].T,c=colors,cmap="twilight",s=10)
            for particle in range(min(12,trace.shape[1])):
                ax.plot(*trace[:k+1,particle].T,color="#aaaaaa",alpha=.35,lw=.6)
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect="equal",title=f"Generation {data[f'probe{i}_progress'][k]:.2f}; native t={times[k]:g}",xlabel="action latent x",ylabel="action latent y")
        fig.suptitle(f"{meta['method'].upper()} · {meta['profile']} · update {meta['training_step']} · probe {i}\nPersistent particles, colored by final action; native latent range retained",fontsize=13)
        for suffix in ("png","pdf"):
            fig.savefig(output/f"atlas_probe{i}.{suffix}",dpi=150,bbox_inches="tight")
        plt.close(fig)
    if movie:
        import imageio.v2 as imageio
        trace=_trace(data,0); times=data["probe0_native_times"]
        final=_final_actions(data,0)
        colors=np.arctan2(final[:,1],final[:,0]); lim=max(1.,float(np.abs(trace).max()))*1.05
        frames=[]
        for k in range(len(trace)):
            fig,ax=plt.subplots(figsize=(6,6),layout="constrained")
            ax.scatter(*trace[k].T,c=colors,cmap="twilight",s=16)
            for particle in range(min(16,trace.shape[1])):
                ax.plot(*trace[:k+1,particle].T,alpha=.3,lw=.7)
            ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect="equal",xlabel="action latent x",ylabel="action latent y",
                   title=f"{meta['method']} | training update {meta['training_step']}\nGeneration step {k}/{len(trace)-1}; native t={times[k]:g}\nAction-space latents, not physical positions")
            fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()); plt.close(fig)
        try:
            imageio.mimsave(output/"generation.mp4",frames,fps=8,codec="libx264")
        except (RuntimeError,OSError,ValueError):
            imageio.mimsave(output/"generation.gif",frames,duration=125,loop=0)
    figures=sorted(output.glob("atlas_probe*.png"))
    page="<!doctype html><meta charset='utf-8'><title>doublemdp diagnostics</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto}img{width:100%}</style>"
    page+=f"<h1>{html.escape(meta['method'])} diagnostics</h1><p>{html.escape(meta['native_value_semantics'])}</p>"
    page+="<p>Environment time, native generation time and training update are distinct. Raw inputs: <a href='plot_inputs.npz'>NPZ</a>; <a href='plot_metadata.json'>metadata</a>.</p>"
    for figure in figures:
        page+=f"<img src='{figure.name}' alt='Native sampling atlas'><p><a href='{figure.with_suffix('.pdf').name}'>PDF</a></p>"
    if (output/"generation.mp4").exists():
        page+="<video controls loop src='generation.mp4' style='max-width:600px'></video>"
    (output/"index.html").write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",nargs="?")
    p.add_argument("--output",required=True)
    p.add_argument("--regenerate",action="store_true",help="Use saved plot inputs; no checkpoint or retraining")
    p.add_argument("--continuation",action="store_true")
    p.add_argument("--particles",type=int,default=128)
    p.add_argument("--no-movie",action="store_true")
    args=p.parse_args()
    if not args.regenerate:
        if not args.run:
            p.error("run required unless --regenerate")
        collect(args.run,args.output,particles=args.particles,continuation=args.continuation)
    render(args.output,movie=not args.no_movie)
    print(Path(args.output)/"index.html")


if __name__=="__main__":
    main()
