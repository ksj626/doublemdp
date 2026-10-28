"""Regenerate original BDPO toy figures/video from saved arrays; no model load.

python -m doublemdp.agents.bdpo_artifacts --out /external/completed-run
"""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from .bdpo_data import centers_8gaussians


def render_toy_artifacts(out, training_updates=None):
    out=Path(out)
    metadata=json.loads((out/'run.json').read_text())
    if training_updates is None:
        training_updates=int(metadata['completed_train_steps'])
    actions=np.load(out/'policy_samples.npy',allow_pickle=False)
    with np.load(out/'traces.npz',allow_pickle=False) as stored:
        arrays={key:stored[key] for key in stored.files}
    with np.load(out/'plot_inputs.npz',allow_pickle=False) as stored:
        beta,reward,reference,empirical=(stored[key] for key in
            ('behavior_density','reward','reference_mass','policy_mass'))
    modes=((actions[:,None]-centers_8gaussians())**2).sum(-1).argmin(-1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(16,6))
    for ax,title,data in zip(axes[0],['Analytic behavior','Reward oracle (evaluation)','Endpoint KL reference','Policy density','Final samples'],[beta,reward,reference,empirical,None]):
        if data is not None:
            ax.imshow(data.T,origin='lower',extent=(-5,5,-5,5),cmap='viridis')
        else:
            ax.scatter(actions[:,0],actions[:,1],s=1,c=modes,cmap='tab10',vmin=0,vmax=9)
        ax.set(xlim=(-5,5),ylim=(-5,5),title=title,aspect='equal')
    steps=arrays['latents'].shape[0]-1
    endpoint=arrays['latents'][-1,:,0]
    colors=((endpoint[:,None]-centers_8gaussians())**2).sum(-1).argmin(-1)
    indices=[int(np.argmin(np.abs(arrays['progress']-p))) for p in [0,.25,.5,.75,1]]
    extent=max(5,float(np.abs(arrays['latents']).max())*1.05)
    for ax,index in zip(axes[1],indices):
        points=arrays['latents'][index,:,0]
        ax.scatter(points[:,0],points[:,1],s=8,c=colors,cmap='tab10',vmin=0,vmax=9)
        ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',
               title=f'Generation {arrays["progress"][index]:.2f}; native t={arrays["native_times"][index]}')
    fig.suptitle('BDPO original 8gaussians task | action latents, not physical positions')
    fig.tight_layout(); fig.savefig(out/'toy.png',dpi=150); fig.savefig(out/'toy.pdf'); plt.close(fig)
    import imageio.v2 as imageio
    frames=[]
    for k,point in enumerate(arrays['latents'][:,:,0]):
        figure,ax=plt.subplots(figsize=(5,5))
        ax.scatter(point[:,0],point[:,1],c=colors,s=12,cmap='tab10',vmin=0,vmax=9)
        ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',title=f'Action generation: native t={steps-k}; update {training_updates}')
        figure.canvas.draw(); frames.append(np.asarray(figure.canvas.buffer_rgba())[...,:3].copy()); plt.close(figure)
    try:
        imageio.mimsave(out/'generation.mp4',frames,fps=12)
        animation='generation.mp4'
    except Exception:
        imageio.mimsave(out/'generation.gif',frames,duration=.08,loop=0)
        animation='generation.gif'
    write_gallery(out, metadata, training_updates, animation)
    return {'figure':str(out/'toy.png'),'pdf':str(out/'toy.pdf'),
            'animation':str(out/animation),'gallery':str(out/'index.html')}


def write_gallery(out, metadata, training_updates, animation):
    metrics=json.loads((out/'toy_evaluation.json').read_text())
    profile=html.escape(metadata['config']['profile'])
    moving=(f'<video controls preload="metadata" src="{animation}"></video>'
            if animation.endswith('.mp4') else f'<img src="{animation}" alt="Native DDPM particle generation">')
    files=['toy.png','toy.pdf',animation,'traces.npz','plot_inputs.npz','policy_samples.npy',
           'toy_evaluation.json','metrics.jsonl','run.json','checkpoint.pkl']
    links=''.join(f'<li><a href="{name}">{name}</a></li>' for name in files if (out/name).exists())
    page=f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>BDPO original 8gaussians</title><style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#172033}}
img{{width:100%;height:auto}}video{{max-width:100%;width:512px}}a{{color:#245bb2}}
</style><h1>BDPO original 8gaussians</h1>
<p>Profile: {profile}; saved RL updates: {training_updates:,}; eta: {metrics['eta']}.
Single-candidate evaluation: {metrics['sample_count']:,} samples.
Mean energy: {metrics['reward_mean']:.6f}; endpoint-grid JS: {metrics['grid_JS']:.6f}.</p>
<p>Particles are native DDPM action latents. The endpoint KL reference is objective-specific;
analytic rewards/density were used for data generation and evaluation only.</p>
<img src="toy.png" alt="Behavior, reward, reference, learned policy and native generation states">
{moving}<h2>Saved artifacts</h2><ul>{links}</ul>
<p>All figures and video regenerate from saved arrays without training, inference, or loading the checkpoint.</p>
</html>"""
    (out/'index.html').write_text(page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(render_toy_artifacts(args.out),indent=2))


if __name__=='__main__':
    main()
