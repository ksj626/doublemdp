#!/usr/bin/env python
"""Plot recorded representative runs, keeping episode and training-seed uncertainty distinct."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path.home()/'doublemdp_outputs')
    args = parser.parse_args()
    root = args.output_root.resolve()
    out = root/'figures/reproduction'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).resolve().parents[1]/'configs/rql_ogbench_manifest.json').read_text())
    entries = {entry['env']: entry for entry in manifest['tasks']}
    tasks = ['antmaze-large-navigate-singletask-task1-v0', 'scene-play-singletask-task2-v0']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout='constrained')
    raw = {}
    metadata = {'policy': 'Final requested checkpoint; no best-checkpoint selection.',
                'local_uncertainty': 'Wilson 95% interval over 50 evaluation episodes per checkpoint, conditional on one trained seed. Evaluation seeds recur across checkpoints; these are not independent training-seed intervals.',
                'paper_uncertainty': 'Published four-training-seed mean and reported interval. Paper 2M versus public-code 1M budget discrepancy is unresolved.',
                'runs': []}
    for ax, task in zip(axes, tasks):
        run = root/'runs'/f'rql_{task}_official_code_seed0'
        status = json.loads((run/'status.json').read_text())
        rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row.get('phase') == 'evaluation' and 'success' in row]
        steps = np.asarray([row['step'] for row in rows])
        scores = np.asarray([row['success'] for row in rows])
        n, z = 50, 1.959963984540054
        center = (scores+z*z/(2*n))/(1+z*z/n)
        radius = z*np.sqrt(scores*(1-scores)/n+z*z/(4*n*n))/(1+z*z/n)
        # Roundoff at p=0 or p=1 must not put an endpoint beyond its estimate.
        low = np.minimum(scores, np.maximum(0, center-radius))
        high = np.maximum(scores, np.minimum(1, center+radius))
        reference = entries[task]['paper_reference']
        ax.axhspan(*reference['ci95_percent'], color='0.75', alpha=.35, label='Paper interval: 4 training seeds')
        ax.axhline(reference['score_percent'], color='0.4', linestyle='--', label='Paper mean')
        ax.errorbar(steps/1e6, 100*scores, yerr=100*np.asarray([scores-low, high-scores]),
                    fmt='o-', capsize=3, color='#176ba0', label='Local seed 0 + episode interval')
        ax.set(title=task.replace('-singletask-', '\n'), xlabel='Offline updates (millions)', ylim=(-3, 103))
        ax.grid(alpha=.2)
        raw.update({task+'_steps': steps, task+'_success': scores,
                    task+'_episode_ci95': np.stack([low, high], axis=1)})
        metadata['runs'].append({'task': task, 'path': str(run), 'status': status,
                                 'paper_reference': reference, 'evaluation_episodes': n})
    axes[0].set_ylabel('Success (%)')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle('RQL public-code budget: 1M updates, one local training seed\nEpisode intervals do not measure training-seed uncertainty', fontsize=12)
    fig.savefig(out/'rql_representatives.png', dpi=160)
    fig.savefig(out/'rql_representatives.pdf')
    plt.close(fig)
    np.savez_compressed(out/'rql_representatives.npz', **raw)
    (out/'rql_representatives.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
