#!/usr/bin/env python3
"""Run the pinned original Hopper source configs with explicit budget overrides.

Use the separately isolated legacy environment. A short --iterations run is
an original-task integration check, never a full-budget reproduction claim.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "dc8e0c9edce7ac2b2ff112abe460e1c21b0b3bdc"
FILES = {"state_3000.pt":"1uV3beg2YuBRh11t7jnRsyd_M9bsFGOkA",
         "normalization.npz":"1HHZ2X6r5io6hjG-MHVFFoPMJV2fJ3lis",
         "train.npz":"1m9CwHwKFzki28RG6HIdcYFbYUC2tH-9Y"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["pretrain","finetune","evaluate"],default="finetune")
    parser.add_argument("--assets",default=str(Path(os.environ.get("DOUBLEMDP_OUTPUTS","/home/dmsdmswns/doublemdp_outputs"))/"dppo_original"))
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--iterations",type=int,help="Explicit non-paper budget override")
    parser.add_argument("--epochs",type=int,help="Explicit BC budget override")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--checkpoint",help="Evaluate the final trained actor and its frozen prefix")
    parser.add_argument("--resume",help="Full BC epoch-boundary checkpoint; original online resume is unsupported")
    args=parser.parse_args()
    if args.resume and args.phase != "pretrain":
        parser.error("Original online resume is unsupported: source run() retains rollout locals and async worker simulator/RNG state outside its checkpoint. Use the maintained port for full online resume.")
    root=Path(__file__).resolve().parents[1]
    upstream=root/"third_party/dppo"
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=upstream,text=True).strip()!=COMMIT:
        raise RuntimeError("DPPO reference checkout is not pinned v0.6")
    assets,output=Path(args.assets).resolve(),Path(args.output).resolve()
    assets.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    # Download only the public author-linked files required for this phase.
    needed=["train.npz"] if args.phase=="pretrain" else ["state_3000.pt","normalization.npz"]
    for name in needed:
        path=assets/name
        if not path.exists():
            import gdown
            gdown.download(id=FILES[name],output=str(path),quiet=False)
    source="cfg/gym/pretrain/hopper-medium-v2" if args.phase=="pretrain" else "cfg/gym/finetune/hopper-v2"
    config="pre_diffusion_mlp" if args.phase=="pretrain" else "ft_ppo_diffusion_mlp"
    command=[sys.executable,str(root/"scripts/dppo_source_entry.py"),f"--config-name={config}",f"--config-dir={source}",
             "wandb=null",f"seed={args.seed}",f"logdir={output}"]
    if args.phase=="pretrain":
        command += [f"train_dataset_path={assets/'train.npz'}"]
        if args.epochs is not None:
            command += [f"train.n_epochs={args.epochs}"]
    else:
        checkpoint=Path(args.checkpoint).resolve() if args.checkpoint else assets/'state_3000.pt'
        command += [f"base_policy_path={checkpoint}",f"normalization_path={assets/'normalization.npz'}"]
        if args.iterations is not None:
            command += [f"train.n_train_itr={args.iterations}"]
        if args.phase=="evaluate":
            command += ["_target_=agent.eval.eval_diffusion_agent.EvalDiffusionAgent","+n_steps=500","+render_num=0"]
    metadata={"method":"bc" if args.phase=="pretrain" else "dppo_online", "profile":"official_code_v0.6",
        "upstream_commit":COMMIT,"task":"hopper-medium-v2","phase":args.phase,"seed":args.seed,
        "config_source":source+"/"+config+".yaml","command":command,
        "budget_override":{"iterations":args.iterations,"epochs":args.epochs},
        "full_budget":args.phase!="evaluate" and args.iterations is None and args.epochs is None,
        "full_evaluation_budget":args.phase=="evaluate",
        "assets":{name:{"path":str(assets/name),"sha256":hashlib.sha256((assets/name).read_bytes()).hexdigest(),
                        "bytes":(assets/name).stat().st_size,"drive_id":FILES[name]} for name in needed},
        "instrumentation":"physical step counting; BC epoch-boundary checkpoint hooks; original algorithm unchanged",
        "resume_support":"exact BC epoch boundary" if args.phase=="pretrain" else "original online resume unsupported",
        "resume_from":str(Path(args.resume).resolve()) if args.resume else None,
        "status":"prepared"}
    if args.checkpoint:
        metadata["evaluation_checkpoint"]={"path":str(checkpoint),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    metadata_path=output/"original_manifest.json"
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    if args.prepare_only:
        print(metadata_path)
        return
    if args.phase=="pretrain":
        sys.path.insert(0,str(root))
        from doublemdp.agents.dppo_original_bc import prepare_bc_source
        patch=root/"scripts/dppo_bc_resume.patch"
        upstream=prepare_bc_source(upstream,assets.parent/"runtime/dppo_bc_resume_v1",patch)
        metadata["source_patch_sha256"]=hashlib.sha256(patch.read_bytes()).hexdigest()
    env=os.environ.copy()
    env.update(DPPO_DATA_DIR=str(assets),DPPO_LOG_DIR=str(output),DPPO_WANDB_ENTITY="unused",
        D4RL_SUPPRESS_IMPORT_ERROR="1",WANDB_MODE="disabled",OMP_NUM_THREADS="1",
        DPPO_COUNT_PHYSICAL_STEPS="1" if args.phase!="pretrain" else "0",
        DPPO_UPSTREAM_ROOT=str(upstream), DPPO_BC_RESUME=str(Path(args.resume).resolve()) if args.resume else "",
        PYTHONPATH=str(root)+os.pathsep+str(upstream)+os.pathsep+env.get("PYTHONPATH",""))
    metadata["status"]="running"
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    with (output/"source_run.log").open("w") as log:
        result=subprocess.run(command,cwd=upstream,env=env,stdout=log,stderr=subprocess.STDOUT)
    metadata["status"]="complete" if result.returncode==0 else "failed"
    metadata["returncode"]=result.returncode
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    raise SystemExit(result.returncode)


if __name__=="__main__":
    main()
