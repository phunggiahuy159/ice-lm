"""Run ice_lm on Modal (https://modal.com): fast GPUs, a persistent volume for data + checkpoints.

One-time setup on your machine (keep the token out of files and git):
    pip install modal
    modal token set --token-id <id> --token-secret <secret>

Usage (run from this folder):
    modal run modal_app.py::tests                           # unit tests + GDN kernel check on a GPU
    modal run modal_app.py::smoke                           # tiny arms B vs C on an offline toy corpus
    modal run modal_app.py::prepare --train-tokens 2.5e9    # download + tokenize into the volume (CPU only)
    modal run modal_app.py::day0                            # Day 0 matcher statistics (CPU only)
    modal run --detach modal_app.py::train --preset 0.6b --arm C --gpu H100:1
    modal run modal_app.py::evaluate --runs 0.6b_A,0.6b_B,0.6b_C
    modal volume ls ice-lm runs/                            # inspect / download results
    modal volume get ice-lm results_0.6b.json .

`ICE_CPU_ONLY=1 modal run modal_app.py::tests_cpu` / `::smoke_cpu` run without any GPU.
`--detach` keeps training running after you close the terminal. Re-running the same `train` command
resumes from the last checkpoint in the volume (and Modal retries a crashed container automatically).
"""
import os
import shutil
import subprocess
import threading

import modal

ROOT = os.path.dirname(os.path.abspath(__file__))
# ICE_CPU_ONLY=1 turns every GPU request into CPU (for accounts without a payment method, or quick checks)
CPU_ONLY = os.environ.get("ICE_CPU_ONLY", "0") == "1"


def GPU(spec):
    return None if CPU_ONLY else spec
CODE = "/root/ice_lm"
VOL = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "numpy", "transformers>=4.45", "datasets>=3.0", "einops", "pytest", "matplotlib",
                 "nbconvert", "ipykernel", "wandb")
    # fast GDN kernels; the code falls back to its PyTorch kernel if these are missing or broken
    # (0.4.2 verified on an RTX 3090; newer releases may need a newer triton than torch 2.6 ships)
    .run_commands("pip install --no-deps fla-core==0.4.2 flash-linear-attention==0.4.2 || true")
    # the container re-imports this file: give it the same ICE_WANDB so its function definitions (secrets)
    # match the ones deployed from this machine
    .env({"ICE_WANDB": os.environ.get("ICE_WANDB", "1")})
    .add_local_dir(os.path.join(ROOT, "ice_lm"), f"{CODE}/ice_lm")
    .add_local_dir(os.path.join(ROOT, "scripts"), f"{CODE}/scripts")
    .add_local_dir(os.path.join(ROOT, "tests"), f"{CODE}/tests")
    .add_local_dir(os.path.join(ROOT, "notebooks"), f"{CODE}/notebooks")
)
vol = modal.Volume.from_name("ice-lm", create_if_missing=True)
# W&B logging: `modal secret create wandb WANDB_API_KEY=<key>` once; set ICE_WANDB=0 to train without it
WANDB = [modal.Secret.from_name("wandb")] if os.environ.get("ICE_WANDB", "1") == "1" else []
app = modal.App("ice-lm", image=image)


def _run(cmd: str, commit_every: float = 0):
    """Run a shell command in the code folder; commit the volume periodically and at the end."""
    print("+", cmd, flush=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(commit_every):
            try:
                vol.commit()
            except Exception as e:  # noqa: BLE001
                print("[modal] volume commit failed:", e, flush=True)

    if commit_every:
        threading.Thread(target=committer, daemon=True).start()
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=CODE)
    finally:
        stop.set()
        vol.commit()


# ---------------------------------------------------------------------------- remote functions
@app.function(gpu=GPU("T4"), volumes={VOL: vol}, timeout=3600)
def _tests():
    _run("python -m pytest -q tests")
    _run("python -c \"import torch; from ice_lm.utils import pick_precision; from ice_lm.gdn import resolve_backend; "
         "d=torch.device('cuda'); amp,_=pick_precision('auto', d); print(torch.cuda.get_device_name(0), amp, "
         "'GDN backend:', resolve_backend('auto', d, amp, 128))\"")


@app.function(cpu=8.0, memory=16384, volumes={VOL: vol}, timeout=3600)
def _tests_cpu():
    _run("python -m pytest -q tests")


@app.function(cpu=8.0, memory=16384, volumes={VOL: vol}, timeout=3600)
def _smoke_cpu():
    _smoke.local()


@app.function(cpu=8.0, memory=16384, volumes={VOL: vol}, timeout=3600)
def _ddp_cpu():
    if not os.path.exists(f"{VOL}/data_synth/meta.json"):
        _run(f"python scripts/prepare_data.py --synthetic --out {VOL}/data_synth")
    base = (f"torchrun --standalone --nproc_per_node=2 scripts/train.py --preset smoke --arm E "
            f"--data_dir {VOL}/data_synth --out_dir /tmp/ddp_E --total_tokens 400000 --eval_every 60 "
            f"--log_every 20 --num_workers 1")
    _run(base + " --max_hours 0.004")      # ~15 s: must stop cleanly and checkpoint
    _run(base)                            # must resume from the checkpoint and finish
    _run(base)                            # finished run: must say so and not retrain
    _run("ls -la /tmp/ddp_E && tail -3 /tmp/ddp_E/log.jsonl")


@app.function(cpu=8.0, memory=32768, volumes={VOL: vol}, timeout=3 * 3600)
def _notebook_cpu():
    """Dev check: execute the Kaggle notebook end-to-end on CPU with tiny settings (real data download)."""
    import json
    nb = json.load(open(f"{CODE}/notebooks/ice_kaggle.ipynb"))
    overrides = {"PRESET": '"smoke"', "ARM": '"C"', "TOTAL_TOKENS": "300_000", "TRAIN_DATA_TOKENS": "3_000_000",
                 "MAX_HOURS": "1.0", "SEQ_LEN": "256", "WORK": '"/tmp/kaggle_working"'}
    for c in nb["cells"]:
        if c["cell_type"] == "code" and any(l.startswith("PRESET ") for l in c["source"]):
            c["source"] = [(f"{l.split('=')[0].rstrip()} = {overrides[l.split('=')[0].strip()]}\n"
                            if "=" in l and l.split("=")[0].strip() in overrides else l) for l in c["source"]]
    # evaluate at short context only (CPU)
    for c in nb["cells"]:
        c["source"] = [l.replace("--n_task 30 --val_batches 500",
                                 "--n_task 5 --val_batches 5 --context_lens 512 1024 --coderecall_gaps 64 256 "
                                 "--n_coderecall 5")
                        .replace("--contexts 4096 16384 32768 --decode_steps 64", "--contexts 256 1024 --decode_steps 8")
                       for l in c["source"]]
    os.makedirs("/tmp/kaggle_working", exist_ok=True)
    json.dump(nb, open("/tmp/nb_in.ipynb", "w"))
    r = subprocess.run("jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 "
                       "--output /tmp/nb_out.ipynb /tmp/nb_in.ipynb", shell=True, cwd="/tmp")
    out = json.load(open("/tmp/nb_out.ipynb")) if os.path.exists("/tmp/nb_out.ipynb") else nb
    for i, c in enumerate(out["cells"]):
        if c["cell_type"] != "code" or (c["source"] and c["source"][0].startswith("%%writefile")):
            continue
        print(f"\n===== cell {i}: {c['source'][0].strip()[:80]}")
        for o in c.get("outputs", []):
            txt = o.get("text") or o.get("data", {}).get("text/plain") or ""
            if o.get("output_type") == "error":
                txt = "ERROR " + o.get("ename", "") + ": " + o.get("evalue", "") + "\n" + "\n".join(o.get("traceback", [])[-5:])
            txt = "".join(txt) if isinstance(txt, list) else txt
            print(txt[-4000:])
    print("nbconvert exit code:", r.returncode)
    if os.path.isdir("/tmp/kaggle_working/figs"):             # keep the figures + results for inspection
        shutil.copytree("/tmp/kaggle_working/figs", f"{VOL}/nb_figs", dirs_exist_ok=True)
        for f in ("results_smoke.json", "benchmark_smoke.json"):
            if os.path.exists(f"/tmp/kaggle_working/{f}"):
                shutil.copy(f"/tmp/kaggle_working/{f}", f"{VOL}/nb_figs/{f}")
        vol.commit()


@app.function(gpu=GPU("T4"), volumes={VOL: vol}, timeout=3600)
def _smoke():
    _run(f"python scripts/prepare_data.py --synthetic --out {VOL}/data_synth")
    for arm in ("B", "C"):
        _run(f"python scripts/train.py --preset smoke --arm {arm} --data_dir {VOL}/data_synth "
             f"--out_dir {VOL}/smoke_runs/{arm} --total_tokens 600000 --eval_every 100 --log_every 50 --resume false")


@app.function(cpu=8.0, memory=16384, volumes={VOL: vol}, timeout=24 * 3600)
def _prepare(train_tokens: float, val_tokens: float, code_frac: float):
    _run(f"python scripts/prepare_data.py --out {VOL}/data --train_tokens {train_tokens} "
         f"--val_tokens {val_tokens} --code_frac {code_frac}", commit_every=600)


@app.function(cpu=4.0, volumes={VOL: vol}, timeout=3600)
def _day0():
    _run(f"python scripts/day0_stats.py --data_dir {VOL}/data --windows 200 --out {VOL}/day0.json")


@app.function(gpu=GPU("H100"), volumes={VOL: vol}, timeout=24 * 3600, retries=2, cpu=8.0, memory=65536,
              secrets=WANDB)
def _train(preset: str, arm: str, total_tokens: float, n_gpus: int, extra: str, name: str = ""):
    import torch
    # copy the token files to local disk: random-access reads are faster than on the network volume
    _run(f"mkdir -p /tmp/data && cp -n {VOL}/data/*.bin {VOL}/data/meta.json /tmp/data/")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if gpu_mem >= 75:      # H100 / A100-80GB: no activation checkpointing needed
        micro, ckpt = {"0.6b": 4, "140m": 8}.get(preset, 8), "false"
    elif gpu_mem >= 40:    # L40S / A100-40GB
        micro, ckpt = {"0.6b": 2, "140m": 4}.get(preset, 8), ("true" if preset == "0.6b" else "false")
    else:                  # 16-24 GB cards
        micro, ckpt = {"0.6b": 1, "140m": 2}.get(preset, 8), "true"
    launcher = f"torchrun --standalone --nproc_per_node={n_gpus}" if n_gpus > 1 else "python"
    tok = f"--total_tokens {int(total_tokens)}" if total_tokens else ""
    _run(f"{launcher} scripts/train.py --preset {preset} --arm {arm} --data_dir /tmp/data "
         f"--out_dir {VOL}/runs/{name or preset + '_' + arm} {tok} --micro_batch {micro} --grad_checkpointing {ckpt} "
         f"--num_workers 6 --max_hours 23.5 {extra}", commit_every=300)


@app.function(gpu=GPU("H100"), volumes={VOL: vol}, timeout=6 * 3600)
def _evaluate(runs: str, n_task: int, val_batches: int, out: str, extra: str):
    paths = " ".join(f"{VOL}/runs/{r}" for r in runs.split(","))
    _run(f"python scripts/evaluate.py --runs {paths} --data_dir {VOL}/data --n_task {n_task} "
         f"--val_batches {val_batches} --out {VOL}/{out} {extra}")
    _run(f"python scripts/plot_results.py --results {VOL}/{out} --out {VOL}/figs_{os.path.splitext(out)[0]}")


# ---------------------------------------------------------------------------- local entrypoints
@app.function(cpu=1.0, volumes={VOL: vol}, timeout=12 * 3600)
def _evaluate_when_done(runs: str, n_task: int, val_batches: int, out: str, extra: str):
    """Waits (on a cheap CPU container) until every run has model_final.pt, then runs _evaluate."""
    import time
    while True:
        vol.reload()
        missing = [r for r in runs.split(",") if not os.path.exists(f"{VOL}/runs/{r}/model_final.pt")]
        if not missing:
            break
        print(f"waiting for {missing}", flush=True)
        time.sleep(120)
    _evaluate.remote(runs, n_task, val_batches, out, extra)


@app.function(cpu=8.0, memory=32768, volumes={VOL: vol}, timeout=6 * 3600, secrets=WANDB)
def _cpu_cmd(cmd: str):
    _run(cmd, commit_every=300)


@app.function(gpu=GPU("H100"), volumes={VOL: vol}, timeout=6 * 3600, cpu=8.0, memory=65536)
def _gpu_cmd(cmd: str):
    _run(cmd, commit_every=300)


@app.local_entrypoint()
def tests():
    _tests.remote()


@app.local_entrypoint()
def smoke():
    _smoke.remote()


@app.local_entrypoint()
def ddp_cpu():
    """2-process DDP on CPU (gloo): train, stop at a time limit, resume, finish."""
    _ddp_cpu.remote()


@app.local_entrypoint()
def notebook_cpu():
    """Execute the Kaggle notebook end-to-end on a CPU container with tiny settings."""
    _notebook_cpu.remote()


@app.local_entrypoint()
def tests_cpu():
    """Unit tests on a CPU container (no GPU / payment method needed)."""
    _tests_cpu.remote()


@app.local_entrypoint()
def smoke_cpu():
    """Tiny B vs C smoke run on a CPU container."""
    _smoke_cpu.remote()


@app.local_entrypoint()
def prepare(train_tokens: float = 2.5e9, val_tokens: float = 2e6, code_frac: float = 0.3):
    _prepare.remote(train_tokens, val_tokens, code_frac)


@app.local_entrypoint()
def day0():
    _day0.remote()


@app.local_entrypoint()
def train(preset: str = "0.6b", arm: str = "C", total_tokens: float = 0, gpu: str = "H100", extra: str = "",
          name: str = ""):
    """gpu: e.g. "H100", "H100:8", "A100-80GB", "L40S". total_tokens=0 uses the preset default.
    name: run folder under runs/ (default <preset>_<arm>), e.g. to keep a shorter-budget run apart."""
    n_gpus = int(gpu.split(":")[1]) if ":" in gpu else 1
    fn = _train.with_options(gpu=gpu) if (hasattr(_train, "with_options") and not CPU_ONLY) else _train
    fn.remote(preset, arm, total_tokens, n_gpus, extra, name)


@app.local_entrypoint()
def evaluate(runs: str = "0.6b_A,0.6b_B,0.6b_C", n_task: int = 50, val_batches: int = 50, out: str = "",
             extra: str = ""):
    """extra: more scripts/evaluate.py flags, e.g. "--n_coderecall 60". Figures go to figs_<out name>/."""
    _evaluate.remote(runs, n_task, val_batches, out or f"results_{runs.split(',')[0].split('_')[0]}.json", extra)


@app.local_entrypoint()
def evaluate_when_done(runs: str = "140m_B,140m_C,140m_C_notrust", n_task: int = 50, val_batches: int = 500,
                       out: str = "results_140m.json", extra: str = ""):
    """modal run --detach modal_app.py::evaluate_when_done : evaluates as soon as the training runs finish."""
    _evaluate_when_done.remote(runs, n_task, val_batches, out, extra)


@app.local_entrypoint()
def run(cmd: str, gpu: bool = False):
    """Run any shell command in the code folder (volume at /vol), e.g.
    modal run modal_app.py::run --cmd "python scripts/day0_stats.py --data_dir /vol/data" [--gpu]"""
    (_gpu_cmd if gpu else _cpu_cmd).remote(cmd)
