"""Build ice_lm/notebooks/ice_eval_kaggle.ipynb with the committed ice_lm code embedded (git archive HEAD)."""
import base64
import os
import json
import subprocess
import sys
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{REPO}/ice_lm/notebooks/ice_eval_kaggle.ipynb"

commit = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
tar = subprocess.check_output(["git", "-C", REPO, "archive", "--format=tar.gz", "HEAD", "ice_lm",
                               ":(exclude)ice_lm/notebooks"])
b64 = base64.b64encode(tar).decode()
print(f"commit {commit}, bundle {len(tar)/1e3:.0f} kB", file=sys.stderr)

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=textwrap.dedent(s).strip("\n")))


def code(s):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=textwrap.dedent(s).strip("\n")))


md(f"""
# ICE round 2 on Kaggle: re-evaluation, ICE-V2 fits, new metrics

Runs the GPU part of round 2 (`research/round2/decision2.md`) that was blocked by the Modal spend limit. Code:
commit `{commit}` of `huypg-zenai/ice-lm`, embedded below, so no GitHub access is needed.

| part | what | why |
|---|---|---|
| `refs` | re-evaluate B, G, G_VR on this GPU | every comparison below is paired on the same hardware (T4, fp32) |
| `gate` | evaluate the old VR scorer refitted on G_VR's frozen backbone | run 1: must match G_VR within 0.005, else the fitter is broken |
| `recall` | repeat / MQAR with full-vocab, frequent and natural tokens | is the hybrid's 0.00 on synthetic recall real or a task artefact? |
| `linecomp` | greedy line completion, exact match, on held-out code | the code use case that loss does not measure |
| `v2` | ICE-V2 scorer fits on frozen G_VR / G / B (+ synthetic) and their evaluation | runs 2-5 of decision2 (pre-registered predictions printed at the end) |

**Kaggle settings:** *Accelerator* **GPU T4 x2**, *Internet* **On**. Then **Save Version → Save & Run All (Commit)**:
it runs in the background (up to 12 h) and keeps the outputs. Expect roughly 6-10 h for everything on 2 x T4;
jobs run in priority order, and no new job starts after `MAX_HOURS`.

**Inputs (pick one):**
* **Modal (default):** *Add-ons → Secrets*: add `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (workspace
  tien-phat140205) and attach them to this notebook. The notebook downloads the checkpoints and data from the
  `ice-lm` volume (read only) and uploads its results to `kaggle_results/` on that volume. No Modal compute is used.
* **Kaggle dataset:** upload a dataset holding `runs/<name>/config.json` + `model_final.pt` for t200_B, t200_G,
  t200_G_VR, fit_GVRbb_vr, and `data/` with meta.json, val_code.bin, val_prose.bin, train.bin; attach it.
  The notebook finds it under `/kaggle/input` (set `SOURCE = "dataset"`).

**Outputs:** `/kaggle/working/ice_results/` (one JSON per job, `logs/`, `SUMMARY.md`) and `ice_results.zip`.
""")

md("## 1. Config")
code("""
PARTS = dict(tests=True, refs=True, gate=True, recall=True, linecomp=True, v2=True)
V2_FEATS     = "dnrq"   # ICE-V2 tables: d margin, n agreement, r recency, q backbone rank (F2 may drop d/n/r)
FIT_TOKENS   = 2e7      # tokens per scorer fit (decision2: 20M)
VAL_BATCHES  = 500      # held-out windows per split, as in results_t200
N_CODERECALL = 100      # items per CodeRecall category and gap
N_TASK       = 50       # items per synthetic task and length
N_SYN        = 50       # items per synthetic conflict template family and gap
LINE_WINDOWS = 100      # held-out code windows for line completion (3 lines each)
PRECISION    = "fp32"   # T4 has no bf16; "auto" would use fp16 (faster, but untested for these bf16-trained models)
MAX_HOURS    = 11.0     # no new job starts after this (Kaggle stops a session at 12 h)
SOURCE       = "auto"   # "modal" | "dataset" | "auto" (dataset if one is attached, else Modal)
UPLOAD_TO_MODAL = True  # copy results to the ice-lm volume (kaggle_results/) after each job
EXTRA_FIT_ARGS, EXTRA_LINE_ARGS = "", ""   # extra flags (dry runs)

import os, time
T_START = time.time()
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")
CODE, INP, RES = f"{WORK}/ice_lm", f"{WORK}/inputs", f"{WORK}/ice_results"
RUNS_IN, RUNS_FIT, DATA = f"{INP}/runs", f"{WORK}/fits", f"{INP}/data"
for d in (INP, RES, f"{RES}/logs", RUNS_FIT):
    os.makedirs(d, exist_ok=True)
""")

md("## 2. Setup")
code("""
import subprocess, sys, torch
def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print((r.stdout + r.stderr)[-3000:], flush=True)
    if check and r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")
    return r
sh("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "| GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(N_GPU)])
# optional fast GDN kernels; the code self-tests them and falls back to its exact PyTorch kernel if they fail
sh("pip install -q --no-deps fla-core==0.4.2 flash-linear-attention==0.4.2", check=False)
sh("pip install -q 'transformers>=4.40' pytest", check=False)
""")

md(f"## 3. Code (commit `{commit}`)")
code(f"""
import base64
import os, io, tarfile, shutil
BUNDLE = "{b64}"
shutil.rmtree(CODE, ignore_errors=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as tf:
    tf.extractall(WORK)
print("code at", CODE, "| commit {commit}")
""")

md("## 4. Inputs")
code("""
import glob, json
NEED_RUNS = sorted({*(["t200_B", "t200_G", "t200_G_VR"] if PARTS["refs"] or PARTS["v2"] or PARTS["recall"] or PARTS["linecomp"] else []),
                    *(["fit_GVRbb_vr"] if PARTS["gate"] else [])})
NEED_DATA = ["meta.json", "val_code.bin", "val_prose.bin"] + (["train.bin"] if PARTS["v2"] else [])

def find_dataset():
    if os.environ.get("ICE_DATASET"):          # local dry runs
        return os.environ["ICE_DATASET"]
    for root in glob.glob("/kaggle/input/*"):
        hits = glob.glob(f"{root}/**/data/meta.json", recursive=True)
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
    return None

ds = find_dataset() if SOURCE in ("auto", "dataset") else None
USE_MODAL = ds is None
if SOURCE == "dataset" and ds is None:
    raise RuntimeError("SOURCE='dataset' but no attached dataset has data/meta.json")
if USE_MODAL or UPLOAD_TO_MODAL:
    try:
        from kaggle_secrets import UserSecretsClient
        sec = UserSecretsClient()
        os.environ["MODAL_TOKEN_ID"] = sec.get_secret("MODAL_TOKEN_ID")
        os.environ["MODAL_TOKEN_SECRET"] = sec.get_secret("MODAL_TOKEN_SECRET")
        sh("pip install -q modal", check=False)
    except Exception as e:   # no secrets: only the dataset route works, and results stay on Kaggle
        print("Modal secrets not available:", type(e).__name__, e)
        if USE_MODAL:
            raise RuntimeError("attach the MODAL_TOKEN_ID / MODAL_TOKEN_SECRET secrets or a dataset (see top)")
        UPLOAD_TO_MODAL = False

def remote_size(rel):
    r = subprocess.run(f"modal volume ls --json ice-lm {os.path.dirname(rel)}", shell=True, capture_output=True, text=True)
    return next((f["size"] for f in json.loads(r.stdout or "[]") if f["filename"] == rel), None)

def fetch(rel):
    # download (Modal; size-checked, 4 tries: large downloads can time out) or link (dataset) one file
    dst = f"{INP}/{rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not USE_MODAL:
        if not os.path.exists(dst):
            os.symlink(f"{ds}/{rel}", dst)
        return
    want = remote_size(rel)
    def ok():
        if not os.path.exists(dst) or (want is not None and os.path.getsize(dst) != want):
            return False
        if dst.endswith(".pt"):                # a retry over a partial file can leave a right-sized corrupt file
            import zipfile
            try:
                return zipfile.ZipFile(dst).testzip() is None
            except Exception:
                return False
        return True
    for attempt in range(4):
        if ok():
            return
        if os.path.exists(dst):
            os.remove(dst)                     # never download over a partial file
        sh(f"modal volume get ice-lm {rel} {os.path.dirname(dst)}/", check=False)
    if ok():
        return
    raise RuntimeError(f"download of {rel} failed or is incomplete ({os.path.getsize(dst) if os.path.exists(dst) else 0} of {want} bytes)")

for f in NEED_DATA:
    fetch(f"data/{f}")
for r in NEED_RUNS:
    for f in ("config.json", "model_final.pt"):
        fetch(f"runs/{r}/{f}")
print(json.load(open(f"{DATA}/meta.json")))
sh(f"du -sh {INP}/runs/* {INP}/data/*", check=False)
""")

md("## 5. Tests (CPU suite + GPU kernel check)")
code("""
if PARTS["tests"]:
    sh(f"cd {CODE} && ICE_CPU_ONLY=1 python -m pytest -q -x tests 2>&1 | tail -5")
    sh(f'''cd {CODE} && python -c "import torch; from ice_lm.gdn import resolve_backend; d=torch.device('cuda'); print(torch.cuda.get_device_name(0), 'GDN backend fp32:', resolve_backend('auto', d, torch.float32, 64), '| fp16:', resolve_backend('auto', d, torch.float16, 64))"''', check=False)
""")

md("""
## 6. Job runner
Each *chain* is a list of jobs run in order on one GPU (a fit, then its evaluation, ...). Chains are taken in
priority order by one worker per GPU. A job is skipped if its output exists (or every later job of its chain is
done), so re-running the notebook resumes. Logs: `ice_results/logs/<job>.log`.
""")
code("""
import threading, queue, datetime
UPLOAD_LOCK = threading.Lock()
STATUS = {}

def upload_results():
    if not UPLOAD_TO_MODAL:
        return
    with UPLOAD_LOCK:
        subprocess.run(f"modal volume put ice-lm {RES} kaggle_results/latest --force", shell=True,
                       capture_output=True, text=True)

def run_chain(chain, gpu):
    name, jobs = chain
    for i, (jname, cmd, out) in enumerate(jobs):
        later = [o for _, _, o in jobs[i:]]
        if all(os.path.exists(o) for o in later):
            STATUS[jname] = "done (cached)"
            continue
        if (time.time() - T_START) / 3600 > MAX_HOURS:
            STATUS[jname] = "skipped (MAX_HOURS)"
            return
        STATUS[jname] = f"running on GPU {gpu}"
        t0 = time.time()
        with open(f"{RES}/logs/{jname}.log", "w") as log:
            rc = subprocess.run(f"cd {CODE} && CUDA_VISIBLE_DEVICES={gpu} {cmd}", shell=True, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        dt = (time.time() - t0) / 60
        if rc or not os.path.exists(out):
            STATUS[jname] = f"FAILED rc={rc} after {dt:.0f} min (see logs/{jname}.log)"
            return                      # later jobs of this chain depend on this one
        STATUS[jname] = f"done in {dt:.0f} min"
        upload_results()

def run_all(chains):
    q = queue.Queue()
    for c in chains:
        q.put(c)
    def worker(gpu):
        while True:
            try:
                c = q.get_nowait()
            except queue.Empty:
                return
            run_chain(c, gpu)
    threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, N_GPU))]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(120)
        el = (time.time() - T_START) / 3600
        print(f"--- {datetime.datetime.now():%H:%M} ({el:.1f} h) " +
              " | ".join(f"{k}: {v}" for k, v in STATUS.items() if not v.startswith("done (cached)")), flush=True)
    for t in threads:
        t.join()
    print("\\n".join(f"{k:28s} {v}" for k, v in STATUS.items()))
""")

md("""
## 7. Jobs
All evaluations use the same settings, so every number in the summary is paired (same items, same GPU type).
ICE-off controls are skipped (`--no_ice_off`): for the frozen-backbone fits ICE-off is exactly the base model.
""")
code("""
P = PRECISION
EVAL = (f"python scripts/evaluate.py --data_dir {DATA} --val_batches {VAL_BATCHES} --n_task {N_TASK} "
        f"--n_coderecall {N_CODERECALL} --n_syn {N_SYN} --precision {P} --no_ice_off")
LINE = f"python scripts/line_completion.py --data_dir {DATA} --windows {LINE_WINDOWS} --precision {P} {EXTRA_LINE_ARGS}"
FIT = (f"python scripts/fit_scorer.py --data_dir {DATA} --tokens {FIT_TOKENS} --micro_batch 2 --accum 4 "
       f"--num_workers 2 --precision {P} {EXTRA_FIT_ARGS}")

def run_dir(name):
    return f"{RUNS_IN}/{name}" if os.path.isdir(f"{RUNS_IN}/{name}") else f"{RUNS_FIT}/{name}"

def eval_chain(name, fit_cmd=None):
    jobs = []
    if fit_cmd:
        jobs.append((f"fit_{name}", f"{FIT} {fit_cmd} --out_dir {RUNS_FIT}/{name}", f"{RUNS_FIT}/{name}/model_final.pt"))
    rd = f"{RUNS_FIT}/{name}" if fit_cmd else f"{RUNS_IN}/{name}"
    jobs.append((f"eval_{name}", f"{EVAL} --runs {rd} --out {RES}/eval_{name}.json", f"{RES}/eval_{name}.json"))
    if PARTS["linecomp"]:
        jobs.append((f"line_{name}", f"{LINE} --runs {rd} --out {RES}/line_{name}.json", f"{RES}/line_{name}.json"))
    return (name, jobs)

V2 = f"--cand vr --v2 --v2_feats {V2_FEATS} --listwise 0.25"
GVR, G, B = f"{RUNS_IN}/t200_G_VR", f"{RUNS_IN}/t200_G", f"{RUNS_IN}/t200_B"
chains = []                                                    # priority order
if PARTS["refs"] or PARTS["gate"]:
    chains.append(eval_chain("t200_G_VR"))
if PARTS["gate"]:
    chains.append(eval_chain("fit_GVRbb_vr"))
if PARTS["refs"]:
    chains.append(eval_chain("t200_B"))
if PARTS["recall"]:
    chains.append(("recall", [("recall_check", f"python scripts/recall_check.py --runs {B} {G} {GVR} --data_dir {DATA} "
                                f"--precision {P} --out {RES}/recall_check.json", f"{RES}/recall_check.json")]))
if PARTS["v2"]:
    chains.append(eval_chain("fit_GVRbb_v2", f"--base_run {GVR} {V2} --init_from {GVR} --seed 7"))     # run 2
if PARTS["refs"]:
    chains.append(eval_chain("t200_G"))
if PARTS["v2"]:
    chains += [eval_chain("fit_Gbb_vr", f"--base_run {G} --cand vr --seed 7"),                         # run 3, old
               eval_chain("fit_Gbb_v2", f"--base_run {G} {V2} --seed 7"),                              # run 3, V2
               eval_chain("fit_Bbb_v2", f"--base_run {B} {V2} --seed 7"),                              # run 4
               eval_chain("fit_GVRbb_v2syn", f"--base_run {GVR} {V2} --init_from {GVR} --syn_frac 0.02 --seed 7"),  # run 5
               eval_chain("fit_GVRbb_v2_s8", f"--base_run {GVR} {V2} --init_from {GVR} --seed 8"),     # run 2, seed 2
               eval_chain("fit_GVRbb_v2q", f"--base_run {GVR} --cand vr --v2 --v2_feats q --listwise 0.25 "
                                           f"--init_from {GVR} --seed 7")]                              # F2 on the model
for name, jobs in chains:
    print(name, "->", [j for j, _, _ in jobs])
""")

md("## 8. Run (this is the long cell)")
code("""
run_all(chains)
""")

md("## 9. Summary: tables, gate and pre-registered predictions")
code("""
evals = sorted(glob.glob(f"{RES}/eval_*.json"))
if evals:
    sh(f"cd {CODE} && python scripts/evaluate.py --from_json {' '.join(evals)} --data_dir {DATA} --reference B "
       f"--n_syn {N_SYN} --out {RES}/all_eval.json > {RES}/all_eval_tables.txt 2>&1; tail -c 20000 {RES}/all_eval_tables.txt", check=False)

def load_eval(name):
    p = f"{RES}/eval_{name}.json"
    return next(iter(json.load(open(p)).values())) if os.path.exists(p) else None

merged = json.load(open(f"{RES}/all_eval.json")) if os.path.exists(f"{RES}/all_eval.json") else {}
def E(name):                       # merged result (has the CodeRecall summaries cr/<category>)
    return next((v for k, v in merged.items() if k.split("#")[0].rstrip("/").endswith("/" + name)), None) or load_eval(name)

KEYS = [("code", "code/all_loss"), ("copyable", "code/copyable_loss"), ("copy>512", "code/copy_far_loss"),
        ("lk_wrong", "code/lk_wrong_loss"), ("follow", "code/follow_wrong_lookup"), ("prose", "prose/all_loss"),
        ("acc", "code/all_acc"), ("cr g1", "cr/conflict_g1"), ("cr g2", "cr/conflict_g2"), ("cr g5", "cr/conflict_g5"),
        ("cr ovr_diff", "cr/override_diff"), ("cr copy_same", "cr/copy_same"),
        ("syn fit", "syn/fit_mean"), ("syn held", "syn/heldout_mean")]
names = ["t200_B", "t200_G", "t200_G_VR", "fit_GVRbb_vr", "fit_GVRbb_v2", "fit_GVRbb_v2_s8", "fit_GVRbb_v2q",
         "fit_Gbb_vr", "fit_Gbb_v2", "fit_Bbb_v2", "fit_GVRbb_v2syn"]
lines = ["| run | " + " | ".join(k for k, _ in KEYS) + " | line em | line em (repeat/reuse/novel) |",
         "|---" * (len(KEYS) + 3) + "|"]
for n in names:
    r = E(n)
    if r is None:
        continue
    lc = json.load(open(f"{RES}/line_{n}.json"))["results"] if os.path.exists(f"{RES}/line_{n}.json") else None
    lc = next(iter(lc.values())) if lc else None
    cells = [f"{r[k]:.3f}" if isinstance(r.get(k), float) else "-" for _, k in KEYS]
    le = f"{lc['all/em']:.3f}" if lc else "-"
    lk = "/".join(f"{lc[k + '/em']:.2f}" for k in ("repeat", "reuse", "novel")) if lc else "-"
    lines.append(f"| {n} | " + " | ".join(cells) + f" | {le} | {lk} |")

def d(a, b, k="code/all_loss"):
    ra, rb = E(a), E(b)
    return None if ra is None or rb is None else ra[k] - rb[k]

checks = []
def check(label, val, ok, pred):
    checks.append(f"| {label} | {'-' if val is None else f'{val:+.4f}' if isinstance(val, float) else val} | {pred} | "
                  f"{'n/a' if val is None else 'PASS' if ok(val) else 'FAIL'} |")
check("gate: refit old VR scorer - G_VR (code)", d("fit_GVRbb_vr", "t200_G_VR"), lambda v: abs(v) <= 0.005, "within ±0.005")
check("run 2: V2 on G_VR bb - G_VR (code)", d("fit_GVRbb_v2", "t200_G_VR"), lambda v: -0.010 <= v <= -0.003, "-0.003 to -0.010")
check("run 2 seed 8 - G_VR (code)", d("fit_GVRbb_v2_s8", "t200_G_VR"), lambda v: -0.010 <= v <= -0.003, "-0.003 to -0.010")
r2 = E("fit_GVRbb_v2")
check("run 2: follow wrong lookup", r2 and r2["code/follow_wrong_lookup"], lambda v: v <= 0.31, "<= 0.31")
check("run 2: CodeRecall conflict_g1", r2 and r2.get("cr/conflict_g1"), lambda v: v >= 0.50, ">= 0.50")
for n in ("fit_Gbb_vr", "fit_Gbb_v2"):
    r = E(n)
    check(f"run 3: {n} code loss", r and r["code/all_loss"], lambda v: 1.705 <= v <= 1.735, "1.705-1.735 (H100 scale)")
check("run 4: V2 on B bb - B (code)", d("fit_Bbb_v2", "t200_B"), lambda v: -0.02 <= v <= -0.005, "-0.005 to -0.02")
rs, rb = E("fit_GVRbb_v2syn"), E("t200_B")
for cat, thr in (("conflict_g2", 0.5), ("conflict_g5", 0.5), ("override_diff", 0.3)):
    v = rs[f"cr/{cat}"] - rb[f"cr/{cat}"] if rs and rb and f"cr/{cat}" in rs else None
    check(f"run 5: syn {cat} - B", v, lambda v, t=thr: v >= t, f">= +{thr}")
check("run 5: code cost of syn vs run 2", d("fit_GVRbb_v2syn", "fit_GVRbb_v2"), lambda v: v <= 0.005, "<= +0.005")
if rs and "syn/fit_mean" in rs and "syn/heldout_mean" in rs and E("fit_GVRbb_v2"):
    base = E("fit_GVRbb_v2")
    gf, gh = rs["syn/fit_mean"] - base.get("syn/fit_mean", 0), rs["syn/heldout_mean"] - base.get("syn/heldout_mean", 0)
    check("run 5: held-out templates keep >= 50% of the gain", gh / gf if gf > 0 else None, lambda v: v >= 0.5, ">= 0.5")

rc = json.load(open(f"{RES}/recall_check.json"))["results"] if os.path.exists(f"{RES}/recall_check.json") else {}
rec = []
if rc:
    arms = list(rc)
    rec = ["| task | " + " | ".join(arms) + " |", "|---" * (len(arms) + 1) + "|"]
    for k in [k[:-4] for k in next(iter(rc.values())) if k.endswith("/acc")]:
        rec.append(f"| {k} | " + " | ".join(f"{rc[a][k + '/acc']:.3f}" for a in arms) + " |")

summary = ("# Kaggle round-2 results\\n\\n" + f"Precision {PRECISION}, {VAL_BATCHES} windows/split, GPU "
           f"{torch.cuda.get_device_name(0) if N_GPU else 'cpu'}\\n\\n## Main\\n" + "\\n".join(lines) +
           "\\n\\n## Gate and pre-registered predictions (decision2.md)\\n| check | value | prediction | |\\n|---|---|---|---|\\n" +
           "\\n".join(checks) + "\\n\\n## Recall check (token accuracy)\\n" + "\\n".join(rec) +
           "\\n\\n## Jobs\\n" + "\\n".join(f"- {k}: {v}" for k, v in STATUS.items()) + "\\n")
open(f"{RES}/SUMMARY.md", "w").write(summary)
print(summary)
shutil.make_archive(f"{WORK}/ice_results", "zip", RES)
upload_results()
print("saved", f"{WORK}/ice_results.zip", "| uploaded to Modal kaggle_results/latest" if UPLOAD_TO_MODAL else "")
""")

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                     language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
json.dump(nb, open(OUT, "w"), indent=1)
print("wrote", OUT, file=sys.stderr)
