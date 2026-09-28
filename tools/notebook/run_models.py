#!/usr/bin/env python3
"""GPU-notebook runner for the self-served June checkpoints (revision round).

Started by tools/notebook/start.sh on a notebook GPU: the ModelScope 192 GB AMD instance (MI300-class,
ROCm, runs all seven June checkpoints and the Nemotron-Super re-collection) or a 24 GB NVIDIA instance
(runs the four small checkpoints; larger models are skipped for lack of memory). For each model, in order:
  1. skip it if its data are already complete (parts A and C; part B for Nemotron-Super);
  2. download exactly the files listed in tools/notebook/manifests/<model>.json from ModelScope;
  3. verify every file against the Hugging Face release (size + SHA-256 / Git object id); stop on any
     mismatch;
  4. serve the model under its June model id: vLLM (bf16); if vLLM cannot start, vLLM with another
     attention backend; if that fails too, tools/notebook/hf_server.py (transformers, bf16), and for a
     model whose own code needs an older transformers (manifest "transformers_version"), hf_server.py
     with that version installed beside the image's (only for that server). A reasoning model whose
     reasoning vLLM cannot split (manifest "think_split_proxy") is served by vLLM behind
     tools/notebook/think_split_proxy.py. The engine used is recorded in
     data/r1/endpoints/<model>_engine.json;
  5. python run_all.py --collect-r1 --models <id>   (resumable);
  6. stop the server, delete the weights (unless --keep-weights), pack the results so far.
Everything is logged to data/r1/logs/notebook_<UTC>/. No response text is printed or logged.
The result file r1_notebook_results_<UTC>.tar.gz (repository root) is what you download.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "tools" / "notebook" / "manifests"
# small models first (they check the whole chain quickly); Qwen3-Next last (largest, newest architecture)
ORDER = ["google/gemma-2-2b-it", "microsoft/phi-4-mini-instruct", "nvidia/nemotron-mini-4b-instruct",
         "meta/llama-3.1-8b-instruct", "mistralai/mixtral-8x7b-instruct-v0.1", "meta/llama-3.3-70b-instruct",
         "nvidia/llama-3.3-nemotron-super-49b-v1.5", "qwen/qwen3-next-80b-a3b-instruct"]
PORT = 8000            # what the collection talks to (config/collection_r1.yaml, endpoint `local`)
UPSTREAM_PORT = 8001   # vLLM's own port when it runs behind think_split_proxy.py
PIP_INDEX = os.environ.get("PIP_INDEX", "https://mirrors.aliyun.com/pypi/simple/")
WORK = ROOT.parent / "r1work"   # set from --work in main()
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M")
LOGDIR = ROOT / "data" / "r1" / "logs" / f"notebook_{STAMP}"
WEIGHTS_CSV = ROOT / "data" / "model_weights" / f"weights_check_{STAMP}.csv"


def log(msg):
    LOGDIR.mkdir(parents=True, exist_ok=True)
    line = f"[{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')}Z] {msg}"
    print(line, flush=True)
    with open(LOGDIR / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def safe(name):
    return name.replace("/", "_")


# ------------------------------------------------------------------ environment
def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def gpu_info():
    """Name, memory (MiB), driver/runtime and bf16 support of GPU 0 (NVIDIA or AMD ROCm)."""
    info = None
    r = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]) \
        if shutil.which("nvidia-smi") else None
    if r is not None and r.returncode == 0 and r.stdout.strip():
        name, mem, drv = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        info = dict(vendor="nvidia", name=name, memory_mib=int(float(mem)), driver=drv)
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            hip = getattr(torch.version, "hip", None)
            base = info or dict(vendor="amd" if hip else "nvidia", name=p.name,
                                memory_mib=int(p.total_memory / 2 ** 20), driver=f"hip {hip}" if hip else "")
            base.update(arch=getattr(p, "gcnArchName", "") or "%d.%d" % (p.major, p.minor),
                        bf16=bool(torch.cuda.is_bf16_supported()), torch=torch.__version__)
            info = base
    except Exception as e:
        if info:
            info["torch_error"] = f"{type(e).__name__}"
    return info


def env_record(gpu):
    info = dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), gpu=gpu,
                python=sys.version.split()[0])
    for mod in ("vllm", "torch", "transformers", "modelscope"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:
            info[mod] = f"not importable ({type(e).__name__})"
    bi = ROOT / "BUNDLE_INFO.json"
    info["bundle"] = json.loads(bi.read_text(encoding="utf-8")) if bi.exists() else None
    (LOGDIR / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    fr = sh([sys.executable, "-m", "pip", "freeze"])
    (LOGDIR / "pip_freeze.txt").write_text(fr.stdout, encoding="utf-8")
    for tool, args in (("nvidia-smi", []), ("rocm-smi", ["--showproductname", "--showmeminfo", "vram",
                                                          "--showdriverversion"])):
        if shutil.which(tool):
            out = sh([tool] + args)
            (LOGDIR / f"{tool.replace('-', '_')}.txt").write_text(out.stdout + out.stderr, encoding="utf-8")
    return info


# ------------------------------------------------------------------ weights
def sha256_file(p, bufsize=16 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_blob_oid(p):
    size = p.stat().st_size
    h = hashlib.sha1(b"blob %d\0" % size)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lfs_pointer_oid(sha, size):
    ptr = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
    return hashlib.sha1(b"blob %d\0" % len(ptr) + ptr).hexdigest()


_MIRROR = {}


def mirror_oids(man):
    """Git object ids of the release files as listed by hf-mirror.com (a mirror of the Hugging Face API),
    used only to re-check a file whose manifest entry disagrees."""
    key = (man["hf_repo"], man["hf_commit"])
    if key not in _MIRROR:
        url = f"https://hf-mirror.com/api/models/{man['hf_repo']}/tree/{man['hf_commit']}?recursive=true"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                _MIRROR[key] = {e["path"]: e.get("oid") for e in json.loads(r.read().decode())}
        except Exception as e:
            log(f"  (hf-mirror listing not available: {type(e).__name__})")
            _MIRROR[key] = {}
    return _MIRROR[key]


def verify(man, mdir):
    rows, ok = [], True
    for f in man["files"]:
        p = mdir / f["path"]
        row = dict(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), model=man["model"],
                   hf_repo=man["hf_repo"], hf_commit=man["hf_commit"], modelscope_repo=man["modelscope_repo"],
                   path=f["path"], expected_size=f["size"], size="", sha256="", git_oid_computed="",
                   hf_git_oid=f["hf_git_oid"], match=False, note="")
        if not p.exists():
            row["note"] = "missing"
            ok = False
            rows.append(row)
            continue
        row["size"] = p.stat().st_size
        s = sha256_file(p)
        row["sha256"] = s
        if f["lfs"]:
            row["git_oid_computed"] = lfs_pointer_oid(s, row["size"])
            row["match"] = (row["size"] == f["size"] and (f.get("sha256") in (None, s))
                            and row["git_oid_computed"] == f["hf_git_oid"])
        else:
            row["git_oid_computed"] = git_blob_oid(p)
            row["match"] = (row["size"] == f["size"] and row["git_oid_computed"] == f["hf_git_oid"])
        if not row["match"]:
            m = mirror_oids(man).get(f["path"])
            if m and m == row["git_oid_computed"]:
                row["match"] = True
                row["note"] = "manifest entry differed; file matches the id listed by hf-mirror for hf_commit"
        ok &= bool(row["match"])
        rows.append(row)
    import csv
    WEIGHTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not WEIGHTS_CSV.exists()
    with open(WEIGHTS_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    return ok, rows


def download(man, mdir):
    files = [f["path"] for f in man["files"]]
    cli = os.environ.get("MS_DOWNLOAD_CMD", "modelscope").split()
    py = [sys.executable, "-c",
          "import sys; from modelscope import snapshot_download as s; "
          "s(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])"]
    use_cli = bool(shutil.which(cli[0])) or "MS_DOWNLOAD_CMD" in os.environ
    for attempt in range(1, 4):
        c = (cli + ["download", "--model", man["modelscope_repo"], "--local_dir", str(mdir)] + files) if use_cli \
            else (py + [man["modelscope_repo"], str(mdir)] + files)
        log(f"  download attempt {attempt}: {man['modelscope_repo']} ({len(files)} files, "
            f"{sum(f['size'] for f in man['files']) / 1e9:.1f} GB)")
        with open(LOGDIR / f"download_{safe(man['model'])}.log", "a", encoding="utf-8") as lf:
            r = subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode == 0 and all((mdir / p).exists() for p in files):
            return True
        time.sleep(10 * attempt)
    return False


# ------------------------------------------------------------------ servers (vLLM, then transformers)
def vllm_base_cmd():
    if os.environ.get("VLLM_CMD"):
        return os.environ["VLLM_CMD"].split()
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]


_HELP = {}


def vllm_supports(flag):
    """Is `flag` a vLLM server option? Newer vLLM prints only option groups for --help; --help=all lists all."""
    if "text" not in _HELP:
        text = ""
        for h in ("--help", "--help=all"):
            try:
                r = sh(vllm_base_cmd() + [h], timeout=300)
                text += (r.stdout or "") + (r.stderr or "")
            except Exception:
                pass
        _HELP["text"] = text
    return flag in _HELP["text"]


def engine_plan(gpu, man):
    """Attempts in order: vLLM default; vLLM with another exact attention backend; transformers (the
    image's version); transformers at the version the model's own code needs, if the manifest names one.
    Each attempt is (engine, option): the attention backend for vLLM, the pinned version for transformers."""
    alt = "TRITON_ATTN" if (gpu or {}).get("vendor") == "amd" else "FLEX_ATTENTION"
    plan = [("vllm", None), ("vllm", alt), ("transformers", None)]
    if man.get("transformers_version"):
        plan.append(("transformers", man["transformers_version"]))
    return plan


def label(engine, opt):
    if not opt:
        return engine
    return f"{engine} ({opt})" if engine == "vllm" else f"transformers {opt}"


def pinned_transformers(man):
    """The packages in the manifest's "transformers_packages" (exact versions: transformers and the
    tokenizers / huggingface_hub / accelerate of its time) installed without dependencies into their own
    folder, which goes first on PYTHONPATH of that one server process only; torch and everything else stay
    the image's. Returns the folder, or None if it cannot be installed or imported."""
    ver = man["transformers_version"]
    pkgs = man.get("transformers_packages") or [f"transformers=={ver}"]
    d = WORK / f"py_transformers_{ver}"
    if not (d / ".ready").exists():
        log(f"  installing {' '.join(pkgs)} for this model only (into {d})")
        shutil.rmtree(d, ignore_errors=True)
        with open(LOGDIR / "pip_pinned_transformers.log", "a", encoding="utf-8") as lf:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(d),
                                "-i", PIP_INDEX] + pkgs, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log("  could not install them (see pip_pinned_transformers.log)")
            return None
        chk = sh([sys.executable, "-c", "import transformers; from transformers.modeling_utils import PreTrainedModel; "
                  "from transformers import AutoModelForCausalLM, AutoTokenizer; print(transformers.__version__)"],
                 env=dict(os.environ, PYTHONPATH=str(d)))
        if chk.returncode != 0 or chk.stdout.strip().splitlines()[-1:] != [ver]:
            log(f"  transformers {ver} does not import here: {(chk.stderr or chk.stdout).strip()[-300:]}")
            return None
        (d / ".ready").touch()
    return d


class Server:
    def __init__(self, proc, lf, args, proxy=None, version=None):
        self.proc, self.lf, self.args, self.proxy, self.version = proc, lf, args, proxy, version


def start_server(man, mdir, engine, opt=None):
    """Start one engine for the model; returns a Server, or None if it could not even be launched."""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
               DO_NOT_TRACK="1")
    proxy_port = None
    version = None
    prefix = ""
    if engine == "vllm":
        port = UPSTREAM_PORT if man.get("think_split_proxy") else PORT
        args = vllm_base_cmd() + ["--model", str(mdir), "--served-model-name", man["model"], "--dtype", "bfloat16",
                                  "--host", "127.0.0.1", "--port", str(port), "--enforce-eager", "--seed", "0"] \
            + man.get("vllm_args", [])
        if vllm_supports("--disable-log-requests"):
            args.append("--disable-log-requests")   # prompts/outputs are not written to the server log
        if opt:
            env["VLLM_ATTENTION_BACKEND"] = opt     # older vLLM
            if vllm_supports("--attention-backend"):
                args += ["--attention-backend", opt]
            prefix = f"VLLM_ATTENTION_BACKEND={opt} "
        if port != PORT:
            proxy_port = port
    else:
        if opt:
            d = pinned_transformers(man)
            if d is None:
                return None
            env["PYTHONPATH"] = str(d) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            prefix = f"PYTHONPATH={d} "
            version = opt
        args = [sys.executable, str(ROOT / "tools" / "notebook" / "hf_server.py"), "--model", str(mdir),
                "--served-model-name", man["model"], "--port", str(PORT)]
        if man.get("trust_remote_code"):
            args.append("--trust-remote-code")
        if man.get("reasoning_split"):
            args.append("--reasoning-split")
    with open(LOGDIR / f"server_cmd_{safe(man['model'])}.txt", "a", encoding="utf-8") as f:
        f.write(prefix + " ".join(args) + "\n")
        if proxy_port:
            f.write(f"python tools/notebook/think_split_proxy.py --port {PORT} --upstream-port {proxy_port}"
                    f"   (run as a thread of run_models.py)\n")
    lf = open(LOGDIR / f"server_{safe(man['model'])}.log", "a", encoding="utf-8")
    lf.write(f"\n===== {label(engine, opt)} {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} =====\n")
    lf.flush()
    proc = subprocess.Popen(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT),
                            start_new_session=True)
    proxy = None
    if proxy_port:
        sys.path.insert(0, str(ROOT / "tools" / "notebook"))
        from think_split_proxy import start_in_thread
        proxy = start_in_thread(PORT, proxy_port)
        lf.write(f"[run_models] think_split_proxy listening on {PORT}, forwarding to vLLM on {proxy_port}\n")
        lf.flush()
    return Server(proc, lf, args, proxy, version)


def wait_ready(man, srv, timeout=2400):
    t0 = time.time()
    url = f"http://127.0.0.1:{PORT}/v1/models"      # through the proxy, if there is one
    while time.time() - t0 < timeout:
        if srv.proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                ids = [m["id"] for m in json.loads(r.read().decode()).get("data", [])]
                if man["model"] in ids:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_server(srv):
    proc = srv.proc
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    else:
        try:                                # children (e.g. vLLM's engine process) of a server that died
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    if srv.proxy is not None:
        srv.proxy.shutdown()
        srv.proxy.server_close()
    srv.lf.close()
    time.sleep(5)


def record_engine(man, engine, opt, srv, gpu):
    args = srv.args
    p = LOGDIR / f"server_{safe(man['model'])}.log"
    keep = [ln for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if any(k in ln.lower() for k in ("sampling param", "generation config", "generation_config",
                                              "sampling defaults", "chat template", "dtype", "version",
                                              "attention backend", "using"))][-60:] if p.exists() else []
    (LOGDIR / f"sampling_defaults_{safe(man['model'])}.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    ver = srv.version
    if ver is None:
        try:
            ver = __import__("vllm" if engine == "vllm" else "transformers").__version__
        except Exception:
            ver = "unknown"
    if srv.proxy is not None:
        split = ("tools/notebook/think_split_proxy.py in front of vLLM (no vLLM reasoning parser): text before "
                 "</think> returned as reasoning_content, the rest as content")
    elif engine == "transformers" and man.get("reasoning_split"):
        split = "tools/notebook/hf_server.py --reasoning-split: text before </think> returned as reasoning_content"
    else:
        split = next((f"vLLM --reasoning-parser {args[i + 1]}" for i, a in enumerate(args[:-1])
                      if a == "--reasoning-parser"), "none")
    rec = dict(model=man["model"], engine=engine, engine_version=ver,
               attention_backend=(opt or "default") if engine == "vllm" else "see hf_server log",
               transformers_pinned=opt if engine == "transformers" else None, reasoning_split=split,
               args=[a if not a.startswith("/") else pathlib.Path(a).name for a in args], dtype="bfloat16",
               gpu=gpu, hf_repo=man["hf_repo"], hf_commit=man["hf_commit"], modelscope_repo=man["modelscope_repo"],
               started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               server_log_excerpt=keep[-20:])
    d = ROOT / "data" / "r1" / "endpoints"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{safe(man['model'])}_engine.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ collection bookkeeping
def counts(name):
    sys.path.insert(0, str(ROOT))
    from src import r1
    cfg = r1._cfg()
    exp = r1._expected(cfg)
    r1root = ROOT / cfg.get("data_root", "data/r1")
    cid = cfg.get("collection_id", "r1")
    f = r1root / "raw" / f"{safe(name)}.jsonl"
    rer = {m["name"]: m for m in cfg.get("rerun_excluded", {}).get("models", [])}
    if name in rer:
        reps = cfg["rerun_excluded"].get("repeats_by_stage", {}).get(rer[name].get("stage", "instruct"), 8)
        got = dict(B_rerun=r1._count(f, lambda r: str(r.get("collection", "")).endswith("_rerun")))
        want = dict(B_rerun=exp["n_june_rating"] * reps)
    else:
        f12 = r1root / cfg.get("budget_check", {}).get("data_subdir", "budget12") / "raw" / f"{safe(name)}.jsonl"
        got = dict(A_isolated=r1._count(f, lambda r: r.get("format") == "rating" and r.get("collection") == cid),
                   A_joint=r1._count(f, lambda r: r.get("format") == "joint"),
                   C_budget12=r1._count(f12, lambda r: True))
        want = {k: exp[k] for k in got}
    return got, want, all(got[k] >= want[k] for k in got)


def collect(name):
    env = dict(os.environ, R1_LOG_DIR=str(LOGDIR))
    cmd = [sys.executable, "run_all.py", "--collect-r1", "--models", name]
    with open(LOGDIR / f"collect_{safe(name)}.log", "a", encoding="utf-8") as lf:
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            lf.write(line)
            s = line.rstrip()
            if any(k in s for k in ("[progress]", "[endpoint]", "[skip]", "[stopped]", "[not collected]",
                                    "[r1]", "[clock]", "  -> ", "Traceback", "Error")):
                print("    " + s, flush=True)
        return p.wait()


# ------------------------------------------------------------------ packing
def pack(final=False):
    out = ROOT / f"r1_notebook_results_{STAMP}.tar.gz"
    tmp = out.with_suffix(".partial")
    r1root = ROOT / "data" / "r1"
    members = []
    for name in ORDER:
        for sub in ("raw", "budget12/raw"):
            p = r1root / sub / f"{safe(name)}.jsonl"
            if p.exists():
                members.append(p)
        members += sorted((r1root / "endpoints").glob(f"{safe(name)}*"))
    for sub in ("raw", "budget12/raw"):     # token-usage logs, renamed so they never overwrite local ones
        u = r1root / sub / "_usage.csv"
        if u.exists():
            shutil.copy2(u, LOGDIR / f"usage_{sub.replace('/', '_')}.csv")
    for d in sorted((r1root / "logs").glob("notebook_*")):      # this run and any earlier (interrupted) runs
        members += sorted(p for p in d.rglob("*") if p.is_file())
    members += sorted((ROOT / "data" / "model_weights").glob("weights_check_*.csv"))
    with tarfile.open(tmp, "w:gz") as t:
        for p in members:
            t.add(p, arcname=str(p.relative_to(ROOT)).replace(os.sep, "/"))
    tmp.replace(out)
    if final:
        log(f"Result file: {out.name} ({out.stat().st_size / 1e6:.1f} MB). Download it; do not open it.")
    return out


def restore_previous():
    """On a fresh instance: if an earlier result archive was uploaded next to the code (repository root or
    its parent folder), put its data back first so that collection resumes instead of starting over."""
    cands = sorted(list(ROOT.glob("r1_notebook_results_*.tar.gz")) + list(ROOT.parent.glob("r1_notebook_results_*.tar.gz")),
                   key=lambda p: p.name)
    if not cands:
        return
    arc = cands[-1]
    allowed = ("data/r1/raw/", "data/r1/budget12/raw/", "data/r1/endpoints/", "data/r1/logs/notebook_",
               "data/model_weights/")
    n = 0
    with tarfile.open(arc, "r:gz") as t:
        for m in t.getmembers():
            name = m.name
            if not m.isfile() or name.startswith("/") or ".." in name.split("/") or not name.startswith(allowed):
                continue
            dest = ROOT / name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(t.extractfile(m).read())
            n += 1
    if n:
        log(f"Restored {n} file(s) from the earlier result archive {arc.name}; finished work will be skipped.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ORDER))
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--work", default=os.environ.get("WORK", str(ROOT.parent / "r1work")))
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    global WORK
    work = WORK = pathlib.Path(a.work)
    (work / "models").mkdir(parents=True, exist_ok=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log(f"Notebook run {STAMP}; repository {ROOT}; work dir {work}")
    restore_previous()

    sys.path.insert(0, str(ROOT))
    from src.clockcheck import check_clock
    check_clock(log_dir=LOGDIR)                     # stops if the clock is off by more than 2 minutes

    gpu = gpu_info()
    if not gpu and not os.environ.get("SKIP_GPU_CHECK"):
        raise SystemExit("No GPU found. Start the notebook with a GPU instance.")
    info = env_record(gpu)
    log(f"GPU: {gpu}; vllm {info.get('vllm')}; torch {info.get('torch')}; transformers {info.get('transformers')}")
    if gpu and not gpu.get("bf16", False) and not os.environ.get("SKIP_GPU_CHECK"):
        raise SystemExit(f"This GPU ({gpu['name']}) has no bf16 support. Choose a newer GPU instance and run again.")

    summary = []
    for name in models:
        man_p = MANIFESTS / f"{safe(name)}.json"
        if not man_p.exists():
            log(f"[{name}] no manifest; skipped")
            continue
        man = json.loads(man_p.read_text(encoding="utf-8"))
        got, exp, done = counts(name)
        if done:
            log(f"[{name}] already complete {got}; skipped")
            summary.append((name, "complete (earlier run)", got))
            continue
        if gpu and gpu["memory_mib"] < man["min_vram_mib"] and not os.environ.get("SKIP_GPU_CHECK"):
            log(f"[{name}] GPU memory {gpu['memory_mib']} MiB < {man['min_vram_mib']} MiB needed; skipped")
            summary.append((name, "skipped: GPU too small", got))
            continue
        need = sum(f["size"] for f in man["files"]) * 1.1
        free = shutil.disk_usage(work).free
        if free < need:
            log(f"[{name}] only {free / 1e9:.1f} GB free in {work}, {need / 1e9:.1f} GB needed; skipped")
            summary.append((name, "skipped: disk", got))
            continue
        mdir = work / "models" / safe(name)
        log(f"[{name}] step 1/4 download")
        drop = (lambda: None) if a.keep_weights else (lambda: shutil.rmtree(mdir, ignore_errors=True))
        if not download(man, mdir):
            log(f"[{name}] download failed (see download_{safe(name)}.log); skipped")
            summary.append((name, "download failed", got))
            drop()
            continue
        log(f"[{name}] step 2/4 verify {len(man['files'])} files against {man['hf_repo']}@{man['hf_commit'][:10]}")
        ok, rows = verify(man, mdir)
        if not ok:
            bad = [r["path"] for r in rows if not r["match"]]
            log(f"[{name}] VERIFICATION FAILED for {bad}; this model is NOT run. Send {WEIGHTS_CSV.name} to the assistant.")
            summary.append((name, "verification failed", got))
            drop()
            continue
        tc = json.loads((mdir / "tokenizer_config.json").read_text(encoding="utf-8"))
        if not (tc.get("chat_template") or (mdir / "chat_template.jinja").exists()):
            log(f"[{name}] tokenizer_config.json has no chat_template; skipped")
            summary.append((name, "no chat template", got))
            drop()
            continue
        srv = None
        used = None
        for engine, opt in engine_plan(gpu, man):
            log(f"[{name}] step 3/4 start {label(engine, opt)}, bf16")
            t_start = time.time()
            srv = start_server(man, mdir, engine, opt)
            if srv is not None and wait_ready(man, srv):
                used = (engine, opt)
                break
            if srv is not None:
                stop_server(srv)
            srv = None
            log(f"[{name}] {label(engine, opt)} did not start after {time.time() - t_start:.0f} s "
                f"(see server_{safe(name)}.log)")
        try:
            if not used:
                log(f"[{name}] no engine could serve this model; skipped. Send server_{safe(name)}.log to the assistant.")
                summary.append((name, "server failed to start", got))
                drop()
                continue
            record_engine(man, used[0], used[1], srv, gpu)
            log(f"[{name}] step 4/4 collect ({', '.join(exp)})")
            for attempt in (1, 2):
                rc = collect(name)
                got, exp, done = counts(name)
                log(f"[{name}] pass {attempt}: exit {rc}; records {got} of {exp}")
                if done:
                    break
        finally:
            if srv is not None:
                stop_server(srv)
        summary.append((name, f"complete ({label(*used)})" if done else "incomplete (run start.sh again)", got))
        drop()
        pack()
    log("Summary:")
    for n, st, g in summary:
        log(f"  {n:42s} {st:34s} {g}")
    pack(final=True)


if __name__ == "__main__":
    main()
