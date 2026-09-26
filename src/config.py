"""Configuration and path handling. All modules read settings from here.

Data root: `data/` by default. `use_mock()` (called by `run_all.py --mock`) redirects
every path to `data_mock/`, so the offline dry run never touches real data and the
measurement step reads the mock files (v1.0.0 skipped them and stopped).
`use_root("data/r1")` is used for the revision-round collection."""
import os
import pathlib
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
_DATA_ROOT = [ROOT / "data"]


def load_yaml(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


EXP = load_yaml("experiment.yaml")
DIMENSIONS = load_yaml("dimensions.yaml")
MODELS = load_yaml("models.yaml")


def use_mock():
    _DATA_ROOT[0] = ROOT / "data_mock"


def use_root(rel):
    _DATA_ROOT[0] = ROOT / rel


def data_root():
    return _DATA_ROOT[0]


def _resolve(rel):
    rel = str(rel)
    if rel.startswith("data/"):
        rel = rel[len("data/"):]
    return data_root() / rel


def path(key):
    p = _resolve(EXP["paths"][key])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir():
    d = _resolve(EXP["paths"]["raw_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir():
    d = _resolve(EXP["paths"]["results_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def excluded_models():
    return list(EXP.get("analysis", {}).get("exclude_models", []) or [])


def family(model):
    fam = EXP.get("analysis", {}).get("families", {}) or {}
    if model in fam:
        return fam[model]
    m = str(model).lower()
    for key, name in [("nemotron", "Llama"), ("llama", "Llama"), ("mixtral", "Mistral"), ("mistral", "Mistral"),
                      ("qwen", "Qwen"), ("gemma", "Gemma"), ("phi", "Phi"), ("gpt-oss", "GPT-OSS")]:
        if key in m:
            return name
    return m.split("/")[-1]


def expected_sign(dim):
    e = str(DIMENSIONS.get(dim, {}).get("expected_sign", "")).strip()
    return 1 if e.startswith("+") else (-1 if e.startswith("-") else 0)


def api_key():
    key = os.environ.get(EXP["api"]["env_key"])
    if not key:
        raise SystemExit(f"\nEnvironment variable {EXP['api']['env_key']} not set. Copy .env.example to .env "
                         f"and add your NVIDIA NIM key (never commit it).\n")
    return key
