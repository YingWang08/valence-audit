"""集中加载配置与路径。所有模块从这里取配置，避免各处硬编码。"""
import os
import pathlib
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()                      # 自动加载项目根目录下的 .env
except ImportError:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def load_yaml(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


EXP = load_yaml("experiment.yaml")
DIMENSIONS = load_yaml("dimensions.yaml")
MODELS = load_yaml("models.yaml")


def path(key):
    """把 experiment.yaml 里 paths.* 的相对路径解析为绝对路径，并确保父目录存在。"""
    p = ROOT / EXP["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir():
    d = ROOT / EXP["paths"]["raw_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir():
    d = ROOT / EXP["paths"]["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def api_key():
    key = os.environ.get(EXP["api"]["env_key"])
    if not key:
        raise SystemExit(
            f"\n找不到环境变量 {EXP['api']['env_key']}。\n"
            f"请把项目根目录的 .env.example 复制为 .env，填入你【新生成】的 NVIDIA 密钥；\n"
            f"或在终端设置： export {EXP['api']['env_key']}=nvapi-xxxx  （Windows: set ...）\n"
        )
    return key
