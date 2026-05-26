from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(os.getcwd())
config_path = PROJECT_ROOT / "config.env"
if not config_path.exists():
    raise FileNotFoundError("Mangler config.env – kør: cp config.env.example config.env")

def load_env(path: Path):
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env

def expand(value: str, env: dict) -> str:
    out = value
    for _ in range(10):
        changed = False
        for k, v in env.items():
            token = "${" + k + "}"
            if token in out:
                out = out.replace(token, v)
                changed = True
        if not changed:
            break
    return out

env = load_env(config_path)
env["PROJECT_ROOT"] = str(PROJECT_ROOT)
for k in list(env.keys()):
    env[k] = expand(env[k], env)

# --- Tjek config-nøgler ---
required_keys = [
    "SEUTAO_REPO_DIR",
    "DOWNLOAD_DIR",
    "PRETRAIN_DIR",
    "AUX_DATA_DIR",
    "RSNA_TRAIN_DIR",
    "RSNA_TEST_DIR",
]

for key in required_keys:
    if key not in env or not env[key]:
        raise RuntimeError(f"Mangler env-var i config.env: {key}")

# --- Tjek SeuTao repo ---
repo_dir = Path(env["SEUTAO_REPO_DIR"])
if not repo_dir.exists():
    print(f"[WARN] SeuTao repo ikke fundet: {repo_dir}")
    print("       Kør 'bash download_all.sh' først.")
    sys.exit(0)

must_exist = [
    repo_dir / "2DNet" / "src" / "prepare_data.py",
    repo_dir / "2DNet" / "src" / "predict.py",
    repo_dir / "SequenceModel" / "main.py",
]
for p in must_exist:
    if not p.exists():
        raise FileNotFoundError(f"Mangler repo-fil: {p}")

# --- Tjek DICOM-data (kun warning, ikke fejl – downloades af pipeline) ---
for key in ["RSNA_TRAIN_DIR", "RSNA_TEST_DIR"]:
    p = Path(env[key])
    if not p.exists():
        print(f"[WARN] DICOM-mappe mangler: {p}")
        print(f"       Kør 'bash download_all.sh' for at downloade fra Kaggle.")

# --- Tjek downloads (kun warning) ---
aux_dir = Path(env["AUX_DATA_DIR"])
pre_dir = Path(env["PRETRAIN_DIR"])

expected_downloads = [
    aux_dir / "data.zip",
    aux_dir / "csv.zip",
]
for p in expected_downloads:
    if not p.exists():
        print(f"[WARN] Mangler download: {p}")
        print(f"       Kør 'bash download_all.sh' for at downloade.")

# --- Tjek torch ---
try:
    import torch
    print(f"torch ok: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
except Exception as e:
    raise RuntimeError(f"Torch import fejlede: {e}")

print("Setup-tjek bestået.")