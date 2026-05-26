from pathlib import Path
import os
import shutil
import zipfile

PROJECT_ROOT = Path(os.getcwd())
config_path = PROJECT_ROOT / "config.env"
if not config_path.exists():
    raise FileNotFoundError("Mangler config.env")

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

def unzip_to(zip_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

def find_checkpoint_dir(root: Path):
    for p in root.rglob("model_epoch_best_0.pth"):
        return p.parent
    return None

env = load_env(config_path)
env["PROJECT_ROOT"] = str(PROJECT_ROOT)
for k in list(env.keys()):
    env[k] = expand(env[k], env)

download_dir = Path(env["DOWNLOAD_DIR"])
pretrain_dir = Path(env["PRETRAIN_DIR"])
aux_dir = Path(env["AUX_DATA_DIR"])
repo_dir = Path(env["SEUTAO_REPO_DIR"])

data_zip = aux_dir / "data.zip"
csv_zip = aux_dir / "csv.zip"
feature_zip = aux_dir / "feature_samples.zip"

if data_zip.exists():
    target = repo_dir / "2DNet" / "data"
    print(f"[unzip] {data_zip} -> {target}")
    unzip_to(data_zip, target)
else:
    print(f"[warn] Mangler {data_zip}")

if csv_zip.exists():
    target = repo_dir / "SequenceModel" / "csv"
    print(f"[unzip] {csv_zip} -> {target}")
    unzip_to(csv_zip, target)
else:
    print(f"[warn] Mangler {csv_zip}")

if feature_zip.exists():
    target = repo_dir / "SequenceModel" / "features"
    print(f"[unzip] {feature_zip} -> {target}")
    unzip_to(feature_zip, target)
else:
    print(f"[warn] Mangler {feature_zip}")

# Udpak pretrained model zip-filer (hvis de er zip-arkiver)
pretrained_zips = {
    "densenet121_512": pretrain_dir / "densenet121_512.zip",
    "densenet169_256": pretrain_dir / "densenet169_256.zip",
    "seresnext101_256": pretrain_dir / "seresnext101_256.zip",
}

for name, zip_path in pretrained_zips.items():
    target = pretrain_dir / name
    if target.exists():
        print(f"[skip] {target} findes allerede")
        continue
    if zip_path.exists():
        print(f"[unzip] {zip_path} -> {target}")
        try:
            unzip_to(zip_path, target)
        except zipfile.BadZipFile:
            # Filen er muligvis allerede en mappe eller et enkelt checkpoint
            print(f"[warn] {zip_path} er ikke en gyldig zip – springer over")

print("Udpakning afsluttet.")