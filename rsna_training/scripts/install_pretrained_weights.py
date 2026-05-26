from pathlib import Path
import shutil
import os

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

def find_checkpoint_dir(root: Path):
    for p in root.rglob("model_epoch_best_0.pth"):
        return p.parent
    return None

env = load_env(config_path)
env["PROJECT_ROOT"] = str(PROJECT_ROOT)
for k in list(env.keys()):
    env[k] = expand(env[k], env)

pretrain_dir = Path(env["PRETRAIN_DIR"])
repo_2d = Path(env["SEUTAO_REPO_DIR"]) / "2DNet"
repo_2d.mkdir(parents=True, exist_ok=True)

mapping = {
    "densenet121_512": repo_2d / "DenseNet121_change_avg_256",
    "densenet169_256": repo_2d / "DenseNet169_change_avg_256",
    "seresnext101_256": repo_2d / "se_resnext101_32x4d_256",
}

for src_name, dst in mapping.items():
    src = pretrain_dir / src_name
    if not src.exists():
        raise FileNotFoundError(f"Mangler pretrained mappe: {src}")

    if dst.exists():
        print(f"[skip] {dst} findes allerede")
        continue

    ckpt_dir = find_checkpoint_dir(src) or src
    print(f"[copy] {ckpt_dir} -> {dst}")
    shutil.copytree(ckpt_dir, dst)

print("Pretrained weights installeret i 2DNet.")