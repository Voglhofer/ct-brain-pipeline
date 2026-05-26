from pathlib import Path
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

env = load_env(config_path)
env["PROJECT_ROOT"] = str(PROJECT_ROOT)
for k in list(env.keys()):
    env[k] = expand(env[k], env)

repo_dir = Path(env["SEUTAO_REPO_DIR"])

train_png = env.get("RSNA_TRAIN_PNG_DIR", str(repo_dir / "2DNet" / "train_png"))
test_png = env.get("RSNA_TEST_PNG_DIR", str(repo_dir / "2DNet" / "test_png"))
concat_train = env.get("RSNA_CONCAT_TRAIN_DIR", str(repo_dir / "2DNet" / "train_concat_3images_256"))
concat_test = env.get("RSNA_CONCAT_TEST_DIR", str(repo_dir / "2DNet" / "stage2_test_concat_3images"))

# Patch 2D settings.py
settings_2d = repo_dir / "2DNet" / "src" / "settings.py"
settings_2d.write_text(
    "train_png_dir = r'" + train_png + "/'\n"
    "test_png_dir = r'" + test_png + "/'\n"
    "concat_train_dir = r'" + concat_train + "/'\n"
    "concat_test_dir = r'" + concat_test + "/'\n"
)

# Patch SequenceModel/settings.py
seq_csv = env.get("SEQUENCE_CSV_ROOT", str(repo_dir / "SequenceModel" / "csv"))
seq_fea = env.get("SEQUENCE_FEATURE_ROOT", str(repo_dir / "SequenceModel" / "features"))
seq_out = env.get("SEQUENCE_FINAL_OUTPUT_ROOT", str(repo_dir / "FinalSubmission"))

seq_settings = repo_dir / "SequenceModel" / "settings.py"
seq_settings.write_text(
    "import os\n"
    "csv_root = r'" + seq_csv + "'\n"
    "feature_path = r'" + seq_fea + "'\n"
    "final_output_path = r'" + seq_out + "'\n"
)

# Patch hardcoded concat paths in predict.py
predict_py = repo_dir / "2DNet" / "src" / "predict.py"
text = predict_py.read_text()

old_train = "/home1/kaggle_rsna2019/process/train_concat_3images_256/"
old_test = "/home1/kaggle_rsna2019/process/stage2_test_concat_3images/"

new_train = concat_train + "/"
new_test = concat_test + "/"

text = text.replace(old_train, new_train)
text = text.replace(old_test, new_test)

predict_py.write_text(text)

print("Patchede 2D settings, Sequence settings og predict.py")